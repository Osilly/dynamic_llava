import argparse
from functools import partial
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle

from llava.model.builder import load_pretrained_model

# from llava.model.dynamic_llava_builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)

from PIL import Image
import math
import torch.nn.functional as F

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score


nltk.data.path.append("llava/eval/bench_test/nltk_data")

special_text = {
    "ASSISTANT:": [319, 1799, 9047, 13566, 29901],
    "USER:": [11889, 29901],
    "</s>": [2],
}


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    torch.cuda.reset_max_memory_allocated()

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    model_memory = torch.cuda.max_memory_allocated()

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    total_num = 0
    sum_meteor = 0.0

    sum_total_token_length = 0
    sum_instruct_token_length = 0
    sum_output_token_length = 0
    sum_output_cache_length = 0
    # sum_max_prefill_memory = 0
    sum_max_decode_memory = 0

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question_list = line["question"]
        label_answer_list = line["answer"]

        try:
            total_num += 1
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
        except:
            print("skip!")
            continue

        round_meteor_list = []
        round_prompt_list = []

        total_token_length = 0
        instruct_token_length = 0
        output_token_length = 0
        output_cache_length = 0
        prefill_cache_length = 0

        max_decode_memory = 0

        for round, (question, label_answer) in enumerate(
            zip(question_list, label_answer_list)
        ):
            label_answer = label_answer.strip()
            split_label_answer = word_tokenize(label_answer.lower())
            if round == 0:
                qs = question.replace("<image>", "").strip()
                cur_prompt = qs

                if getattr(model.config, "mm_use_im_start_end", False):
                    qs = (
                        DEFAULT_IM_START_TOKEN
                        + DEFAULT_IMAGE_TOKEN
                        + DEFAULT_IM_END_TOKEN
                        + "\n"
                        + qs
                    )
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                cur_prompt = "<image>" + "\n" + cur_prompt

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = (
                    tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .cuda()
                )

                past_key_values = None
                round_prompt_list.append(prompt)
            else:
                # prompt = "</s>" + "USER:" + question + "ASSISTANT:"
                prompt = "USER:" + question + "ASSISTANT:"
                input_ids = tokenizer.encode(prompt, return_tensors="pt")[:, 1:].cuda()

                round_prompt_list.append(prompt)

            answer_ids = None
            for j in range(args.max_new_tokens):

                if j > 0:
                    images = None
                    image_sizes = None

                if j == 0:
                    if images is not None:
                        total_token_length += (
                            images.shape[-2] * images.shape[-1] // 14 // 14
                        )
                    total_token_length += input_ids.shape[-1]
                    instruct_token_length += input_ids.shape[-1]
                else:
                    total_token_length += input_ids.shape[-1]
                    output_token_length += input_ids.shape[-1]

                with torch.inference_mode():
                    outputs = model(
                        input_ids,
                        images=images,
                        image_sizes=image_sizes,
                        past_key_values=past_key_values,
                    )

                answer_id = torch.argmax(outputs.logits[0, -1:, :], dim=-1).unsqueeze(0)
                # print(answer_id)

                if answer_ids is not None:
                    answer_ids = torch.cat([answer_ids, answer_id], dim=1)
                else:
                    answer_ids = answer_id

                past_key_values = outputs.past_key_values

                if round == 0 and j == 0:
                    prefill_cache_length = past_key_values[-1][0].shape[-2]
                elif j == 0:
                    prefill_cache_length += input_ids.shape[-1]

                if j == 1:  # second output token
                    torch.cuda.reset_max_memory_allocated()
                    start_decode_memory = torch.cuda.max_memory_allocated()

                if (
                    j == args.max_new_tokens - 1
                    or answer_id[0, 0] == special_text["</s>"][0]
                ):
                    torch.cuda.reset_max_memory_allocated()
                    max_decode_memory += (
                        torch.cuda.max_memory_allocated() - start_decode_memory
                    )

                if round == len(label_answer_list) - 1 and (
                    j == args.max_new_tokens - 1
                    or answer_id[0, 0] == special_text["</s>"][0]
                ):
                    output_cache_length = (
                        past_key_values[-1][0].shape[-2] - prefill_cache_length
                    )

                if answer_id[0, 0] == special_text["</s>"][0]:
                    break

                input_ids = answer_id

            answer = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[
                0
            ].strip()
            split_answer = word_tokenize(answer)

            meteor = single_meteor_score(split_label_answer, split_answer)
            round_meteor_list.append(meteor)

        mean_round_meteor = sum(round_meteor_list) / len(round_meteor_list)

        sum_meteor += mean_round_meteor

        sum_total_token_length += total_token_length
        sum_instruct_token_length += instruct_token_length
        sum_output_token_length += output_token_length
        sum_output_cache_length += output_cache_length
        sum_max_decode_memory += max_decode_memory

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": str(round_prompt_list),
                    "answer": str(label_answer_list),
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                    "total_token_length": str(total_token_length),
                    "instruct_token_length": str(instruct_token_length),
                    "output_token_length": str(output_token_length),
                    "output_cache_length": str(output_cache_length),
                    "max_decode_memory": str(max_decode_memory),
                    "meteor": str(round_meteor_list),
                    "mean_round_meteor": str(mean_round_meteor),
                }
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.write(
        json.dumps(
            {
                "mean_total_token_length": str(sum_total_token_length / total_num),
                "mean_instruct_token_length": str(
                    sum_instruct_token_length / total_num
                ),
                "mean_output_token_length": str(sum_output_token_length / total_num),
                "mean_output_cache_length": str(sum_output_cache_length / total_num),
                "mean_max_decode_memory": str(sum_max_decode_memory / total_num),
                "mean_meteor": str(sum_meteor / total_num),
            }
        )
        + "\n"
    )
    ans_file.flush()

    ans_file.close()
    print("mean_meteor: " + str(sum_meteor / total_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)
