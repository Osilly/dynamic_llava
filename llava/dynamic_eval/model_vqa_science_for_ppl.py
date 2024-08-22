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

# from llava.model.builder import load_pretrained_model
from llava.model.dynamic_llava_builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)

from PIL import Image
import math


def forward_hook(forward_data, module, input, output):
    # print(f"Inside {module.__class__.__name__} forward")
    # forward_data["inputs"].append(input)
    forward_data["outputs"].append((output[0, -1, 0] > output[0, -1, 1]).item())


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
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    image_num = 0
    sum_self_ppl = 0.0
    sum_gpt4o_ppl = 0.0
    sum_masked_answer_token_rate = 0.0

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        # question = line["conversations"][0]
        # qs = question["value"].replace("<image>", "").strip()
        qs = "Describe the image in detail."
        cur_prompt = qs

        if "image" in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
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
        else:
            continue
            # images = None
            # image_sizes = None

        # if args.single_pred_prompt:
        #     qs = (
        #         qs
        #         + "\n"
        #         + "Answer with the option's letter from the given choices directly."
        #     )
        #     cur_prompt = (
        #         cur_prompt
        #         + "\n"
        #         + "Answer with the option's letter from the given choices directly."
        #     )

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

        forward_data = {"inputs": [], "outputs": []}
        hook_function = partial(forward_hook, forward_data)
        hook = model.model.output_text_score_predictor.register_forward_hook(
            hook_function
        )

        image_num += 1
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        answer = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[
            0
        ].strip()

        # self ppl
        logits = outputs.scores
        probs = [torch.softmax(logit, dim=-1) for logit in logits]
        log_probs = [torch.log(prob) for prob in probs]
        self_log_probs = [torch.max(log_prob) for log_prob in log_probs]
        self_ppls = [
            torch.exp(-self_log_prob).item() for self_log_prob in self_log_probs
        ]
        self_ppl = sum(self_ppls) / len(self_ppls)
        sum_self_ppl += self_ppl

        # output token decision
        answer_hard_decisions = forward_data["outputs"]
        masked_input_ids = outputs.sequences[:, 1:-1][
            torch.tensor(answer_hard_decisions).unsqueeze(0) == False
        ].unsqueeze(0)
        masked_answer = tokenizer.batch_decode(
            masked_input_ids,
            skip_special_tokens=False,
        )[0].strip()
        masked_answer_token_rate = masked_input_ids.shape[1] / (
            outputs.sequences.shape[1] - 2
        )
        sum_masked_answer_token_rate += masked_answer_token_rate

        # print(len(forward_data["outputs"]), forward_data["outputs"])
        hook.remove()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "answer": answer[4:-4],
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                    "answer_hard_decisions": str(answer_hard_decisions),
                    "answer_token_len": str(outputs.sequences.shape[1] - 2),
                    "masked_answer_token_len": str(masked_input_ids.shape[1]),
                    "masked_answer_token_rate": str(masked_answer_token_rate),
                    "masked_answer": masked_answer,
                    "self_ppl": str(self_ppl),
                    "gpt4o_ppl": str(0.0),
                }
            )
            + "\n"
        )
        ans_file.flush()
        # print(str(outputs.sequences.shape[1]), answer)
        # print(str(masked_input_ids.shape[1]), masked_answer)
        # print(str(self_ppl))

    ans_file.write(
        json.dumps(
            {
                "mean_self_ppl": str(sum_self_ppl / image_num),
                "mean_gpt4o_ppl": str(0.0),
                "mean_masked_answer_token_rate": str(
                    sum_masked_answer_token_rate / image_num
                ),
            }
        )
        + "\n"
    )
    ans_file.flush()

    ans_file.close()
    print("mean_self_ppl: " + str(sum_self_ppl / image_num))
    print("mean_gpt4o_ppl: " + str(sum_gpt4o_ppl / image_num))


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
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)