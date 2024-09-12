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
import torch.nn.functional as F

special_text = {
    "ASSISTANT:": [319, 1799, 9047, 13566, 29901],
    "USER:": [11889, 29901],
    "</s>": [2],
}


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

    # questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    total_num = 0
    sum_ppl = 0.0
    sum_masked_answer_token_rate = 0.0

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

        round_ppl_list = []
        round_prompt_list = []
        round_answer_hard_decisions_list = []
        round_masked_answer_token_rate = []
        for round, (question, label_answer) in enumerate(
            zip(question_list, label_answer_list)
        ):
            if round == 0:
                qs = question.replace("<image>", "").strip()
                cur_prompt = qs
                label_answer = label_answer.strip()

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
                prompt = "</s>" + "USER:" + question + "ASSISTANT:"
                input_ids = tokenizer.encode(prompt, return_tensors="pt")[:, 1:].cuda()

                round_prompt_list.append(prompt)

            label_ids = tokenizer(label_answer).input_ids[1:]
            answer_hard_decisions = None
            logits = []
            labels = []
            # masked_input_ids = None
            for j, label_id in enumerate(label_ids):
                label_id = (
                    torch.tensor([label_id])
                    .to(dtype=input_ids.dtype, device=input_ids.device)
                    .unsqueeze(0)
                )
                # if j == 0:
                #     input_ids = torch.cat([input_ids, label_id], dim=1)
                #     continue
                forward_data = {"inputs": [], "outputs": []}
                hook_function = partial(forward_hook, forward_data)
                hook = model.model.output_text_score_predictor.register_forward_hook(
                    hook_function
                )

                if j > 0:
                    images = None
                    image_sizes = None
                with torch.inference_mode():
                    # outputs = model.generate(
                    #     input_ids,
                    #     images=images,
                    #     image_sizes=image_sizes,
                    #     do_sample=True if args.temperature > 0 else False,
                    #     temperature=args.temperature,
                    #     max_new_tokens=1,
                    #     use_cache=True,
                    #     output_scores=True,
                    #     return_dict_in_generate=True,
                    #     past_key_values=past_key_values,
                    # )
                    outputs = model(
                        input_ids,
                        images=images,
                        image_sizes=image_sizes,
                        past_key_values=past_key_values,
                    )
                # input_ids = torch.cat([input_ids, label_id.unsqueeze(0)], dim=1)
                input_ids = label_id
                past_key_values = outputs.past_key_values

                # answer = tokenizer.batch_decode(
                #     outputs.sequences[:, -1], skip_special_tokens=False
                # )[0].strip()
                # print(
                #     answer,
                #     tokenizer.batch_decode(label_id, skip_special_tokens=False)[0].strip(),
                # )
                # print(past_key_values[3][0].shape)
                # print(forward_data["outputs"])

                # ppl
                # logits = outputs.scores[0]
                logit = outputs.logits[:, -1:, :]
                logits.append(logit)
                labels.append(label_id)

                # output token decision
                if answer_hard_decisions is not None:
                    answer_hard_decisions.append(forward_data["outputs"][0])
                else:
                    answer_hard_decisions = forward_data["outputs"]
                # if masked_input_ids is not None:
                #     masked_input_ids = torch.cat(
                #         [
                #             masked_input_ids,
                #             outputs.sequences[:, -1:][
                #                 torch.tensor(answer_hard_decisions).unsqueeze(0) == False
                #             ].unsqueeze(0),
                #         ],
                #         dim=1,
                #     )
                # else:
                #     masked_input_ids = outputs.sequences[:, -1:][
                #         torch.tensor(answer_hard_decisions).unsqueeze(0) == False
                #     ].unsqueeze(0)

                # print(len(forward_data["outputs"]), forward_data["outputs"])
                hook.remove()

            logits = torch.cat(logits, dim=1).squeeze(0)
            labels = torch.cat(labels, dim=1).squeeze(0)
            log_probs = F.cross_entropy(logits, labels)
            ppls = torch.exp(log_probs).item()
            round_ppl_list.append(ppls)
            # sum_ppl += ppls

            # masked_answer = tokenizer.batch_decode(
            #     masked_input_ids,
            #     skip_special_tokens=False,
            # )[0].strip()
            # masked_answer_token_rate = masked_input_ids.shape[1] / len(label_ids)
            round_answer_hard_decisions_list.append(answer_hard_decisions)
            masked_answer_token_rate = (
                len(answer_hard_decisions) - sum(answer_hard_decisions)
            ) / len(answer_hard_decisions)
            round_masked_answer_token_rate.append(masked_answer_token_rate)
            # sum_masked_answer_token_rate += masked_answer_token_rate

        mean_round_ppl = sum(round_ppl_list) / len(round_ppl_list)
        mean_round_masked_answer_token_rate = sum(round_masked_answer_token_rate) / len(
            round_masked_answer_token_rate
        )

        sum_ppl += mean_round_ppl
        sum_masked_answer_token_rate += mean_round_masked_answer_token_rate

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
                    "answer_hard_decisions": str(round_answer_hard_decisions_list),
                    # "answer_token_len": str(len(answer_hard_decisions) + 1),
                    # "masked_answer_token_len": str(masked_input_ids.shape[1]),
                    "masked_answer_token_rate": str(round_masked_answer_token_rate),
                    # "masked_answer": masked_answer,
                    "ppl": str(round_ppl_list),
                    "mean_round_masked_answer_token_rate": str(
                        mean_round_masked_answer_token_rate
                    ),
                    "mean_round_ppl": str(mean_round_ppl),
                }
            )
            + "\n"
        )
        ans_file.flush()
        # print(str(outputs.sequences.shape[1]), answer)
        # print(str(masked_input_ids.shape[1]), masked_answer)
        # print(str(ppl))

    ans_file.write(
        json.dumps(
            {
                "mean_ppl": str(sum_ppl / total_num),
                "mean_masked_answer_token_rate": str(
                    sum_masked_answer_token_rate / total_num
                ),
            }
        )
        + "\n"
    )
    ans_file.flush()

    ans_file.close()
    print("mean_ppl: " + str(sum_ppl / total_num))


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
