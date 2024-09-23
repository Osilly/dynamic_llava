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
        model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
    )

    model_memory = torch.cuda.max_memory_allocated()
    print("model_memory: " + str(model_memory / (1024**3)) + "G")

    # questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    datas = json.load(open(os.path.expanduser(args.data_file), "r"))

    image_file_list = []
    question_round_list = [[] for _ in range(len(datas[0]["conversations"]) // 2)]
    label_answer_round_list = [[] for _ in range(len(datas[0]["conversations"]) // 2)]
    for data in datas:
        image_file_list.append(os.path.join(args.image_file, data["image"]))
        for i, line in enumerate(data["conversations"]):
            if line["from"] == "human":
                question_round_list[i // 2].append(line["value"])
            else:
                label_answer_round_list[i // 2].append(line["value"])

    images = None
    # image_sizes = []
    for image_file in image_file_list:
        image = Image.open(image_file)
        image_tensor = process_images([image], image_processor, model.config)[0]
        if images == None:
            images = image_tensor.unsqueeze(0).half().cuda()
        else:
            images = torch.cat([images, image_tensor.unsqueeze(0).half().cuda()])
        # image_sizes.append(image.size)

    if args.batch_size <= images.shape[0]:
        images = images[: args.batch_size]
    else:
        repeat_count = args.batch_size // images.shape[0]
        extra_count = args.batch_size % images.shape[0]

        repeated_images = images[: images.shape[0]].repeat(repeat_count, 1, 1, 1)

        extra_images = images[:extra_count]

        images = torch.cat([repeated_images, extra_images], dim=0)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    total_token_length = 0
    record = {
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "total_token_length": [],
        "kv_cache_length": [],
        "model_memory": model_memory,
        "max_memory": [],
        "without_model_memory": [],
    }
    with open(
        args.result_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(record, f, ensure_ascii=False, indent=4)

    # round_ppl_list = []
    for round, (question_batch_list, label_answer_batch_list) in enumerate(
        zip(question_round_list, label_answer_round_list)
    ):
        if round == 0:
            qs_batch_list = [
                question.replace("<image>", "").strip()
                for question in question_batch_list
            ]
            label_answer_batch_list = [
                label_answer.strip() for label_answer in label_answer_batch_list
            ]

            if getattr(model.config, "mm_use_im_start_end", False):
                qs_batch_list = [
                    (
                        DEFAULT_IM_START_TOKEN
                        + DEFAULT_IMAGE_TOKEN
                        + DEFAULT_IM_END_TOKEN
                        + "\n"
                        + qs
                    )
                    for qs in qs_batch_list
                ]
                # qs = (
                #     DEFAULT_IM_START_TOKEN
                #     + DEFAULT_IMAGE_TOKEN
                #     + DEFAULT_IM_END_TOKEN
                #     + "\n"
                #     + qs
                # )
            else:
                qs_batch_list = [
                    (DEFAULT_IMAGE_TOKEN + "\n" + qs) for qs in qs_batch_list
                ]
                # qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            input_ids_batch_list = []
            for qs in qs_batch_list:
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
                input_ids_batch_list.append(input_ids)

            max_input_ids_len = max(
                input_ids.shape[1] for input_ids in input_ids_batch_list
            )
            for j, cur_input_ids in enumerate(input_ids_batch_list):
                cur_len = cur_input_ids.shape[1]
                cur_input_ids = torch.cat(
                    [
                        torch.zeros(
                            (
                                cur_input_ids.shape[0],
                                max_input_ids_len - cur_len,
                            ),
                            dtype=cur_input_ids.dtype,
                            device=cur_input_ids.device,
                        ),
                        cur_input_ids,
                    ],
                    dim=1,
                )
                input_ids_batch_list[j] = cur_input_ids

            input_ids = torch.cat(input_ids_batch_list, dim=0)

            past_key_values = None
        else:
            input_ids_batch_list = []
            for question in question_batch_list:
                # prompt = "</s>" + "USER:" + question + "ASSISTANT:"
                prompt = "USER:" + question + "ASSISTANT:"
                input_ids = tokenizer.encode(prompt, return_tensors="pt")[:, 1:].cuda()
                input_ids_batch_list.append(input_ids)

            max_input_ids_len = max(
                input_ids.shape[1] for input_ids in input_ids_batch_list
            )
            for j, cur_input_ids in enumerate(input_ids_batch_list):
                cur_len = cur_input_ids.shape[1]
                cur_input_ids = torch.cat(
                    [
                        torch.zeros(
                            (
                                cur_input_ids.shape[0],
                                max_input_ids_len - cur_len,
                            ),
                            dtype=cur_input_ids.dtype,
                            device=cur_input_ids.device,
                        ),
                        cur_input_ids,
                    ],
                    dim=1,
                )
                input_ids_batch_list[j] = cur_input_ids

            input_ids = torch.cat(input_ids_batch_list, dim=0)

        label_ids_batch_list = []
        for label_answer in label_answer_batch_list:
            label_answer += "</s>"
            label_ids = (
                torch.tensor(tokenizer(label_answer).input_ids[1:])
                .to(dtype=input_ids.dtype, device=input_ids.device)
                .unsqueeze(0)
            )
            label_ids_batch_list.append(label_ids)

        max_label_ids_len = max(
            label_ids.shape[1] for label_ids in label_ids_batch_list
        )
        for j, cur_label_ids in enumerate(label_ids_batch_list):
            cur_len = cur_label_ids.shape[1]
            cur_label_ids = torch.cat(
                [
                    cur_label_ids,
                    torch.zeros(
                        (
                            cur_label_ids.shape[0],
                            max_label_ids_len - cur_len,
                        ),
                        dtype=cur_label_ids.dtype,
                        device=cur_label_ids.device,
                    ),
                ],
                dim=1,
            )
            label_ids_batch_list[j] = cur_label_ids

        label_ids = torch.cat(label_ids_batch_list, dim=0)

        # logits = []
        # labels = []

        if args.batch_size <= input_ids.shape[0]:
            input_ids = input_ids[: args.batch_size, :]
            label_ids = label_ids[: args.batch_size, :]
        else:
            repeat_count = args.batch_size // input_ids.shape[0]
            extra_count = args.batch_size % input_ids.shape[0]

            repeated_batches = input_ids[: input_ids.shape[0], :].repeat(
                repeat_count, 1
            )
            repeated_labels = label_ids[: input_ids.shape[0], :].repeat(repeat_count, 1)

            extra_samples = input_ids[:extra_count, :]
            extra_labels = label_ids[:extra_count, :]

            input_ids = torch.cat([repeated_batches, extra_samples], dim=0)
            label_ids = torch.cat([repeated_labels, extra_labels], dim=0)

        for j in range(label_ids.shape[1]):
            label_id = label_ids[:, j : j + 1]

            if j > 0:
                images = None
                # image_sizes = None
            with torch.inference_mode():
                if images is not None:
                    total_token_length += (
                        images.shape[-2] * images.shape[-1] // 14 // 14
                    )
                    total_token_length += input_ids.shape[-1] - 1
                else:
                    total_token_length += input_ids.shape[-1]
                outputs = model(
                    input_ids,
                    images=images,
                    # image_sizes=image_sizes,
                    past_key_values=past_key_values,
                )
            input_ids = label_id
            past_key_values = outputs.past_key_values
            total_cache_length = past_key_values[0][-1][0].shape[-2]

            # record
            max_memory = torch.cuda.max_memory_allocated()
            record["total_token_length"].append(total_token_length)
            record["kv_cache_length"].append(total_cache_length)
            record["max_memory"].append(max_memory)
            record["without_model_memory"].append(max_memory - model_memory)
            with open(
                args.result_file,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(record, f, ensure_ascii=False, indent=4)

            if total_token_length % 100 == 0:
                print("\n#--------------------------------------------------#")
                print("total_token_length: " + str(total_token_length))
                print("kv_cache_length: " + str(total_cache_length))
                max_memory = torch.cuda.max_memory_allocated()
                print("max_memory: " + str(max_memory / (1024**3)) + "G")
                print(
                    "without_model_memory (kv cache): "
                    + str((max_memory - model_memory) / (1024**3))
                    + "G"
                )
                print("#--------------------------------------------------#")

            # # ppl
            # logit = outputs.logits[:, -1:, :]
            # logits.append(logit)
            # labels.append(label_id)

        # logits = torch.cat(logits, dim=1).squeeze(0)
        # logits = logits.view(-1, logits.shape[-1])
        # labels = torch.cat(labels, dim=1).squeeze(0)
        # labels = labels.view(-1)
        # log_probs = F.cross_entropy(logits, labels)
        # ppls = torch.exp(log_probs).item()
        # print(ppls)
        # round_ppl_list.append(ppls)

    # mean_round_ppl = sum(round_ppl_list) / len(round_ppl_list)
    # print("mean_ppl: " + str(mean_round_ppl))

    print("\n#--------------------------------------------------#")
    print("total_token_length: " + str(total_token_length))
    print("kv_cache_length: " + str(total_cache_length))
    max_memory = torch.cuda.max_memory_allocated()
    print("max_memory: " + str(max_memory / (1024**3)) + "G")
    print(
        "without_model_memory (kv cache): "
        + str((max_memory - model_memory) / (1024**3))
        + "G"
    )
    print("#--------------------------------------------------#")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    # parser.add_argument("--num_beams", type=int, default=1)
    # parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
