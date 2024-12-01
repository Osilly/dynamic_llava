# dynamic_llava

The official pytorch implement of "Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification".

## Environment

Keep your workspace path is in the code, and then:

```
conda create -n llava python=3.10 -y
conda activate dynamic_llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Train

Dynamic-LLaVA is trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Prepare data

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

### Download open-resource checkpoints of MLLMs

To Dynamic-LLaVA, you can get the base checkpoints in [[LLaVA-1.5-7B]](https://huggingface.co/liuhaotian/llava-v1.5-7b) and  [[LLaVA-1.5-13B]](https://huggingface.co/liuhaotian/llava-v1.5-13b) for training Dynamic-LLaVA-7B and Dynamic-LLaVA-13B, respectively.

### Training Dynamic-LLaVA

We provide the training scripts for Dynamic-LLaVA-7B and Dynamic-LLaVA-13B, while you can find in `run`.

For training Dynamic-LLaVA-7B, you can directly conduct the shell `run/train_dynamic_llava_7b.sh`, the detailed command is as follows:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run/train_dynamic_llava_7b.sh
```

The details of `run/train_dynamic_llava_7b.sh` are as follows:

```shell
#!/bin/bash

deepspeed llava/train/train_sparse.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path [llava-v1.5-7b] \ # your open-resource checkpoint path
    --version v1 \
    --data_path [./playground/data/llava_v1_5_mix665k.json] \ # your instruct-following dataset
    --image_folder [./playground/data] \ # your instruct-following dataset
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --requires_image True \
    --bf16 True \
    --output_dir ./results/dynamic-llava-7b \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 40000 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --predictor_lr 2e-4 \
    --predictor_weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --mask_loss_weight 100.0 \
    --gumbel_start_tau 1.0 \
    --gumbel_end_tau 0.1 \
    --use_vision_predictor True \
    --use_text_predictor True \
    --use_output_text_predictor True \
    --use_instruct_predictor False \
    --vision_keep_rate 0.2 \
    --output_text_keep_rate 0.5 \
    --output_text_len_for_training 50 \

```



For training Dynamic-LLaVA-13B, you can directly conduct the shell `run/train_dynamic_llava_13b.sh`, the detailed command is as follows:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run/train_dynamic_llava_13b.sh
```

The details of `run/train_dynamic_llava_13b.sh` are as follows:

```shell
#!/bin/bash

deepspeed llava/train/train_sparse.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path [llava-v1.5-13b] \ # your open-resource checkpoint path
    --version v1 \
    --data_path [./playground/data/llava_v1_5_mix665k.json] \ # your instruct-following dataset
    --image_folder [./playground/data] \ # your instruct-following dataset
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --requires_image True \
    --bf16 True \
    --output_dir ./results/dynamic-llava-13b \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 40000 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --predictor_lr 2e-4 \
    --predictor_weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --mask_loss_weight 100.0 \
    --gumbel_start_tau 1.0 \
    --gumbel_end_tau 0.1 \
    --use_vision_predictor True \
    --use_text_predictor True \
    --use_output_text_predictor True \
    --use_instruct_predictor False \
    --vision_keep_rate 0.2 \
    --output_text_keep_rate 0.5 \
    --output_text_len_for_training 50 \

```



## Evaluation

We provide the evaluation scripts to evaluate the benchmarks.

### VQAv2

For evaluate Dynamic-LLaVA-7B in VQAv2 benchmark, you can directly conduct the shell `run/dynamic_eval/eval_for_vqav2.sh`, the detailed command is as follows:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run/dynamic_eval/eval_for_vqav2.sh
```

The details of `run/dynamic_eval/eval_for_vqav2.sh` are as follows:

```shell
#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="dynamic-llava-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.dynamic_eval.model_vqa_loader \
        --model-path [./results/dynamic-llava-7b] \ # your Dynamic-LLaVA checkpoint path
        --question-file [./playground/data/eval/vqav2/$SPLIT.jsonl] \ # your benchmark path
        --image-folder [./playground/data/eval/vqav2/test2015] \ # your benchmark path
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT \
--test_dir "./playground/data/eval/vqav2" \
--result_dir "./playground/data/eval/vqav2"


```

And then, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `./playground/data/eval/vqav2/answers_upload`.

### GQA

For evaluate Dynamic-LLaVA-7B in GQA benchmark, you can directly conduct the shell `run/dynamic_eval/eval_for_gqa.sh`, the detailed command is as follows:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run/dynamic_eval/eval_for_gqa.sh
```

The details of `run/dynamic_eval/eval_for_gqa.sh` are as follows:

```shell
#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="dynamic-llava-7b"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="[./playground/data/eval/gqa/data]" # your benchmark path

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.dynamic_eval.model_vqa_loader \
        --model-path [./results/dynamic-llava-7b] \ # your Dynamic-LLaVA checkpoint path
        --question-file [./playground/data/eval/gqa/$SPLIT.jsonl] \ # your benchmark path
        --image-folder [./playground/data/eval/gqa/data/images] \ # your benchmark path
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced

```

## Acknowledgements

This project is based on [LLaVA](https://github.com/haotian-liu/LLaVA). Thanks for their wonderful works.
