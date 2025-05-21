#!/bin/bash

# Distributed training configuration
export http_proxy=http://9.131.113.25:11113
export https_proxy=http://9.131.113.25:11113
export all_proxy=http://9.131.113.25:11113
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=1


FRAME_SELECT_LAYER=19
DROP_FUNC=token_selection_pe
# DeepSpeed configuration
deepspeed=./scripts/zero3.json
# scl enable gcc-toolset-13  bash
# Model configuration
llm=/mnt/sh/mmvision/home/yunzhuzhang/huggingface/Qwen/Qwen2.5-VL-7B-Instruct  # Using HuggingFace model ID
# llm=/mnt/sh/mmvision/home/yunzhuzhang/Qwen2.5-VL/qwen-vl-finetune/output/qwen2_5vl_small_token_selector_spearnmanloss_epoch3
# Training hyperparameters
lr=1e-5
batch_size=1
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=vprit_long

# Output configuration
run_name="qwen2_5vl_small_token_selector_dbg"
output_dir=./output/${run_name}
export WANDB_PROJECT="qwen2_5vl_token_selector"
export WANDB_API_KEY=1f3a190546e4ad0be14534e5155dce501d07950b
# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --tune_token_selector True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --video_max_frame_pixels 172480 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 270000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --video_max_frames 64 \
    --base_interval 1 \
    --run_name ${run_name} \
    --token_selector_layer 19 \
    --drop_func_name token_selection \
    --token_selector_path "/mnt/sh/mmvision/home/yunzhuzhang/huggingface/Qwen/Qwen2-0.5B-Instruct" \
    --report_to wandb"

#--save_steps 1000 \
# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

