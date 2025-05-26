#!/bin/bash

# Set up the data folder
nnUNet_compile=false
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
# scl enable gcc-toolset-13
gcc -v
export http_proxy=http://9.131.113.25:11113
export https_proxy=http://9.131.113.25:11113

#scl enable gcc-toolset-13  bash 
IMAGE_FOLDER="/mnt/sh/mmvision/data/video/public/lmms-lab/LLaVA-Video-178K/data"
VIDEO_FOLDER="/mnt/sh/mmvision/data/video/public/lmms-lab/LLaVA-Video-178K/data"
DATA_YAML="/mnt/sh/mmvision/data/video/public/lmms-lab/output_rnd05.yaml" # e.g exp.yaml
ARNOLD_WORKER_GPU=8 
ARNOLD_WORKER_NUM=1  
ARNOLD_ID=0        
METIS_WORKER_0_HOST=localhost  
port_in_cmd=9527     

alias python=python3
############### Show Envs ####################
export WANDB_PROJECT=llava_video_token_selector
export WANDB_API_KEY=1f3a190546e4ad0be14534e5155dce501d07950b
export TOKENIZERS_PARALLELISM=1
nvidia-smi

################ Arnold Jobs ################

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

BASE_RUN_NAME="llava_max_frames_64_patchsize_729_all_19_qwen_token_selector_5_percent_mixed_up_l1_loss"

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="${BASE_RUN_NAME}"
PREV_STAGE_CHECKPOINT="/mnt/sh/mmvision/home/yunzhuzhang/huggingface/lmms-lab/LLaVA-Video-7B-Qwen2"
export HF_HOME="/mnt/sh/mmvision/home/yunzhuzhang/huggingface"
PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${WORLD_SIZE}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${port_in_cmd}" \
    train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="token_selector" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir /mnt/sh/mmvision/home/yunzhuzhang/csp/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 270000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory False \
    --lazy_preprocess True \
    --report_to wandb \
    --resume True \
    --torch_compile False \
    --dataloader_drop_last True \
    --frames_upbound 64 \
    --mm_newline_position grid \
    --add_time_instruction False \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --attn_implementation "sdpa" \
    --lora_enable False \
    --token_selector_path /mnt/sh/mmvision/home/yunzhuzhang/huggingface/lmms-lab/llava-onevision-qwen2-0.5b-ov \
    --verbose_logging True 2> ./logs/train_${MID_RUN_NAME}.log 
exit 0;
