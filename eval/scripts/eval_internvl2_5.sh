export http_proxy=
export https_proxy=
export HF_HOME=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

use_token_selector=true
token_selector_path="your selector path"
token_selector_layer=15
tkn_budget=2016
drop_func_name=token_selection_qwen
token_selector_type=qwen2
# /mnt/sh/mmvision/home/yunzhuzhang/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/internvl2_5_8b_dynamic_token_selector_epoch3_stage2
accelerate launch --num_processes 1 --main_process_port 12344 -m lmms_eval \
    --model internvl2 \
    --model_args "pretrained=OpenGVLab/InternVL2_5-8B,num_segments=512,modality=video,use_token_selector=$use_token_selector,token_selector_path=$token_selector_path,token_selector_layer=$token_selector_layer,tkn_budget=$tkn_budget,drop_func_name=$drop_func_name,token_selector_type=$token_selector_type" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_video_7B \
    --output_path ./logs_internvl25_3b
