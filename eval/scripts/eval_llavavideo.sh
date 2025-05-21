export http_proxy=http://9.131.113.25:11113
export https_proxy=http://9.131.113.25:11113
export NO_PROXY=
export HF_HOME=/mnt/csp/mmvision/home/yunzhuzhang/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENAI_API_URL=
export OPENAI_API_ID=
export SOURCE=
export OPENAI_API_KEY=

use_token_selector=true
token_selector_path=/mnt/sh/mmvision/home/yunzhuzhang/eval_dev/models/llava_qwen_0.5b_5_percent_mixed_up
token_selector_layer=19
tkn_budget=6720
drop_func_name=token_selection
input_frames=512

accelerate launch --num_processes 8 --main_process_port 12345  -m lmms_eval \
    --model llava_vid \
    --model_args "pretrained=/mnt/sh/mmvision/home/yunzhuzhang/huggingface/lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=$input_frames,mm_spatial_pool_mode=bilinear,use_token_selector=$use_token_selector,token_selector_path=$token_selector_path,token_selector_layer=$token_selector_layer,tkn_budget=$tkn_budget,drop_func_name=$drop_func_name" \
    --tasks videomme_all \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_video_7B \
    --output_path ./logs
