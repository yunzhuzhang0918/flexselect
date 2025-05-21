
export http_proxy=
export https_proxy=

export HF_HOME=

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=204800


use_token_selector=true
token_selector_path=qwen-vl-finetune/output/qwen2_5vl_small_token_selector
token_selector_layer=19
tkn_budget=2016
drop_func_name=token_selection


accelerate launch --num_processes 1 --main_process_port 12344 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,fps=2,use_flash_attention_2=True,max_num_frames=256,use_token_selector=$use_token_selector,token_selector_path=$token_selector_path,token_selector_layer=$token_selector_layer,tkn_budget=$tkn_budget,drop_func_name=$drop_func_name" \
    --tasks videomme_all \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_video_7B \
    --output_path ./logs_qwen2_5vl_external
