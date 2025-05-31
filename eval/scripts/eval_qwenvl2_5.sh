export HF_HOME=./data

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=204800


use_token_selector=true
token_selector_path=models/flexselect_qwen2.5vl
token_selector_layer=20
tkn_budget=7040
drop_func_name=token_selection

pip install transformers==4.49.0
accelerate launch --num_processes 1 --main_process_port 12344 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "pretrained=models/Qwen2.5-VL-7B-Instruct,fps=2,use_flash_attention_2=True,max_num_frames=128,use_token_selector=$use_token_selector,token_selector_path=$token_selector_path,token_selector_layer=$token_selector_layer,tkn_budget=$tkn_budget,drop_func_name=$drop_func_name" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_video_7B \
    --output_path ./logs_qwen2_5vl_external
