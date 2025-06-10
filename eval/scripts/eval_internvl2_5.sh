export HF_HOME=./data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

use_token_selector=true
token_selector_path=models/flexselect_internvl2.5
token_selector_layer=15
tkn_budget=8256
drop_func_name=token_selection_qwen
token_selector_type=qwen2
pip install transformers==4.45.2
accelerate launch --num_processes 1 --main_process_port 12344 -m lmms_eval \
    --model internvl2 \
    --model_args "pretrained=models/InternVL2_5-8B,num_segments=512,modality=video,use_token_selector=$use_token_selector,token_selector_path=$token_selector_path,token_selector_layer=$token_selector_layer,tkn_budget=$tkn_budget,drop_func_name=$drop_func_name,token_selector_type=$token_selector_type" \
    --tasks videomme,mlvu_dev,longvideobench_val_v_sub,lvbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix internvl_7B \
    --output_path ./logs_internvl25
