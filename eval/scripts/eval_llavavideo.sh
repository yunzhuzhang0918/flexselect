export HF_HOME=./data

#For videoeval_pro you should provide open key here. For videomme, mlvu_dev, lvbench, longvideobench_val_v, ignore these env variant.
export OPENAI_API_URL=
export OPENAI_API_ID=
export OPENAI_API_KEY=

#Token selector releated.
use_token_selector=false # set to false to disable token selector
token_selector_path=./models/flexselect_llava_video # set to `self` will lead flexselect to use reference layer to select tokens. set to a path will lead flexselect work in lightweight mode. 
token_selector_layer=19
tkn_budget=6720
drop_func_name=token_selection
input_frames=512
pip install transformers==4.45.2
accelerate launch --num_processes 1 --main_process_port 12345  -m lmms_eval \
    --model llava_vid \
    --model_args "pretrained=models/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=$input_frames,mm_spatial_pool_mode=bilinear,use_token_selector=$use_token_selector,token_selector_path=$token_selector_path,token_selector_layer=$token_selector_layer,tkn_budget=$tkn_budget,drop_func_name=$drop_func_name" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_video_7B \
    --output_path ./logs
