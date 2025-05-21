export http_proxy=
export https_proxy=
export HF_HOME=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TOKEN_BUDGET=1000
export TOKEN_SELECTOR_MODEL=SELF
export DROP_FUNC=token_selection_pe
export FRAME_SELECT_LAYER=60
accelerate launch --num_processes 4 --main_process_port 12344 denote_llava_72b.py
