# FlexSelect
The official repository for paper "FlexSelect: Flexible Token Selection for Efficient Long Video Understanding".


[`Webpage`]() ｜ [`Paper`]()

# Introduction
![Framework](assets/framework.png)

We present FlexSelect, a flexible and efficient token selection method that leverages cross-modal attention scores in VideoLLMs to identify query-relevant visual tokens. Our approach combines: (1) training-free attention-based token ranking, and (2) a lightweight selector for fast filtering.

# Todo:
- [x] Evaluation Code release of FlexSelect with LLaVA-Vide, Qwen2.5VL, InternVL2.5.
- [x] Training Code release of FlexSelect with LLaVA-Vide, Qwen2.5VL, InternVL2.5.
- [ ] Visualization code release of FlexSelect with LLaVA-Vide, Qwen2.5VL, InternVL2.5.
- [ ] Release the trained token selector.

# License

FlexSelect is released under the [`CC BY-NC-SA 4.0 license`](https://creativecommons.org/licenses/by-nc-sa/4.0/).

# Performance

We conduct experiments on three video LLMs (LLaVA-video, Qwen2.5VL, InternVL2.5) under for benchmarks: LongVideoBench, VideoMME, LVbench, MLVU.


# Data Preparation

All four used benchmarks can be downloaded from huggingface website: [`LongVideoBench`](https://huggingface.co/datasets/longvideobench/LongVideoBench), [`VideoMME`](https://huggingface.co/datasets/lmms-lab/Video-MME), [`MLVU`](https://huggingface.co/datasets/MLVU/MVLU), and [`LVBench`](https://huggingface.co/datasets/THUDM/LVBench).

# Pretrained Model

The pretrained model can be found in their respective repositories: [`LLaVA-Video-7B`](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2), [`LLaVA-Video-72B`](https://huggingface.co/lmms-lab/LLaVA-Video-72B-Qwen2), [`InternVL2.5-8B`](https://huggingface.co/OpenGVLab/InternVL2_5-8B), [`Qwen2.5VL-7B`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and [`Qwen2.5VL-72B`](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)..

# evaluation

FlexSelect works in two modes: training-free mode and lightweight mode.  We evaluate them using LMMS-Eval. We follow the environment installation guideline of [`LMMS-EVAL`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/README.md#installation).

run command 
```bash 
sh eval/scripts/training_free/eval_llavavideo.sh 
sh eval/scripts/training_free/eval_internvl2_5.sh 
sh eval/scripts/training_free/eval_qwenvl2_5.sh
```
to reproduce our result.

Here are explanations of variants in our eval scripts:

| Parameter                 | Type       | Options / Notes                                                                 | Default  |
|---------------------------|------------|---------------------------------------------------------------------------------|----------|
| **`use_token_selector`**  | `boolean`  | - `true`: Enable FlexSelect token selection<br>- `false`: Disable (standard eval) | `false`  |
| **`token_selector_path`** | `string`   | - `"self"`: Training-free mode<br>- `"path/to/model"`: Lightweight mode (需替换为实际路径) | `"self"` |
| **`token_selector_layer`**| `integer`  | 参考层编号（仅 **Training-free 模式** 生效）                                   | `-1`     |
| **`drop_func_name`**      | `string`   | 语义相关性分数计算方式：<br>- `"token_selection"`: 取头维+文本维平均值<br>- `"token_selection_argmax"`: 取头维+文本维 argmax | `"token_selection"` |
| **`tkn_budget`**          | `integer`  | 最大选择 token 数量（预算控制）                                                | `32`     |


Here are explanations of some commandline choice:

### 1. **Model Selection (`--model`)**
Specify the evaluation model with the following options:  

| Value          | Model Evaluated               |
|----------------|-------------------------------|
| `llava_vid`    | LLaVA-Video-7B                |
| `internvl2`    | InternVL2.5                   |
| `qwen2_5_vl`   | Qwen2.5VL                     |

### 2. **Task Selection (`--tasks`)**
| Value                  | Task Name          | Notes                              |
|------------------------|--------------------|------------------------------------|
| `videomme`             | Video-MME          | Standard video evaluation          |
| `mlvu_dev`             | MLVU               | Multi-language video understanding |
| `lvbench`              | LVBench            | Short-video benchmark              |
| `longvideobench_val_v` | LongVideoBench     | Default variant (e.g., for LLaVA)  |
| `longvideobench_val_i` | LongVideoBench     | **InternVL series only** (uses caption) |




# token selector training

FlexSelect trains 0.5B token selector for LLaVA-Video-7B, Qwen2.5VL-7B and InternVL2.5-8B.

We follow the environment installation guideline of corresponding project to construct training environment:

- LLaVA-Video: https://github.com/LLaVA-VL/LLaVA-NeXT?tab=readme-ov-file#2-install-the-inference-package
- Qwen2.5VL: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md
- InternVL2.5: https://internvl.readthedocs.io/en/latest/internvl2.5/finetune.html

```bash
# Step 1: Train LLaVA-Video selector
cd train/LLaVA-Video && sh scripts/train_selector.sh

# Step 2: Finetune Qwen2.5-VL
cd train/Qwen2.5-VL/qwen-vl-finetune && sh scripts/sft_7b.sh

# Step 3: Finetune InternVL (dynamic resolution)
cd train/InternVL/internvl_chat && sh shell/internvl2.5/2nd_finetune/internvl2_5_8b_dynamic_res_2nd_finetune_full.sh
```

The training data can be found at:...
We will release our trained token selector model.



# Acknowledgement

This repository is built upon [`LMMS-EVAL`](https://github.com/EvolvingLMMs-Lab/lmms-eval), [`LLaVA-Video`](https://github.com/LLaVA-VL/LLaVA-NeXT), [`InternVL2.5`](https://github.com/OpenGVLab/InternVL), and [`Qwen2.5VL`](https://github.com/QwenLM/Qwen2.5-VL). Thanks for those well-organized codebases.

# Citation

