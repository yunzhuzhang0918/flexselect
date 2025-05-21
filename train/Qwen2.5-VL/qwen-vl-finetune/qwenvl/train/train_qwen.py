# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, Qwen2_5_VLProcessor
from qwenvl.data.data_qwen import make_supervised_data_module
from transformers import AutoConfig
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, AutoModel
import copy

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.weight.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.weight.requires_grad = False

    if model_args.tune_token_selector:
        for n, p in model.token_selector.named_parameters():
            p.requires_grad = True
        for n, p in model.visual.merger_token_selector.named_parameters():
            p.requires_grad = False
    else:
        for n, p in model.token_selector.named_parameters():
            p.requires_grad = False
        for n, p in model.visual.merger_token_selector.named_parameters():
            p.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        
        config = Qwen2_5_VLConfig.from_pretrained(model_args.model_name_or_path)
        config.torch_dtype = (torch.bfloat16 if training_args.bf16 else None)
        config.token_selector_path = model_args.token_selector_path
        config.token_selector_config = AutoConfig.from_pretrained(model_args.token_selector_path)
        config.token_selector_config.token_selector_layer = model_args.token_selector_layer
        config.token_selector_config.drop_func_name = model_args.drop_func_name
        config.vision_config.token_selector_hidden_size = config.token_selector_config.hidden_size
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            config=config,
            torch_dtype = (torch.bfloat16 if training_args.bf16 else None)
        )
        
        
        data_args.image_processor = Qwen2_5_VLProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if model_args.token_selector_path:
        llm_weights = AutoModel.from_pretrained(model_args.token_selector_path).state_dict()
        merger_weights = copy.deepcopy(model.visual.merger).state_dict()
        new_merger_weights = {}
        new_merger_weights["ln_q.weight"] = merger_weights["ln_q.weight"].clone()
        new_merger_weights["mlp.0.bias"] = merger_weights["mlp.0.bias"].clone()
        new_merger_weights["mlp.0.weight"] = merger_weights["mlp.0.weight"].clone()
        new_merger_weights["mlp.2.bias"] = merger_weights["mlp.2.bias"][:896].clone()
        new_merger_weights["mlp.2.weight"] = merger_weights["mlp.2.weight"][:896, :].clone()
        del merger_weights
        incompatible_keys = model.token_selector.load_state_dict(llm_weights, strict=False)
        print(f"Loaded token selector weights from {model_args.token_selector_path}. Incompatible keys: {incompatible_keys}")
        incompatible_keys = model.visual.merger_token_selector.load_state_dict(new_merger_weights, strict=False)
        print(f"Loaded token selector weights from {model_args.token_selector_path}. Incompatible keys: {incompatible_keys}")

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    # set1 = [p.numel() for l, p in model.token_selector.named_parameters()]
    # set2 = [p.numel() for l, p in model.named_parameters() if p.requires_grad]
    rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
    rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    token_selector_model = trainer.model.token_selector
    token_selector_model.merger_token_selector = copy.deepcopy(trainer.model.visual.merger_token_selector)
    trainer.model = token_selector_model
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
