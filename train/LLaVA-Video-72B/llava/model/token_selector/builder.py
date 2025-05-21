import torch
import torch.nn as nn
import re

from transformers import Qwen2Config
from .qwen_selector import Qwen2ModelForSelector


def build_token_selector(config, delay_load=False, **kwargs):
    token_selector_path = config.token_selector_path
    token_selector_config = Qwen2Config.from_pretrained(token_selector_path)
    return Qwen2ModelForSelector(token_selector_config, **kwargs)