import torch
import torch.nn as nn
import re

from qwen2 import Qwen2Config
from .qwen_selector import Qwen2ModelForSelector, Qwen2ModelForSelector_FromSelf


def build_token_selector(config, token_selector_path=None, bigger_model=None, delay_load=False, **kwargs):
    if bigger_model is not None:
        
        return Qwen2ModelForSelector_FromSelf(config, bigger_model, **kwargs)
    else:
        token_selector = Qwen2ModelForSelector.from_pretrained(config.token_selector_path)
        return token_selector