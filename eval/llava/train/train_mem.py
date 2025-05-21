import sys
sys.path.append("/mnt/sh/mmvision/home/yunzhuzhang/LLaVA-NeXT-NPU/tome")
from llava.train.train import train
import torch
import torch_npu
# import deepspeed
# import deepspeed_npu
from torch_npu.contrib import transfer_to_npu
if __name__ == "__main__":
    
    train()
