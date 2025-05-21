import argparse
import argparse
import math
import os
import types
from typing import List
from typing import Optional, Tuple
import datasets
import cv2
import matplotlib.cm as cm  # Import colormap module
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.conversation import SeparatorStyle, conv_templates
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import (
    tokenizer_image_token,
    process_images
)
from tqdm import tqdm
from transformers.cache_utils import Cache
import random

visual_start = 14
visual_end = 6734

plt.rcParams.update({
    'font.family': 'serif',          # 主字体
    'font.serif': ['DejaVu Serif'],  # Times新罗马
    'font.size': 12,                 # 正文字号
    'axes.labelsize': 14,            # 坐标轴标签字号
    'axes.titlesize': 14,            # 标题字号
    'xtick.labelsize': 12,           # X轴刻度字号
    'ytick.labelsize': 12,           # Y轴刻度字号
    'legend.fontsize': 12,           # 图例字号
    'figure.dpi': 300,               # 输出分辨率
    'figure.figsize': (8, 4),        # 图像尺寸（英寸）
    'axes.grid': True,               # 显示网格
    'grid.alpha': 0.3,               # 网格透明度
    'axes.linewidth': 0.8,           # 坐标轴线宽
    'lines.linewidth': 2,            # 折线线宽
    'savefig.bbox': 'tight',         # 保存时自动裁剪
    'savefig.pad_inches': 0.05       # 裁剪留白
})

def plot_layer_ratio(ratio, save_path="layer_analysis.pdf"):
    """
    绘制符合学术论文规范的层分析折线图
    
    参数：
    ratio : numpy.ndarray 或 torch.Tensor [28]
        各层的比例值
    save_path : str
        保存路径（推荐PDF格式）
    """
    # 转换数据格式
    if isinstance(ratio, torch.Tensor):
        ratio = ratio.cpu().numpy()
    ratio = np.asarray(ratio)

    # 创建画布
    fig, ax = plt.subplots()

    # 绘制折线（带数据点标记）
    n_layers= len(ratio)
    layers = np.arange(0, n_layers)  # 假设层编号从1开始
    line = ax.plot(layers, ratio, 
                  marker='o', markersize=6,
                  linewidth=2, alpha=0.8,
                  color='#2c7bb6',  # 学术常用蓝色
                  markeredgecolor='w',  # 白色边缘增强对比度
                  markeredgewidth=0.8)

    # 坐标轴设置
    ax.set_xlabel("Layer Index", labelpad=10)
    ax.set_ylabel("Recall@K", labelpad=10)
    ax.set_xticks(np.arange(0, n_layers, 2))  # 每2层显示一个刻度
    ax.set_xlim(0.5, 0.5 + n_layers)  # 留出边界空白
    
    # Y轴格式化为百分比
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # 添加趋势线（多项式拟合）
    if len(ratio) > 5:
        z = np.polyfit(layers, ratio, 3)
        p = np.poly1d(z)
        ax.plot(layers, p(layers), 
               linestyle='--', linewidth=1.5,
               color='#d7191c',  # 对比色红色
               alpha=0.7,
               label='Trend line')

    # 图例（如果有趋势线）
    if 'Trend line' in [line.get_label() for line in ax.lines]:
        ax.legend(frameon=True, framealpha=0.9)

    # 调整布局
    plt.tight_layout(pad=2.0)
    
    # 保存矢量图
    plt.savefig(save_path, 
        bbox_inches='tight', 
        pad_inches=0.05,  # 减少边缘留白
        transparent=False)  # 设置为True可获得透明背景
    plt.close()
def load_video(video_path, max_frames_num, fps, force_sample=False):
    # print(video_path)#/mnt/csp/mmvision/data/videollm/benchmarks/videomme/data/mAwgdX5VxGc.mp4
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()

    return spare_frames, frame_time, video_time


import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import \
        flash_attn_varlen_qkvpacked_func
    has_flash_attn = True
except:
    print('FlashAttention2 is not installed.')
    has_flash_attn = False


from transformers.image_transforms import (
    convert_to_rgb,
    rescale,
    resize
)

from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from functools import partial, reduce
def transform_for_llava_video(images):
    transforms = [
        convert_to_rgb,
        to_numpy_array,
        partial(resize, size=(384, 384), resample=PILImageResampling.BICUBIC, data_format=ChannelDimension.LAST),
        # partial(rescale, scale=1 / 255, data_format=ChannelDimension.FIRST),
    ]
    images = reduce(lambda x, f: [*map(f, x)], transforms, images)
    return images
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


    
def eager_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
       
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        # chose_layer = random.sample(range(6,19), 6)
       
        is_causal = True
        causal_mask = None
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output, (query_states, key_states), past_key_value


def parse_args():
    parser = argparse.ArgumentParser(description="Remove keys from JSON file")
    parser.add_argument("-m", "--model_dir", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("-d", "--data", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/Video-MME-Sample/videomme")
    parser.add_argument("--needle_data", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/lmms-lab/v_niah_needles")
    parser.add_argument("-o", "--output_dir", type=str, default="pics/")
    parser.add_argument("-r", "--data_root", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/videomme/data")
    parser.add_argument("--nframes", type=int, default=32)
    args = parser.parse_args()
    return args


def generate_causal_mask(seq_len, num_heads):
    """
    生成 causal mask，形状为 (1, num_heads, seq_len, seq_len)
    """
    # 生成下三角矩阵（对角线及其以下为 1，其余为 0）
    mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # 将上三角部分设置为 -inf
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    # 扩展维度到 (1, num_heads, seq_len, seq_len)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 先扩展为 (1, 1, seq_len, seq_len)
    mask = mask.expand(1, num_heads, seq_len, seq_len)  # 复制到 num_heads 维度
    return mask


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    overwrite_config = {}
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_dir, None, 'llava_qwen', device_map="auto", torch_dtype="bfloat16", overwrite_config=overwrite_config, attn_implementation="sdpa")
    

    # model.language_model.model.layers = model.language_model.model.layers[:layer_idx]
    
    for i, x in enumerate(model.model.layers):
        x.self_attn.forward = types.MethodType(eager_forward, x.self_attn)  # 将x重新绑定forward函数

    
    similarities = []
    data = datasets.load_dataset(args.data)["test"]
    ### transfrom to multi qa
    
    needles = datasets.load_dataset(args.needle_data)["test"]
    samples = data.filter(lambda x: x['duration'] == 'long')
    ratios = []
    total_sample = 0
    acc_sample = 0
    for needle in needles:
        for doc in tqdm(list(samples)[:10]):
            total_sample += 1
            video_path = doc["videoID"] + ".mp4"
            
            video_path = os.path.join(args.data_root, video_path)
            video, frame_time, video_time = load_video(video_path, args.nframes, 1, False)
            
        
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].bfloat16().cuda()
            needle_image = image_processor.preprocess(needle['image'], return_tensors="pt")["pixel_values"].bfloat16().cuda()
            answer = needle['answer']
            # answer = doc["answer"]
            insert_idx = torch.randint(5, video.shape[0]-5, (1,)).item()
            needle_image = needle_image.squeeze(0)
            video[insert_idx] = needle_image

            tokens_per_frame = 210
            start_token = insert_idx * tokens_per_frame
            end_token = start_token + tokens_per_frame - 1
            print(f"needle visual token: from {start_token} to {end_token}")
            # global visual_start, visual_end
            # visual_start = start_token
            # visual_end = end_token
            
            qs = needle["question"]
            # qs = doc["question"] + "\n".join(doc["options"]) + "Answer with the better option letter directly. The best answer:"
            # print(video_path, qs)
            _, _, video_width, video_height = video.shape
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates["qwen_1_5"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            

            # print("input_ids", input_ids.size(), input_ids.cpu().numpy()[0].tolist()[:256])
            vid_id = torch.where(input_ids[0].eq(IMAGE_TOKEN_INDEX))[0].item()
            q_len = len(input_ids[0]) - vid_id - 1
            
            with torch.inference_mode(), torch.no_grad():
                outputs = model(input_ids=input_ids, labels=None, images=[video], modalities="video", output_attentions=True)
                logits = outputs["logits"][:, -1, :]
                pred_id = logits.argmax(dim=-1)
                pred_answer = tokenizer.decode(pred_id)
                print("answer:", answer, "pred_answer:", pred_answer)
                if pred_answer == answer:
                    acc_sample += 1
                
                """
                重写eager_forward到attention layer的outputs中，由于在flash attention或者sdpa中，attention_weights是None，因此我们将
                计算过的query_states, key_states，写入attention_weights，在外层通过index获取visual token或者text tokens
                通过attention weights拿到计算过的query 和 key states，原始还是只有flash attention或者sdpa
                """
                attn_weights_list = []
                
                for query_states, key_states in outputs.attentions:
                    _, num_head, seq_len, _ = query_states.shape
                    q = query_states[:, :, -q_len:, :]
                    k = key_states
                    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(query_states.size(-1))
                    causal_mask = generate_causal_mask(q_len, num_head).to(q.device)
                    
                    attn_weights[:, :, :, -q_len:] = attn_weights[:, :, :, -q_len:] + causal_mask
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)[:, :, :, vid_id:-q_len]
                    attn_weights_list.append(attn_weights)
                    
        
            attentions = torch.cat(attn_weights_list, dim=0).float()  # layer, heads, seq, tgt
            layer_num, head_num, _, _ = attentions.shape
            # 两次最大值，取mean不行
            
            attentions = attentions.mean(dim=1)
            attentions = attentions.mean(dim=1)  # layer, tgt
            top_index = attentions.topk(k=210, dim=-1)[1]
            
            in_range_mask = (top_index >= start_token) & (top_index <= end_token)
            hit_counts = in_range_mask.sum(dim=1).float()
            ratio = hit_counts / top_index.shape[1]
            ratios.append(ratio)
            del attentions
            del outputs
            torch.cuda.empty_cache()
    
    ratio = torch.stack(ratios, dim=0).mean(dim=0)
    print("acc ratio:", acc_sample / total_sample)
    plot_layer_ratio(ratio, f"{args.output_dir}/recall@k_llava_video_7B.png")
    
    
if __name__ == '__main__':
    main()



