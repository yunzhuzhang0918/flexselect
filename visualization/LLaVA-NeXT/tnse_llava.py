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
from sklearn.decomposition import PCA
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
from matplotlib.colors import ListedColormap
from matplotlib.colors import to_rgba
import random



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

def generate_class_labels(token_classes, num_tokens):
    """生成每个token的类别标签"""
    # 确定默认类别
    default_class = next((k for k, v in token_classes.items() if v == "others"), None)
    if not default_class:
        raise ValueError("必须包含一个值为'others'的类别作为默认类别")
    
    # 初始化所有token为默认类别
    labels = np.full(num_tokens, default_class, dtype=object)
    
    # 处理其他类别
    for cls_name, cls_range in token_classes.items():
        if cls_name == default_class:
            continue
        
        # 解析索引范围
        if isinstance(cls_range, (list, tuple)) and len(cls_range) == 2:
            start = max(0, int(cls_range[0]))
            end = min(num_tokens-1, int(cls_range[1]))
            labels[start:end+1] = cls_name  # 闭区间
        else:
            raise ValueError(f"无效的索引范围格式: {cls_range}")

    return labels

def plot_hidden_pca_8layers(hidden_states, output_dir, token_classes=None):
    """
    Visualize PCA of hidden states for 8 selected layers in a 2x4 grid
    with independent scales and multi-column legend below x-axis
    
    Plots layers: 0, 4, 8, 12, 16, 20, 24, 27 in 2 rows × 4 columns
    For layer 27, excludes the first token from visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    selected_layers = [0, 4, 8, 12, 16, 20, 24, 27]
    
    # Create figure with adjusted layout (2 rows, 4 columns + legend space)
    fig = plt.figure(figsize=(16, 8))  # Wider figure for 2x4 grid
    gs = fig.add_gridspec(3, 4, height_ratios=[24, 24, 2])  # 2 rows + legend row
    
    # Create axes for the plots in 2x4 grid
    axs = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 
         fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
         fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])],
    ]
    legend_ax = fig.add_subplot(gs[2, :])  # Full-width axis for legend
    
    # Flatten the axes list for easier iteration
    flat_axs = [ax for row in axs for ax in row]
    
    # Turn off individual axis labels and legend axis
    for ax in flat_axs:
        ax.set_xlabel('')
        ax.set_ylabel('')
    legend_ax.axis('off')
    
    # Prepare legend elements (only need to do this once)
    if token_classes:
        has_needle = "needle_vision_token" in token_classes
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      label='Text Tokens', markerfacecolor='red', markersize=8),
        ]
        if has_needle:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          label='Needle Visual Tokens', markerfacecolor='blue', markersize=8))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                      label='Non-needle Visual Tokens', markerfacecolor='green', markersize=8))
    
    # Process only selected layers
    for i, layer_id in enumerate(selected_layers):
        if layer_id >= len(hidden_states):
            continue
            
        hs = hidden_states[layer_id]
        
        # Convert data format
        if isinstance(hs, torch.Tensor):
            hs = hs.float().detach().cpu().numpy()
        hs = np.asarray(hs)

        # Check dimensions
        if hs.ndim == 1:
            hs = hs.reshape(1, -1)
        elif hs.ndim == 3:
            hs = hs.squeeze()
        if hs.shape[1] < 2:
            continue

        # PCA reduction
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(hs)

        
        labels = generate_class_labels(token_classes, hs.shape[0]) if token_classes else None

        # Prepare visualization data
        if token_classes:
            # Create masks for each class (using potentially modified labels)
            text_mask = (labels == "text_token")
            vision_mask = (labels == "vision_token")
            needle_mask = (labels == "needle_vision_token") if has_needle else None
            
            # Downsample vision tokens (50% random selection)
            if vision_mask.sum() > 0:
                vision_indices = np.where(vision_mask)[0]
                keep_indices = np.random.choice(
                    vision_indices, 
                    size=int(len(vision_indices)*0.5), 
                    replace=False
                )
                vision_mask = np.zeros_like(vision_mask)
                vision_mask[keep_indices] = True
            
            # Plot in specific order: vision -> text -> needle
            ax = flat_axs[i]
            
            # 1. Plot other vision tokens first (most transparent)
            if vision_mask.sum() > 0:
                ax.scatter(
                    reduced[vision_mask, 0], 
                    reduced[vision_mask, 1],
                    c='green',
                    alpha=0.6,
                    s=5,
                    edgecolor='white',
                    linewidth=0.5
                )
            
            # 2. Plot text tokens next
            if has_needle and needle_mask.sum() > 0:
                ax.scatter(
                    reduced[needle_mask, 0], 
                    reduced[needle_mask, 1],
                    c='blue',
                    alpha=0.5,
                    s=20,
                    edgecolor='white',
                    linewidth=0.5
                )
            if text_mask.sum() > 0:
                ax.scatter(
                    reduced[text_mask, 0], 
                    reduced[text_mask, 1],
                    c='red',
                    alpha=0.5,
                    s=20,
                    edgecolor='white',
                    linewidth=0.5
                )
            
        else:
            # Default plotting if no token classes provided
            ax.scatter(
                reduced[:, 0], 
                reduced[:, 1],
                c='#2c7bb6',
                alpha=0.6,
                s=20,
                edgecolor='white',
                linewidth=0.5
            )

        ax.set_title(f"Layer {layer_id}")
        
        # Add grid and set dynamic limits
        ax.grid(True, alpha=0.3)
        x_padding = (reduced[:, 0].max() - reduced[:, 0].min()) * 0.1
        y_padding = (reduced[:, 1].max() - reduced[:, 1].min()) * 0.1
        ax.set_xlim(reduced[:, 0].min()-x_padding, reduced[:, 0].max()+x_padding)
        ax.set_ylim(reduced[:, 1].min()-y_padding, reduced[:, 1].max()+y_padding)

    # Add unified legend in 3 columns
    if token_classes:
        legend = legend_ax.legend(
            handles=legend_elements,
            loc='center',
            ncol=3,  # Three columns
            frameon=True,
            framealpha=0.8,
            borderaxespad=0.5
        )
        legend.get_frame().set_facecolor('white')
    
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])  # Adjust for legend
    save_path = os.path.join(output_dir, "8layers_2x4_pca.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

from scipy.stats import gaussian_kde
from typing import List, Dict, Union


def create_token_classes(
    denote_kwargs: Dict[str, Union[List[int], str]],
    seq_len: int
) -> np.ndarray:
    """
    根据位置标记规则生成token类别数组
    
    参数：
    denote_kwargs : Dict
        {
            "needle_vision_token": [start, end],  # 特殊视觉token区间
            "text_token": [start, end],           # 文本token区间
            "vision_token": "others"              # 其他位置默认视觉token
        }
    seq_len : int
        序列总长度
    
    返回：
    np.ndarray
        形状为(seq_len,)的整数数组，取值：
        0 - 文本token
        1 - 普通视觉token
        2 - 特殊视觉token（needle）
    
    异常：
    - 当区间重叠或参数不合法时抛出ValueError
    """
    # 参数校验
    required_keys = {"needle_vision_token", "text_token", "vision_token"}
    if not required_keys.issubset(denote_kwargs.keys()):
        missing = required_keys - denote_kwargs.keys()
        raise ValueError(f"缺少必要字段: {missing}")
    
    # 初始化全为普通视觉token (1)
    token_classes = np.full(seq_len, fill_value=1, dtype=np.int32)
    
    def process_interval(key: str, label: int):
        """处理单个区间标记"""
        interval = denote_kwargs[key]
        if not isinstance(interval, list) or len(interval) != 2:
            raise TypeError(f"{key}的值应为[start, end]的列表")
        
        start, end = interval
        # 确保start <= end
        if start > end:
            start, end = end, start
        
        # 边界检查
        if start < 0 or end >= seq_len:
            raise ValueError(
                f"{key}区间[{start}, {end}]超出序列长度范围[0, {seq_len-1}]")
        
        # 标记区间
        token_classes[start:end] = label  # 包含两端点
        
    # 处理文本token（优先级最低）
    try:
        process_interval("text_token", 0)
    except Exception as e:
        raise ValueError("处理text_token时出错") from e
    
    # 处理特殊视觉token（优先级高于text）
    try:
        process_interval("needle_vision_token", 2)
    except Exception as e:
        raise ValueError("处理needle_vision_token时出错") from e
    
    # 检查区间重叠
    text_start, text_end = denote_kwargs["text_token"]
    needle_start, needle_end = denote_kwargs["needle_vision_token"]
    
    # 计算重叠区域
    overlap_start = max(text_start, needle_start)
    overlap_end = min(text_end, needle_end)
    
    if overlap_start <= overlap_end:
        raise ValueError(
            f"text_token[{text_start}-{text_end}] 与 "
            f"needle_vision_token[{needle_start}-{needle_end}] 存在重叠"
        )
    
    return token_classes

def analyze_hidden_pca(
    hidden_states: List[np.ndarray], 
    token_classes: np.ndarray,
    n_components: int = 2
) -> List[Dict[str, Union[np.ndarray, float]]]:
    """
    对每一层隐藏状态进行text/vision token的PCA投影分析
    
    参数：
    hidden_states : List[np.ndarray]
        各层的隐藏状态列表，每个数组形状为 (batch_size, seq_len, hidden_dim)
    token_classes : np.ndarray
        token类别标记数组，形状为 (total_tokens,)，0表示text，1表示vision
    n_components : int
        PCA降维后的维度，默认为2
    
    返回：
    List[Dict]
        每层分析结果的字典列表，包含：
        - text_proj: text token的投影坐标
        - vision_proj: vision token的投影坐标 
        - overlap_prob: vision样本在text分布中的平均概率
        - explained_variance_ratio: PCA解释方差比
    """
    results = []
    
  
    for layer_idx, layer_hidden in enumerate(hidden_states):
        # 展平当前层的隐藏状态 (batch, seq, dim) -> (N, dim)
        
        batch_size, seq_len, hidden_dim = layer_hidden.shape
        
        flattened_hidden = layer_hidden.reshape(-1, hidden_dim)
        flattened_hidden = flattened_hidden.float().cpu().numpy()
        # 获取当前层的token类别
        
        # 分割text/vision tokens
        
        text_mask = (token_classes == 0)
        needle_vision_mask = (token_classes == 2)
        
        text_hidden = flattened_hidden[text_mask]
        vision_hidden = flattened_hidden[needle_vision_mask]
        
        # 合并数据并训练PCA
        
        pca = PCA(n_components=n_components)

        pca.fit(flattened_hidden)
        # 投影
        text_proj = pca.transform(text_hidden)
        vision_proj = pca.transform(vision_hidden)
        
        # 计算分布重叠概率
        if text_proj.shape[0] > 1:  # 需要至少2个样本才能估计KDE
            kde = gaussian_kde(text_proj.T, bw_method='scott')
            vision_prob = kde(vision_proj.T).mean()
        else:
            vision_prob = np.nan
        
        # 保存结果
        
        print(layer_idx, vision_prob)
       
        results.append({
            "layer": layer_idx,
            "text_proj": text_proj,
            "vision_proj": vision_proj,
            "overlap_prob": float(vision_prob),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
        })
    
    return results



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
        return attn_output, attn_output, past_key_value


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
    needles = datasets.load_dataset(args.needle_data)["test"]
    samples = data.filter(lambda x: x['duration'] == 'short')
    ratios = []
    for needle in needles:
        for doc in tqdm(samples):
            video_path = doc["videoID"] + ".mp4"
            
            video_path = os.path.join(args.data_root, video_path)
            # qs = "What is the second class the boys are taking in the video?\nA. Math class.\nB. Science class.\nC. Music class.\nD. English class."
            video, frame_time, video_time = load_video(video_path, args.nframes, 1, False)
            
        
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].bfloat16().cuda()
            
            needle_images = needle['image']
            # if all(isinstance(item, str) for item in needle_images):
            #     needle_images = [Image.open(img) for img in needle_images]
            needle_image = image_processor.preprocess(needle_images, return_tensors="pt")["pixel_values"].bfloat16().cuda()
            needle_image_number = needle_image.shape[0]
            insert_idxs = random.sample(range(5, video.shape[0] - 5), needle_image_number)
            needle_image = needle_image.squeeze(0)
            video[insert_idxs] = needle_image
            
            qs = needle["question"]
            print(video_path, qs)
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
                
                """
                重写eager_forward到attention layer的outputs中，由于在flash attention或者sdpa中，attention_weights是None，因此我们将
                计算过的query_states, key_states，写入attention_weights，在外层通过index获取visual token或者text tokens
                通过attention weights拿到计算过的query 和 key states，原始还是只有flash attention或者sdpa
                """
                seq_len = outputs.attentions[0].shape[1]
                denote_kwargs ={}
                denote_kwargs["vision_token"] =  "others"
                attentions = torch.cat(outputs.attentions, dim=0)
                for insert_idx in insert_idxs:
                    denote_kwargs[f"needle_vision_token"] = [insert_idx * 210, (insert_idx + 1) * 210]
                denote_kwargs["text_token"] = [seq_len - 1 - q_len - vid_id, seq_len - 1 - vid_id]
                
                
               
                plot_hidden_pca_8layers(attentions[:, 14:], args.output_dir ,denote_kwargs)
                ## just visualize one sample, change this for more sample
                break
        break
    
    
if __name__ == '__main__':
    main()