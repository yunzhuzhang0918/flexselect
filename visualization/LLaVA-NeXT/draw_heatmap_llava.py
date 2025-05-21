import argparse
import argparse
import math
import os
import types
from typing import List
from typing import Optional, Tuple, Union
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
from transformers.cache_utils import Cache


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
        return attn_output, (query_states, key_states), past_key_value


def parse_args():
    parser = argparse.ArgumentParser(description="Visualization Sample")
    parser.add_argument("-m", "--model_dir", type=str, default="/mnt/sh/mmvision/home/yunzhuzhang/huggingface/lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("-l", "--layer", type=int, default=None)
    parser.add_argument("-o", "--output_dir", type=str, default="pics/")
    parser.add_argument("-s", "--selection", type=str, default="heatmap")
    parser.add_argument("--ntokens", type=int, default=840)
    parser.add_argument("--nframes", type=int, default=32)
    args = parser.parse_args()
    return args
def process_attention_scores(images_list, attention_scores, output_dir, ntokens=1000):
    """
    Process attention scores and generate visualized images for each layer.
    
    Args:
        attention_scores (np.ndarray): Shape (num_layers, 32, 14, 14)
        images_list (list): List of 32 images, each of shape (384, 384, 3)
        output_dir (str): Directory to save output images
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    num_layers = attention_scores.shape[0]
    
    # Step 1: Resize images to 378x378
    num_images = len(images_list)
    resized_images = []
    for img in images_list:
        resized = cv2.resize(img, (378, 378))  # Resize to (378, 378, 3)
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)  # RGB转BGR
        resized_images.append(resized)
    images_np = np.array(resized_images)  # Shape (32, 378, 378, 3)
    
    # Step 2: Generate original concatenated image (for visualization)
    original_combined = np.hstack(images_np)  # Shape (378, 32*378, 3)
    cv2.imwrite(os.path.join(output_dir, "original_combined.png"), original_combined)
    
    # Step 3: Process each layer
    for layer_idx in range(num_layers):
        # Get scores for current layer (shape: 32, 14, 14)
        layer_scores = attention_scores[layer_idx]
        
        # Flatten scores and get top 1000 indices
        scores_flat = layer_scores.reshape(-1)
        topk_indices = np.argsort(scores_flat)[-ntokens:]  # Indices of top 1000 scores
        
        # Convert flat indices to (i, y, x) tuples
        top_patches = set()
        for idx in topk_indices:
            i = idx // (14 * 14)
            rem = idx % (14 * 14)
            y = rem // 14
            x = rem % 14
            top_patches.add((i.item(), y.item(), x.item()))
        
        # Process each image to mask non-top patches
        processed_images = []
        for i_img in range(num_images):
            img = images_np[i_img].copy()  # Shape (378, 378, 3)
            
            # Mask all patches not in top_patches
            for y_patch in range(14):
                for x_patch in range(14):
                    if (i_img, y_patch, x_patch) not in top_patches:
                        x_start = x_patch * 27
                        x_end = x_start + 27
                        y_start = y_patch * 27
                        y_end = y_start + 27
                        img[y_start:y_end, x_start:x_end] = 0  # Set to black
            
            processed_images.append(img)
        
        # Concatenate and save
        combined = np.hstack(processed_images)
        cv2.imwrite(os.path.join(output_dir, f"{layer_idx}.png"), combined)

def visualize_images_with_heatmaps(images: List[Union[Image.Image, np.ndarray]], scores: np.ndarray,
                                 output_dir: str = "visualization.png"):
    """
    Visualizes images with heatmaps highlighting important regions based on scores.

    Args:
        images: A list of N PIL.Image objects or numpy arrays, each with size (448, 448).
        scores: A NumPy array of shape (batch, N, 448, 448) representing the importance scores.
               Values should be in the range 0-1.
        output_dir: The path to save the generated image.
    """
    scores = torch.nn.functional.interpolate(scores, size=(384,384),mode='bilinear', align_corners=False)
    layer_num, num_images, height, width = scores.shape

    # Validate input dimensions
    if len(images) != num_images:
        raise ValueError("Number of images does not match the number of score maps per batch.")
    
    # Convert all images to PIL Image objects if they're numpy arrays
    pil_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            # Handle numpy array (assume it's in 0-255 range)
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img))
        else:
            pil_images.append(img)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_idx in range(layer_num):
        # Create a figure with exact dimensions to match the images
        fig = plt.figure(figsize=(num_images, 1), dpi=height)
        
        # Remove all margins and padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        # Create a grid of 1 row and num_images columns
        grid = plt.GridSpec(1, num_images, wspace=0, hspace=0)
        
        heatmaps = cm.get_cmap('jet')(scores[layer_idx])  # Use 'jet' colormap
        heatmaps = (heatmaps * 255).astype(np.uint8)
        
        for i in range(num_images):
            ax = plt.subplot(grid[i])
            # Get the image and ensure it's RGBA
            img = pil_images[i].convert("RGBA")
            heatmap = Image.fromarray(heatmaps[i]).convert("RGBA")
            
            # Blend the heatmap with the image
            blended_img = Image.blend(img, heatmap, alpha=0.4)
            
            # Display blended image
            ax.imshow(blended_img)
            ax.axis('off')
            
        # Save the figure without any padding
        batch_output_path = os.path.join(output_dir, f"{layer_idx}.png")
        fig.savefig(batch_output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print(f"Visualization saved to {output_dir}.")

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


def analyze_top(attention_map, topk=1000):
    layer_num = attention_map.size(0)
    if attention_map.ndim != 2:
        attention_map = attention_map.view(layer_num, -1)
    

    
    # Step 2: 提取每层的Top 500索引并转换为集合
    top_indices = []
    for i in range(layer_num):
        # 使用 topk 直接获取最大的500个索引
        _, indices = torch.topk(attention_map[i], k=topk)
        # 转换为CPU上的集合（如果张量在GPU上）
        top_indices.append(set(indices.cpu().numpy().tolist()))
    
    # Step 3: 计算所有层之间的交集比例
    similarity_matrix = torch.zeros((layer_num, layer_num))
    for i in range(layer_num):
        for j in range(layer_num):
            intersection = len(top_indices[i] & top_indices[j])
            similarity_matrix[i, j] = intersection / topk  # 比例基于500
    import pdb; pdb.set_trace()
    return similarity_matrix



from collections import defaultdict
import seaborn as sns

def plot_similarity_heatmap(similarity_matrix, layer_names=None, 
                           title="Layer Similarity Heatmap", 
                           cmap="viridis", figsize=(8, 6),
                           save_path="./smilarity.png", dpi=300):
    """
    可视化相似度矩阵为热力图并保存
    新增参数:
        save_path: 保存路径（如 "heatmap.png"），None表示不保存
        dpi: 输出分辨率（默认300）
    """
    # 转换为numpy数组
    if isinstance(similarity_matrix, torch.Tensor):
        matrix = similarity_matrix.numpy()
    else:
        matrix = np.array(similarity_matrix)
    
    # 创建画布
    plt.figure(figsize=figsize)
    
    # 生成热力图
    ax = sns.heatmap(
        matrix,
        annot=False,
        fmt=".2f",
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 10}
    )
    
    # 设置坐标轴标签
    # if layer_names is None:
    #     layer_names = [f"Layer {i}" for i in range(matrix.shape[0])]
    # ax.set_xticks(np.arange(len(layer_names)) + 0.5)
    # ax.set_yticklabels(layer_names, rotation=0)
    # ax.set_xticklabels(layer_names, rotation=45, ha='right')
    
    # 添加标题和颜色条标签
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Layers")
    plt.ylabel("Layers")
    cbar = ax.collections[0].colorbar
    cbar.set_label("Similarity Ratio", rotation=270, labelpad=15)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"热力图已保存至：{save_path}")
    
    plt.tight_layout()

def index_to_3d(i):
    frame = i // (14 * 15)
    pos_in_frame = i % (14 * 15)
    row = pos_in_frame // 15
    col = pos_in_frame % 15
    return (frame, row, col)

# 辅助函数：合并相邻token并过滤小区域
def merge_and_filter(indices, region_thre):
    # 按帧分组 {frame: set_of_(row,col)}
    frame_coords = defaultdict(set)
    for idx in indices:
        frame, row, col = index_to_3d(idx)
        frame_coords[frame].add((row, col))
    
    # 并查集合并相邻坐标
    valid_indices = []
    for frame, coords in frame_coords.items():
        coords = list(coords)
        parent = {}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # 初始化并查集
        for coord in coords:
            parent[coord] = coord
        
        # 合并上下左右相邻的坐标
        for i in range(len(coords)):
            r1, c1 = coords[i]
            for j in range(i+1, len(coords)):
                r2, c2 = coords[j]
                if (abs(r1 - r2) == 1 and c1 == c2) or (abs(c1 - c2) == 1 and r1 == r2):
                    if coords[i] in parent and coords[j] in parent:  # 避免KeyError
                        union(coords[i], coords[j])
        
        # 统计区域大小
        regions = defaultdict(list)
        for coord in coords:
            root = find(coord)
            regions[root].append(coord)
        
        # 保留区域大小 >=4 的token
        for region in regions.values():
            if len(region) >= region_thre:
                for (r, c) in region:
                    idx = frame * (14*15) + r * 15 + c
                    valid_indices.append(idx)
    
    return set(valid_indices)

def calculate_progressive_similarity(attention_maps,topk=1000, region_thre=16):
    """
    生成渐进式相似度矩阵
    矩阵元素[i,j]表示第i层与第j层（j>i）的交集占第i层集合的比例
    下三角区域（j<=i）的值固定为0
    """
    # 复用之前的辅助函数
    # 获取各层过滤后的token集合
    layer_num = attention_maps.shape[0]
    score = attention_maps
    layer_sets = []
    
    for i in range(layer_num):
        _, indices = torch.topk(score[i], k=topk)
        filtered = merge_and_filter(indices.cpu().numpy().tolist(), region_thre)
        layer_sets.append(filtered)
    
    # 计算相似度矩阵
    similarity_matrix = torch.zeros((layer_num, layer_num))
    
    for i in range(layer_num):
        set_i = layer_sets[i]
        size_i = len(set_i)
        
        if size_i == 0:
            continue  # 没有需要计算的交集
            
        # 维护累积交集
        cumulative_intersection = set(set_i)
        
        for j in range(i+1, layer_num):
            # 与当前层取交集
            cumulative_intersection &= layer_sets[j]
            
            # 计算当前累积交集比例
            current_ratio = len(cumulative_intersection) / size_i
            similarity_matrix[i, j] = current_ratio
            
            # 如果交集已为空，后续层无需计算
            if not cumulative_intersection:
                break
    plot_similarity_heatmap(similarity_matrix, save_path=f"./smilarity/{topk}_{region_thre}_progresss.png")
    import pdb; pdb.set_trace()
    return similarity_matrix   

def calculate_similarity_matrix(attention_maps, topk=500, region_thre = 8):
    """
    输入: attention_maps - 形状为 [layer_num, head_num, text_len, vision_len] 的torch张量
    输出: similarity_matrix - 形状为 [layer_num, layer_num] 的相似度矩阵
    """
    # 辅助函数：将一维索引转换为三维坐标 (帧号, 行号, 列号)
    

    # 主流程
    layer_num = attention_maps.shape[0]
    
    # Step 1: 提取每层的Top 500索引
    score = attention_maps  # [layer_num, vision_len]
    top_indices = []
    for i in range(layer_num):
        _, indices = torch.topk(score[i], k=topk)
        top_indices.append(indices.cpu().numpy().tolist())
    
    # Step 2: 合并相邻token并过滤小区域
    filtered_sets = [merge_and_filter(indices) for indices in top_indices]
    
    # Step 3: 计算相似度矩阵
    similarity_matrix = torch.zeros((layer_num, layer_num))
    for i in range(layer_num):
        for j in range(layer_num):
            intersection = len(filtered_sets[i] & filtered_sets[j])
            min_size = min(len(filtered_sets[i]), len(filtered_sets[j]))
            similarity_matrix[i, j] = intersection / min_size if min_size > 0 else 0.0
    
    plot_similarity_heatmap(similarity_matrix, save_path=f"./smilarity/{topk}_{region_thre}.png")
    import pdb; pdb.set_trace()
    return similarity_matrix

def draw_attention_shift(attention_scores):
    layer_names = ['Layer 0', 'Layer 3', 'Layer 14', 'Layer 19']
    # 创建子图
    fig, axes = plt.subplots(4, 1, figsize=(15, 8))
    plt.rcParams.update({'font.size': 16})
    # 绘制每个图层的热力图
    for i, (ax, scores, name) in enumerate(zip(axes, attention_scores, layer_names)):
        scores = scores.unsqueeze(dim=0).cpu()
        im = ax.imshow(scores, aspect='auto', cmap='viridis')
        ax.set_title(name)
        ax.set_xlabel('Position')
        ax.set_ylabel('Attention')
        fig.colorbar(im, ax=ax, label='Attention Score')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('attention_shift.png', dpi=300, bbox_inches='tight')

# def draw_attention_shift_line(attention_scores):
#     """
#     绘制注意力分数的曲线图并保存为图片
#     :param attention_scores: 包含多个注意力分数的列表，每个分数的形状为 (1, 6720)
#     :param layer_names: 每个图层的名称列表
#     """
#     # 设置全局字体大小
#     from matplotlib import font_manager
#     font_path = 'times.ttf'  # 替换为实际的字体文件路径
#     font_manager.fontManager.addfont(font_path)
    
#     plt.rcParams['font.family'] = 'Times New Roman'
#     layer_names = ['Layer 0', 'Layer 3', 'Layer 14', 'Layer 19']
#     plt.rcParams.update({'font.size': 18})

#     # 创建子图15, 2 * len(attention_scores)
#     fig, axes = plt.subplots(len(attention_scores), 1, figsize=(7, 5))

#     # 如果只有一个子图，axes 不是列表，需要转换为列表
#     if len(attention_scores) == 1:
#         axes = [axes]

#     # 绘制每个图层的曲线图
#     for i, (ax, scores, name) in enumerate(zip(axes, attention_scores, layer_names)):
#         # 确保数据是一维的
#         scores = scores.cpu()
#         if scores.ndim > 1:
#             scores = scores.squeeze()  # 将 (1, 6720) 转换为 (6720,)
#         ax.plot(scores, label=name)  # 绘制曲线图
#         ax.set_title(name, fontsize=18)  # 设置标题字体大小
#         # ax.set_xlabel('Position', fontsize=18)  # 设置 x 轴标签字体大小
#         ax.set_ylabel('Attention', fontsize=18)  # 设置 y 轴标签字体大小
#         ax.tick_params(axis='both', labelsize=18)  # 设置刻度标签字体大小
#         # ax.legend(fontsize=16)  # 设置图例字体大小

#     # 调整布局
#     plt.tight_layout()

#     # 保存图片
#     plt.savefig('attention_shift_xian.pdf', bbox_inches='tight', format="pdf")

#     # 显示图片
#     plt.show()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    overwrite_config = {}
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_dir, None, 'llava_qwen', device_map="auto", torch_dtype="bfloat16", overwrite_config=overwrite_config, attn_implementation="sdpa")
    

    layer_idx = args.layer
    # model.language_model.model.layers = model.language_model.model.layers[:layer_idx]
    
    for i, x in enumerate(model.model.layers):
        x.self_attn.forward = types.MethodType(eager_forward, x.self_attn)  # 将x重新绑定forward函数

    # 1, 16, 8449, 128
    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    video_path = "/mnt/csp/mmvision/data/videollm/benchmarks/videomme/compress_videos/fFjv93ACGo8.mp4"
    qs = "What's the color of the cup appeared in this video?"

    # video_path = "/mnt/csp/mmvision/data/videollm/benchmarks/videomme/compress_videos/6Cr_8tvvQ0k.mp4"
    # qs = "What is the animal in this video?"

    
    # video_path = "/mnt/csp/mmvision/data/videollm/benchmarks/videomme/compress_videos/gD0MvkDGAMg.mp4"
    # qs = "What is the first celestial object shown in the video?"

    # video_path = "/mnt/csp/mmvision/data/videollm/benchmarks/videomme/compress_videos/M69Sn3OERZo.mp4"
    # qs = "What animal saves the monkey in the video?"

    # qs = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\nWhat's the color of the cup appeared in this video?\nA. black\nB. white.\nC. pink.\nD. blue.\nThe best answer is:"
    # qs = "When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?\nA. Apples.\nB. Candles.\nC. Berries.\nD. The three kinds are of the same number."
    # qs = "What is the second class the boys are taking in the video?\nA. Math class.\nB. Science class.\nC. Music class.\nD. English class."
    video, frame_time, video_time = load_video(video_path, args.nframes, 1, False)
    
    images_list = transform_for_llava_video(video)
   
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].bfloat16().cuda()
    
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
        attn_weights_list = []
        # torch.cuda.empty_cache()
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
    
    # 取mean
    attentions = attentions.mean(dim=1)
    attentions = attentions.mean(dim=1)  # layer, tgt
    
    
    attentions = (attentions - attentions.min(dim=-1, keepdim=True)[0]) / (attentions.max(dim=-1, keepdim=True)[0] - attentions.min(dim=-1, keepdim=True)[0])
   
    # 14, 15是llava video的硬编码 
    attentions = attentions.view(layer_num, args.nframes, 14, 15)
    
    # attentions = torch.nn.functional.interpolate(attentions, size=(384,384),mode='bilinear', align_corners=False)
    
    attentions = attentions.detach().cpu()
    if args.selection == "heatmap":
        visualize_images_with_heatmaps(images_list, attentions, output_dir=args.output_dir)
    elif args.selection == "drop":
        process_attention_scores(images_list, attentions, output_dir=args.output_dir, ntokens=args.ntokens)
    


if __name__ == '__main__':
    main()



