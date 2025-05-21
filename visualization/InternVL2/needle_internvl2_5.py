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
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode
from conversation import get_conv_template
from tqdm import tqdm
from transformers.cache_utils import Cache
from io import BytesIO
import base64
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
    ax.set_ylabel("Hit Ratio", labelpad=10)
    ax.set_xticks(np.arange(0, n_layers, 5))  # 每2层显示一个刻度
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
        format='png',
        transparent=False)  # 设置为True可获得透明背景
    plt.close()


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def apply_rotary_pos_emb_qwen(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=1):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    num_segments = min(num_segments, int(max_frame / fps))
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    timestamps = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        timestamps.append(frame_index / fps)
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, timestamps

def vision_sdpa_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    if self.qk_normalization:
        B_, H_, N_, D_ = q.shape
        q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

    x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.config.attention_dropout)
    x = rearrange(x, 'b h s d -> b s (h d)')

    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def eager_forward_qwen(
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
        query_states, key_states = apply_rotary_pos_emb_qwen(query_states, key_states, cos, sin)

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

def eager_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        "b q (h gs d) -> b q h gs d",
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # cos, sin = self.rotary_emb(value_states, position_ids)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # if past_key_value is not None:
    #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with
    # custom attn_mask, Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of
    # an inline conditional assignment in SDPA to support both torch.compile's dynamic shapes and full graph
    # options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = bool(causal_mask is None and q_len > 1)

    attn_output = torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=E1102
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)
    return attn_output, (query_states, key_states), past_key_value


def parse_args():
    parser = argparse.ArgumentParser(description="Remove keys from JSON file")
    parser.add_argument("-m", "--model_dir", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/OpenGVLab/InternVL2_5-8B")
    parser.add_argument("-d", "--data", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/Video-MME-Sample/videomme")
    parser.add_argument("--needle_data", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/lmms-lab/v_niah_needles")
    parser.add_argument("-o", "--output_dir", type=str, default="pics/20250317/question2_largemodel_500")
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
    # tokenizer, model, image_processor, _ = load_pretrained_model(args.model_dir, None, 'llava_qwen', device_map="auto", torch_dtype="bfloat16", overwrite_config=overwrite_config, attn_implementation="sdpa")
   
    model = AutoModel.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map = "cuda:0",  use_flash_attn=True).eval()
    model.img_context_token_id = 92546
    # model.language_model.model.layers = model.language_model.model.layers[:layer_idx]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    for x in model.vision_model.encoder.layers:
        x.attn.forward = types.MethodType(vision_sdpa_attn, x.attn)
    for x in model.language_model.model.layers:
        x.attention.forward = types.MethodType(eager_forward, x.attention)  # 将x重新绑定forward函数
        # x.self_attn.forward = types.MethodType(eager_forward_qwen, x.self_attn)  # 将x重新绑定forward函数

    
    similarities = []
    data = datasets.load_dataset(args.data)["test"]
    needles = datasets.load_dataset(args.needle_data)["test"]
    samples = data.filter(lambda x: x['duration'] == 'short')
    ratios = []

    for needle in needles:
        for doc in tqdm(list(samples)[:10]):
            message = [{"role": "system", "content": "You are a helpful assistant."}]
            video_path = doc["videoID"] + ".mp4"
            video_path = os.path.join(args.data_root, video_path)
            qs = needle["question"]
            answer = needle["answer"]
            print(qs)
            #process needle images
            
            video_pixel_values, num_patches_list, timestamps = load_video(video_path, num_segments=args.nframes, max_num=1)
            video_pixel_values = video_pixel_values.to(torch.bfloat16).cuda()
            T, C, h, w = video_pixel_values.shape
            needle_pixel_values = load_image(needle['image']).to(torch.bfloat16).cuda()
            insert_idx = torch.randint(5, T-5, (1,)).item()
            video_pixel_values[insert_idx] = needle_pixel_values
            
            video_prefix = "".join([f"<image>" for i in range(len(num_patches_list))])
            qs = video_prefix + qs
            
            template = get_conv_template("internvl2_5")
            template.system_message = "You are a helpful assistant."
            template.append_message(template.roles[0], qs)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            IMG_START_TOKEN='<img>'
            IMG_END_TOKEN='</img>'
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * 256 * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            model_inputs = tokenizer(query, return_tensors='pt')
            input_ids = model_inputs['input_ids'].to(video_pixel_values.device)
            attention_mask = model_inputs['attention_mask'].to(video_pixel_values.device)
            

            
            # print("input_ids", input_ids.size(), input_ids.cpu().numpy()[0].tolist()[:256])
            vid_id_start = torch.where(input_ids[0].eq(92544))[0].min().item()#92544
            vid_id_end = torch.where(input_ids[0].eq(92545))[0].max().item()#92545
            # vid_id_start = torch.where(input_ids[0].eq(151665))[0].min().item()#92544
            # vid_id_end = torch.where(input_ids[0].eq(151666))[0].max().item()#92545
           
            q_len = input_ids.shape[-1] - vid_id_end - 1
            
            with torch.inference_mode(), torch.no_grad():
                
                outputs = model(video_pixel_values, input_ids, attention_mask, image_flags=torch.ones_like(input_ids), output_attentions=True)
                
                logits = outputs["logits"][:, -1, :]
                pred_id = logits.argmax(dim=-1)
                pred_answer = tokenizer.decode(pred_id)
                print("answer:", answer, "pred_answer:", pred_answer)
                # import pdb; pdb.set_trace()
                """
                重写eager_forward到attention layer的outputs中，由于在flash attention或者sdpa中，attention_weights是None，因此我们将
                计算过的query_states, key_states，写入attention_weights，在外层通过index获取visual token或者text tokens
                通过attention weights拿到计算过的query 和 key states，原始还是只有flash attention或者sdpa
                """
                attn_weights_list = []
                for query_states, key_states in outputs.attentions:
                    _, num_head, seq_len, _ = query_states.shape
                    q = query_states[:, :, vid_id_end + 1:, :]
                    k = key_states
                    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(query_states.size(-1))
                    causal_mask = generate_causal_mask(q_len, num_head).to(q.device)
                    
                    attn_weights[:, :, :, vid_id_end + 1:] = attn_weights[:, :, :, vid_id_end + 1:] + causal_mask
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)[:, :, :, vid_id_start + 1:vid_id_end]
                    attn_weights_list.append(attn_weights)
                    del query_states
                    del key_states
        
            attentions = torch.cat(attn_weights_list, dim=0).float()  # layer, heads, seq, tgt
            layer_num, head_num, _, _ = attentions.shape
            # 两次最大值，取mean不行
            
            # attentions = attentions.max(dim=1)[0]
            # attentions = attentions.max(dim=1)[0]  # layer, tgt
            attentions = attentions.mean(dim=1)
            attentions = attentions.mean(dim=1)# layer, tgt
            top_index = attentions.topk(k=258, dim=-1)[1]
            tokens_per_frame = 258
            
            start_token = insert_idx * tokens_per_frame
            end_token = start_token + tokens_per_frame - 1
            in_range_mask = (top_index >= start_token) & (top_index <= end_token)
            hit_counts = in_range_mask.sum(dim=1).float()
            ratio = hit_counts / top_index.shape[1]
            ratios.append(ratio)
            del attentions
            del outputs
            torch.cuda.empty_cache()
    
    ratio = torch.stack(ratios, dim=0).mean(dim=0)
    # torch.save(ratio, "attn_map/NHR/internvl2_5/all.pt")
    plot_layer_ratio(ratio, "recall@k_internvl2_5.png")
    
if __name__ == '__main__':
    main()



