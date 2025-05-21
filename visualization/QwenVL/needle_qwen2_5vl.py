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
from transformers import AutoModel, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
# from qwen2_5_vl import process_vision_info
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
    ax.set_ylabel("Recall@K", labelpad=10)
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

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

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output, (query_states, key_states), past_key_value


def parse_args():
    parser = argparse.ArgumentParser(description="Remove keys from JSON file")
    parser.add_argument("-m", "--model_dir", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("-d", "--data", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/Video-MME-Sample/videomme")
    parser.add_argument("--needle_data", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/lmms-lab/v_niah_needles")
    parser.add_argument("-o", "--output_dir", type=str, default="pics/20250317/question2_largemodel_500")
    parser.add_argument("-r", "--data_root", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/videomme/data")
    parser.add_argument("--nframes", type=int, default=64)
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
   
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_dir,torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="flash_attention_2").eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_dir, max_pixels=1605632, min_pixels=256 * 28 * 28)    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # model.language_model.model.layers = model.language_model.model.layers[:layer_idx]
    
    for i, x in enumerate(model.model.layers):
        x.self_attn.forward = types.MethodType(eager_forward, x.self_attn)  # 将x重新绑定forward函数

    
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
            print(qs)
            #process needle images
            base64_image = needle['image'].convert("RGB")
            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_string = base64_bytes.decode("utf-8")
            message.append({"role": "user", "content": [{"type": "video", "video": video_path, "max_pixels": 360*420, "max_frames": args.nframes}, {"type": "text", "text": qs}]})
            
            needle_message = [{"content": [{"type": "image","image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": 360*420}]}]
            text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
            _, video_inputs = process_vision_info([message])
        
            
            image_inputs, _ = process_vision_info([needle_message])
            t, _, h, w = video_inputs[0].shape
            
            for i in range(len(image_inputs)):
                image_inputs[i] = image_inputs[i].resize((h,w))
            video_inputs = processor(
                text=text,
                images=None,
                videos=video_inputs,
                fps=1.0,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
            needle_inputs = processor(
                text=text,
                images=image_inputs,
                videos=None,
                # fps=self.fps,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
            needle_pixel_value = needle_inputs["pixel_values"]
            C = needle_pixel_value.shape[-1]
            video_pixel_value = video_inputs["pixel_values_videos"].view(t // 2,  -1, C)
            
            insert_idx = torch.randint(5, video_pixel_value.shape[0]-5, (1,)).item()
            
            video_pixel_value[insert_idx] = needle_pixel_value

            video_inputs["pixel_values_videos"] = video_pixel_value.view(-1, C)
            
            video_width, video_height = video_inputs["video_grid_thw"][0][1].item(), video_inputs["video_grid_thw"][0][2].item()
            
            input_ids = video_inputs["input_ids"]
            
            # print("input_ids", input_ids.size(), input_ids.cpu().numpy()[0].tolist()[:256])
            vid_id_start = torch.where(input_ids[0].eq(151652))[0].item()
            vid_id_end = torch.where(input_ids[0].eq(151653))[0].item()
            q_len = input_ids.shape[-1] - vid_id_end - 1
            
            with torch.inference_mode(), torch.no_grad():
                outputs = model(**video_inputs, output_attentions=True)
                
                """
                重写eager_forward到attention layer的outputs中，由于在flash attention或者sdpa中，attention_weights是None，因此我们将
                计算过的query_states, key_states，写入attention_weights，在外层通过index获取visual token或者text tokens
                通过attention weights拿到计算过的query 和 key states，原始还是只有flash attention或者sdpa
                """
                attn_weights_list = []
                torch.cuda.empty_cache()
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
            
            attentions = attentions.max(dim=1)[0]
            attentions = attentions.max(dim=1)[0]  # layer, tgt
            top_index = attentions.topk(k=200, dim=-1)[1]
            tokens_per_frame = video_width * video_height // 4
            
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
    plot_layer_ratio(ratio, "ratio_qwen2_5_7B_vl.png")
    
if __name__ == '__main__':
    main()



