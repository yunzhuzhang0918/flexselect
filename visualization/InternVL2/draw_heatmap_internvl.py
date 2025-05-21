import argparse
import argparse
import math
import os
import types
from typing import List
from typing import Optional, Tuple

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


def load_image(image, input_size=448, max_num=6):
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
    img_list = []
    for frame_index in frame_indices:
        origin_img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        timestamps.append(frame_index / fps)
        img = dynamic_preprocess(origin_img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
        img_list.append(origin_img)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, timestamps, img_list


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


def llm_decoder_attn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"`
        # once this is implemented.

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

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

    return attn_output, None, past_key_value


IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def eager_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # InternLM2FlashAttention2 attention does not support output_attentions
    if 'padding_mask' in kwargs:
        # warnings.warn(
        #     'Passing `padding_mask` is deprecated and will be removed in v4.37. '
        #     'Please make sure use `attention_mask` instead.`'
        # )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
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
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    repeat_key_states = repeat_kv(key_states, self.num_key_value_groups)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len
    )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.wo(attn_output)

    if not output_attentions:
        attn_weights = None

    """
    将flash attenton2或者sdpa计算过的query_states, repeat_key_states，写入attention_weights的placeholder，
    在外层获取
    """
    return attn_output, (query_states, repeat_key_states), past_key_value


def parse_args():
    parser = argparse.ArgumentParser(description="Remove keys from JSON file")
    parser.add_argument("-m", "--model_dir", type=str, default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/OpenGVLab/InternVL2_5-8B")
    parser.add_argument("-l", "--layer", type=int, default=None)
    parser.add_argument("-o", "--output_dir", type=str, default="pics/20250318/qs2")
    parser.add_argument("--ntokens", type=int, default=256)
    parser.add_argument("--nframes", type=int, default=32)
    args = parser.parse_args()
    return args


def visualize_images_with_heatmaps(images: List[Image.Image], scores: np.ndarray,
                                   output_dir: str = "visualization.png"):
    """
    Visualizes images with heatmaps highlighting important regions based on scores.

    Args:
        images: A list of N PIL.Image objects, each with size (448, 448).
        scores: A NumPy array of shape (batch, N, 448, 448) representing the importance scores for each pixel in each image.
               Values should be in the range 0-1.
        output_dir: The path to save the generated image.
    """

    batch_size, num_images, height, width = scores.shape

    # Validate input dimensions
    if len(images) != num_images:
        raise ValueError("Number of images does not match the number of score maps per batch.")
    for img in images:
        if img.size != (width, height):
            raise ValueError("Image size does not match the score map size.")

    for batch_idx in range(batch_size):
        # Create a figure and axes for the current batch
        fig_width_px = num_images * 448
        fig_height_px = 448
        dpi = 100  # Adjust DPI as needed.  Higher DPI = Higher Resolution
        fig_width_in = fig_width_px / dpi
        fig_height_in = fig_height_px / dpi

        fig, axes = plt.subplots(1, num_images, figsize=(fig_width_in, fig_height_in), dpi=dpi)
        if num_images == 1:
            axes = [axes]  # Make sure axes is always a list even if only 1 image

        # Adjust spacing to minimize gaps between images
        plt.subplots_adjust(wspace=0.01, hspace=0.01)  # Reduce horizontal and vertical spacing
        plt.margins(0, 0)  # Remove margins

        # Loop through images and heatmaps
        for i in range(num_images):
            # Overlay heatmap onto image
            img = images[i].copy().convert("RGBA")  # Ensure image is in RGBA format
            heatmap = cm.get_cmap('jet')(scores[batch_idx, i])  # Use 'jet' colormap
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = Image.fromarray(heatmap).resize((width, height), resample=Image.Resampling.BILINEAR).convert(
                "RGBA")  # Resize and ensure RGBA

            # Blend the heatmap with the image
            blended_img = Image.blend(img, heatmap, alpha=0.5)  # Adjust alpha for heatmap intensity

            # Display blended image
            axes[i].imshow(blended_img)
            axes[i].axis('off')  # Hide axes ticks and labels
        # Save the figure
        batch_output_path = os.path.join(output_dir, f"{batch_idx}.png")
        fig.savefig(batch_output_path, bbox_inches='tight', pad_inches=0)  # Remove extra padding
        plt.close(fig)  # close the figure to prevent memory leaks

    # print(f"Visualization saved to {output_path} (split by batch).")

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
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, use_flash_attn=True, torch_dtype="bfloat16")
    model.cuda()

    layer_idx = args.layer
    # model.language_model.model.layers = model.language_model.model.layers[:layer_idx]
    for i, x in enumerate(model.language_model.model.layers):
        x.attention.forward = types.MethodType(eager_forward, x.attention)  # 将x重新绑定forward函数

    # 1, 16, 8449, 128
    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

    

    video_path = "/mnt/csp/mmvision/data/videollm/benchmarks/videomme/compress_videos/fFjv93ACGo8.mp4"
    question = "What's the color of the cup appeared in this video?"
    pixel_values, num_patches_list, timestamps, img_list = load_video(video_path, num_segments=args.nframes, max_num=1)

    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])

    query = video_prefix

    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * args.ntokens + IMG_END_TOKEN

    for i in range(len(num_patches_list)):
        query = query.replace('<image>', image_tokens, 1)

    query = query + question

    conv = [{"role": "user", "content": query}]

    input_ids = tokenizer.apply_chat_template(conv)
    input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
    image_flags = torch.ones(pixel_values.size(0)).unsqueeze(1).cuda()

    # print("input_ids", input_ids.size(), input_ids.cpu().numpy()[0].tolist()[:256])

    with torch.inference_mode(), torch.no_grad():
        outputs = model(pixel_values, input_ids,
                        image_flags=image_flags, output_attentions=True, output_hidden_states=True)

        max_idx = torch.where(input_ids[0].eq(img_end_token_id))[0].max().item() + 2  # <\img>\n question
        is_visual_tokens = input_ids[0] == img_context_token_id
        """
        重写eager_forward到attention layer的outputs中，由于在flash attention或者sdpa中，attention_weights是None，因此我们将
        计算过的query_states, key_states，写入attention_weights，在外层通过index获取visual token或者text tokens
        通过attention weights拿到计算过的query 和 key states，原始还是只有flash attention或者sdpa
        """
        attn_weights_list = []
        for query_states, key_states in outputs.attentions:
            q = query_states[:, max_idx:].transpose(1, 2)
            _, num_head, q_len, _ = q.shape
            k = key_states#[:, :, is_visual_tokens]
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(query_states.size(-1))
            causal_mask = generate_causal_mask(q_len, num_head).to(q.device)
            attn_weights[:, :, :, max_idx:] += causal_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)[:, :, :, is_visual_tokens]
            attn_weights_list.append(attn_weights)

    attentions = torch.cat(attn_weights_list, dim=0).float()  # layer, heads, seq, tgt
    layer_num, head_num, _, _ = attentions.shape
    # 两次最大值，取mean不行
    attentions = attentions.mean(dim=1)
    attentions = attentions.mean(dim=1)  # layer, tgt

    # 归一化， layer, tgt
    attentions = (attentions - attentions.min(dim=-1, keepdim=True)[0]) / (
                attentions.max(dim=-1, keepdim=True)[0] - attentions.min(dim=-1, keepdim=True)[0])

    attentions = attentions.view(layer_num, 1, args.nframes, 16, 16)

    attentions = torch.nn.functional.interpolate(attentions, size=(args.nframes, 448, 448),
                                                 mode='trilinear', align_corners=False)
    attentions = attentions.squeeze(1).detach().cpu().numpy()

    images_list = [x.resize((448, 448)) for x in img_list]
    visualize_images_with_heatmaps(images_list, attentions, output_dir=args.output_dir)


if __name__ == '__main__':
    main()



