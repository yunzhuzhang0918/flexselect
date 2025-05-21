import torch
from typing import List, Optional, Tuple, Union, Dict
import torch.nn as nn
import math
import types
import os

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


        

def token_selection(self, hidden_states, vision_embedding_pos, position_embeddings, tkn_number = 1000):
    bsz, q_len, C = hidden_states.shape
    
    hidden_states_norm = self.input_layernorm(hidden_states)
    total_start = vision_embedding_pos[0][0][0]
    total_tokens = 0
    for start, length in vision_embedding_pos[0]:
        total_tokens += length
    text_len = q_len - total_start - total_tokens
    
    
    
    cos, sin = position_embeddings
    
    q_states = self.self_attn.q_proj(hidden_states_norm).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
    k_states = self.self_attn.k_proj(hidden_states_norm).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    k_states = repeat_kv(k_states, self.num_key_value_groups)

    q_states, k_states = apply_multimodal_rotary_pos_emb(q_states, k_states, cos, sin, self.mrope_section)
    
    text_states = q_states[:, :, total_start + total_tokens:, :]
    vision_states = k_states
    
    text_guide_score = (torch.matmul(text_states, vision_states.transpose(-1, -2)) / math.sqrt(self.head_dim))
    
    causal_mask = generate_causal_mask(text_len, self.num_heads).to(text_guide_score.device)

    text_guide_score[:, :, :, total_start + total_tokens:] = text_guide_score[:, :, :, total_start + total_tokens:] + causal_mask
    text_guide_score = torch.nn.functional.softmax(text_guide_score, dim=-1)[:, :, :, total_start:total_start + total_tokens]
    
    bsz, num_head, text_len, vision_len = text_guide_score.shape
    # return text_guide_score.max(dim=2)[0].transpose(2, 1)
    text_guide_score = text_guide_score.view(bsz, num_head * text_len, vision_len).max(dim=1)[0]        
    selected_tokens = torch.zeros(total_tokens, dtype=torch.bool, device=hidden_states.device)
    _, indexes = text_guide_score.topk(k=tkn_number, dim=1)
    
    selected_tokens[indexes] = True

    return text_guide_score, selected_tokens


def token_selection_mean(self, hidden_states, vision_embedding_pos, position_embeddings, tkn_number = 1000):
    bsz, q_len, C = hidden_states.shape
    
    hidden_states_norm = self.input_layernorm(hidden_states)
    total_start = vision_embedding_pos[0][0][0]
    total_tokens = 0
    for start, length in vision_embedding_pos[0]:
        total_tokens += length
    text_len = q_len - total_start - total_tokens
    
    cos, sin = position_embeddings
    
    q_states = self.self_attn.q_proj(hidden_states_norm).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
    k_states = self.self_attn.k_proj(hidden_states_norm).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    k_states = repeat_kv(k_states, self.num_key_value_groups)

    q_states, k_states = apply_multimodal_rotary_pos_emb(q_states, k_states, cos.to(q_states.device), sin.to(q_states.device), self.mrope_section)
    
    text_states = q_states[:, :, total_start + total_tokens:, :]
    vision_states = k_states
    
    text_guide_score = (torch.matmul(text_states, vision_states.transpose(-1, -2)) / math.sqrt(self.head_dim))
    
    causal_mask = generate_causal_mask(text_len, self.num_heads).to(text_guide_score.device)

    text_guide_score[:, :, :, total_start + total_tokens:] = text_guide_score[:, :, :, total_start + total_tokens:] + causal_mask
    text_guide_score = torch.nn.functional.softmax(text_guide_score, dim=-1)[:, :, :, total_start:total_start + total_tokens]
    
    bsz, num_head, text_len, vision_len = text_guide_score.shape
    # return text_guide_score.max(dim=2)[0].transpose(2, 1)
    text_guide_score = text_guide_score.view(bsz, num_head * text_len, vision_len).mean(dim=1)  
    selected_tokens = torch.zeros(total_tokens, dtype=torch.bool, device=hidden_states.device)
    _, indexes = text_guide_score.topk(k=tkn_number, dim=1)
    
    selected_tokens[indexes] = True

    return text_guide_score, selected_tokens

# def token_selection_pe_pool(self, hidden_states, vision_embedding_pos, tkn_number = 1000):
#     bsz, q_len, C = hidden_states.shape
    
#     hidden_states_norm_origin = self.input_layernorm(hidden_states)
#     total_start = vision_embedding_pos[0][0][0]
#     total_tokens = 0
#     for start, length in vision_embedding_pos[0]:
#         total_tokens += length
#     text_len = q_len - total_start - total_tokens
#     vision_tokens_norm_origin = hidden_states_norm_origin[:, total_start:total_start + total_tokens, :]
#     text_tokens = hidden_states_norm_origin[:, total_start + total_tokens:, :]
#     text_states = self.self_attn.q_proj(text_tokens).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
#     position_ids = torch.arange(total_start + total_tokens, q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
#     cos, sin = self.rotary_emb(text_states, position_ids)
#     text_states = apply_rotary_pos_emb_onedim(text_states, cos, sin)
#     bsz, num_head, text_len, C = text_states.shape

#     #need to be fixed
#     adaptive_max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
#     q_max = adaptive_max_pool(text_states.view(num_head * bsz, text_len, C).transpose(1, 2).float())
#     q_max = q_max.transpose(1, 2).view(bsz, num_head, 1, C).bfloat16()
    
#     vision_states = self.self_attn.k_proj(vision_tokens_norm_origin).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#     position_ids = torch.arange(total_start, total_start + total_tokens, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
#     cos, sin = self.rotary_emb(vision_states, position_ids)
#     vision_states = apply_rotary_pos_emb_onedim(vision_states, cos, sin)
#     vision_states = repeat_kv(vision_states, self.num_key_value_groups)
#     #need to be fixed
#     vision_states = torch.cat([vision_states, q_max], dim=2)
#     text_guide_score = (torch.matmul(text_states, vision_states.transpose(-1, -2)) / math.sqrt(self.head_dim))
#     text_guide_score = torch.nn.functional.softmax(text_guide_score, dim=-1)[:, :, :, :-1]
#     text_guide_score = text_guide_score.max(dim=1)[0].max(dim=1)[0]
#     selected_tokens = torch.zeros(total_tokens, dtype=torch.bool, device=hidden_states.device)
#     _, indexes = text_guide_score.topk(k=tkn_number, dim=1)
    
#     selected_tokens[indexes] = True
    
#     return text_guide_score, selected_tokens
       
        
# def build_vision_projector():
   
#     mlp_depth = 2
#     mm_hidden_size = 1152
#     hidden_size = 896
#     modules = [nn.Linear(mm_hidden_size, hidden_size)]
#     for _ in range(1, mlp_depth):
#         modules.append(nn.GELU())
#         modules.append(nn.Linear(hidden_size, hidden_size))
#     return nn.Sequential(*modules)

    

    

# class Qwen2ModelForSelector(Qwen2PreTrainedModel):
#     """
#     Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

#     Args:
#         config: Qwen2Config
#     """

#     def __init__(self, config: Qwen2Config):
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size

#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
#         self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
#         self.layers = nn.ModuleList(
#             [Qwen2DecoderLayer_Selector(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#         ) 
#         self.mm_projector = build_vision_projector()
#         # self.vision_tower = build_vision_tower(config, delay_load=False)
#         self._attn_implementation = config._attn_implementation
#         self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.rotary_emb = Qwen2RotaryEmbedding(config=config)
#         self.gradient_checkpointing = False
#         # self.mlp = nn.Sequential(
#         #     nn.Linear(14, 64),  # 第一层：28维 -> 64维
#         #     nn.ReLU(),          # 激活函数
#         #     nn.Linear(64, 1)    # 第二层：64维 -> 1维
#         # )
#         # Initialize weights and apply final processing
#         self.post_init()
    
#     def forward(
#         self,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         vision_embedding_pos: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = False,
#         tkn_number: int = 0,
#     ):

#         assert inputs_embeds is not None, "inputs embed for selector should not be empty!"        
#         position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
#         hidden_states = inputs_embeds
#         # create position embeddings to be shared across the decoder layers
#         position_embeddings = self.rotary_emb(hidden_states, position_ids)

#         # decoder layers
#         if output_attentions:
#             all_attentions = []
#             num_layers = len(self.layers)
#         else:
#             num_layers = -1
#             all_attentions = None
#         for i, decoder_layer in enumerate(self.layers[:num_layers]):
#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     decoder_layer.__call__,
#                     hidden_states,
#                     None,
#                     position_ids,
#                     None,
#                     output_attentions,
#                     False,
#                     None,
#                     position_embeddings,
#                     vision_embedding_pos,
#                     False
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=None,
#                     position_ids=position_ids,
#                     past_key_value=None,
#                     output_attentions=output_attentions,
#                     use_cache=False,
#                     cache_position=None,
#                     position_embeddings=position_embeddings,
#                     vision_embedding_pos=vision_embedding_pos,
#                     is_selector=False
#                 )
            
#             hidden_states = layer_outputs[0]
#             if output_attentions:
#                 all_attentions.append(layer_outputs[1])
#         if output_attentions:
#             text_guide_score = None
#         else:
#             text_guide_score = self.layers[-1].token_selection_pe(hidden_states, vision_embedding_pos, position_embeddings, tkn_number)
#             # text_guide_score = self.mlp(text_guide_score).squeeze(dim=-1)
#             # selected_tokens = torch.zeros(text_guide_score.shape[-1], dtype=torch.bool, device=hidden_states.device)
#             # _, indexes = text_guide_score.topk(k=1000, dim=1)
#             # selected_tokens[indexes] = True
#             # text_guide_score = (text_guide_score, selected_tokens)
#         return {
#             "text_guide_score": text_guide_score,
#             "attentions": all_attentions
#         }



        
        

