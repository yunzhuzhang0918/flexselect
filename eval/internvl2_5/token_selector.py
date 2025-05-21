from .modeling_internlm2 import InternLM2DecoderLayer, InternLM2PreTrainedModel
from .configuration_internlm2 import InternLM2Config
import torch
from transformers import Qwen2PreTrainedModel
from typing import List, Optional, Tuple, Union, Dict
import torch.nn as nn
import math
import types
import os
from einops import rearrange
from transformers import Qwen2ForCausalLM, PreTrainedModel, AutoConfig
from .modeling_internlm2 import InternLM2ForCausalLM

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
def apply_rotary_pos_emb_onedim(q, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

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


       
        

def token_selection(self, hidden_states, vision_embedding_pos, tkn_number = 1000):
    bsz, q_len, C = hidden_states.shape
    
    hidden_states_norm_origin = self.attention_norm(hidden_states)
    total_start = vision_embedding_pos[0][0][0]
    total_tokens = 0
    for start, length in vision_embedding_pos[0]:
        total_tokens += length
    text_len = q_len - total_start - total_tokens



    bsz, q_len, _ = hidden_states.size()

    qkv_states = self.attention.wqkv(hidden_states_norm_origin)

    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.attention.num_key_value_groups,
        d=self.attention.head_dim,
    )

    query_states = qkv_states[..., : self.attention.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    kv_seq_len = key_states.shape[-2]

    position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

    cos, sin = self.attention.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.attention.num_key_value_groups)


    text_states = query_states[:, :, total_start + total_tokens:, :]
    vision_states = key_states

    text_guide_score = (torch.matmul(text_states, vision_states.transpose(-1, -2)) / math.sqrt(self.attention.head_dim))
    
    causal_mask = generate_causal_mask(text_len, self.attention.num_heads).to(text_guide_score.device)

    text_guide_score[:, :, :, total_start + total_tokens:] = text_guide_score[:, :, :, total_start + total_tokens:] + causal_mask
    text_guide_score = torch.nn.functional.softmax(text_guide_score, dim=-1)[:, :, :, total_start:total_start + total_tokens]
    
    bsz, num_head, text_len, vision_len = text_guide_score.shape
    text_guide_score = text_guide_score.view(bsz, num_head * text_len, vision_len).max(dim=1)[0]        
    selected_tokens = torch.zeros(total_tokens, dtype=torch.bool, device=hidden_states.device)
    _, indexes = text_guide_score.topk(k=tkn_number, dim=1)
    
    selected_tokens[indexes] = True

    return text_guide_score, selected_tokens

def token_selection_qwen(self, hidden_states, vision_embedding_pos, tkn_number = 1000):
    bsz, q_len, C = hidden_states.shape
    
    hidden_states_norm_origin = self.input_layernorm(hidden_states)
    total_start = vision_embedding_pos[0][0][0]
    total_tokens = 0
    for start, length in vision_embedding_pos[0]:
        total_tokens += length
    text_len = q_len - total_start - total_tokens
    vision_tokens_norm_origin = hidden_states_norm_origin#[:, start:total_start + total_tokens, :]
    text_tokens = hidden_states_norm_origin[:, total_start + total_tokens:, :]
    text_states = self.self_attn.q_proj(text_tokens).view(bsz, -1, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
    position_ids = torch.arange(total_start + total_tokens, q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    cos, sin = self.self_attn.rotary_emb(text_states, position_ids)
    text_states = apply_rotary_pos_emb_onedim(text_states, cos, sin)
    
    vision_states = self.self_attn.k_proj(vision_tokens_norm_origin).view(bsz, -1, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)
    position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    cos, sin = self.self_attn.rotary_emb(vision_states, position_ids)
    vision_states = apply_rotary_pos_emb_onedim(vision_states, cos, sin)
    vision_states = repeat_kv(vision_states, self.self_attn.num_key_value_groups)
    text_guide_score = (torch.matmul(text_states, vision_states.transpose(-1, -2)) / math.sqrt(self.self_attn.head_dim))
    
    causal_mask = generate_causal_mask(text_len, self.self_attn.num_heads).to(text_guide_score.device)

    text_guide_score[:, :, :, total_start + total_tokens:] = text_guide_score[:, :, :, total_start + total_tokens:] + causal_mask
    text_guide_score = torch.nn.functional.softmax(text_guide_score, dim=-1)[:, :, :, total_start:total_start + total_tokens]
    
    bsz, num_head, text_len, vision_len = text_guide_score.shape
    # return text_guide_score.max(dim=2)[0].transpose(2, 1)
    text_guide_score = text_guide_score.view(bsz, num_head * text_len, vision_len).max(dim=1)[0]
    # text_guide_score = text_guide_score.mean(dim=1)
    # text_guide_score = text_guide_score.mean(dim=1)
    selected_tokens = torch.zeros(total_tokens, dtype=torch.bool, device=hidden_states.device)
    _, indexes = text_guide_score.topk(k=tkn_number, dim=1)
    
    selected_tokens[indexes] = True

    return text_guide_score, selected_tokens

def token_selection_qwen_mean(self, hidden_states, vision_embedding_pos, tkn_number = 1000):
    bsz, q_len, C = hidden_states.shape
    
    hidden_states_norm_origin = self.input_layernorm(hidden_states)
    total_start = vision_embedding_pos[0][0][0]
    total_tokens = 0
    for start, length in vision_embedding_pos[0]:
        total_tokens += length
    text_len = q_len - total_start - total_tokens
    vision_tokens_norm_origin = hidden_states_norm_origin#[:, start:total_start + total_tokens, :]
    text_tokens = hidden_states_norm_origin[:, total_start + total_tokens:, :]
    text_states = self.self_attn.q_proj(text_tokens).view(bsz, -1, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
    position_ids = torch.arange(total_start + total_tokens, q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    cos, sin = self.self_attn.rotary_emb(text_states, position_ids)
    text_states = apply_rotary_pos_emb_onedim(text_states, cos, sin)
    
    vision_states = self.self_attn.k_proj(vision_tokens_norm_origin).view(bsz, -1, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)
    position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    cos, sin = self.self_attn.rotary_emb(vision_states, position_ids)
    vision_states = apply_rotary_pos_emb_onedim(vision_states, cos, sin)
    vision_states = repeat_kv(vision_states, self.self_attn.num_key_value_groups)
    text_guide_score = (torch.matmul(text_states, vision_states.transpose(-1, -2)) / math.sqrt(self.self_attn.head_dim))
    
    causal_mask = generate_causal_mask(text_len, self.self_attn.num_heads).to(text_guide_score.device)

    text_guide_score[:, :, :, total_start + total_tokens:] = text_guide_score[:, :, :, total_start + total_tokens:] + causal_mask
    text_guide_score = torch.nn.functional.softmax(text_guide_score, dim=-1)[:, :, :, total_start:total_start + total_tokens]
    
    bsz, num_head, text_len, vision_len = text_guide_score.shape
    # return text_guide_score.max(dim=2)[0].transpose(2, 1)
    text_guide_score = text_guide_score.view(bsz, num_head * text_len, vision_len).mean(dim=1)
    # text_guide_score = text_guide_score.mean(dim=1)
    # text_guide_score = text_guide_score.mean(dim=1)
    selected_tokens = torch.zeros(total_tokens, dtype=torch.bool, device=hidden_states.device)
    _, indexes = text_guide_score.topk(k=tkn_number, dim=1)
    
    selected_tokens[indexes] = True

    return text_guide_score, selected_tokens


def token_selection_mean(self, hidden_states, vision_embedding_pos, tkn_number = 1000):
    bsz, q_len, C = hidden_states.shape
    
    hidden_states_norm_origin = self.attention_norm(hidden_states)
    total_start = vision_embedding_pos[0][0][0]
    total_tokens = 0
    for start, length in vision_embedding_pos[0]:
        total_tokens += length
    text_len = q_len - total_start - total_tokens



    bsz, q_len, _ = hidden_states.size()

    qkv_states = self.attention.wqkv(hidden_states_norm_origin)

    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.attention.num_key_value_groups,
        d=self.attention.head_dim,
    )

    query_states = qkv_states[..., : self.attention.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    kv_seq_len = key_states.shape[-2]

    position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

    cos, sin = self.attention.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.attention.num_key_value_groups)


    text_states = query_states[:, :, total_start + total_tokens:, :]
    vision_states = key_states

    text_guide_score = (torch.matmul(text_states, vision_states.transpose(-1, -2)) / math.sqrt(self.attention.head_dim))
    
    causal_mask = generate_causal_mask(text_len, self.attention.num_heads).to(text_guide_score.device)

    text_guide_score[:, :, :, total_start + total_tokens:] = text_guide_score[:, :, :, total_start + total_tokens:] + causal_mask
    text_guide_score = torch.nn.functional.softmax(text_guide_score, dim=-1)[:, :, :, total_start:total_start + total_tokens]
    
    bsz, num_head, text_len, vision_len = text_guide_score.shape
    text_guide_score = text_guide_score.view(bsz, num_head * text_len, vision_len).mean(dim=1)
    selected_tokens = torch.zeros(total_tokens, dtype=torch.bool, device=hidden_states.device)
    _, indexes = text_guide_score.topk(k=tkn_number, dim=1)
    
    selected_tokens[indexes] = True

    return text_guide_score, selected_tokens



#这个还没改
def token_selection_pool(self, hidden_states, vision_embedding_pos, tkn_number = 1000):
    bsz, q_len, C = hidden_states.shape
    
    hidden_states_norm_origin = self.input_layernorm(hidden_states)
    total_start = vision_embedding_pos[0][0][0]
    total_tokens = 0
    for start, length in vision_embedding_pos[0]:
        total_tokens += length
    text_len = q_len - total_start - total_tokens
    vision_tokens_norm_origin = hidden_states_norm_origin[:, total_start:total_start + total_tokens, :]
    text_tokens = hidden_states_norm_origin[:, total_start + total_tokens:, :]
    text_states = self.self_attn.q_proj(text_tokens).view(bsz, -1, self.attention.num_heads, self.attention.head_dim).transpose(1, 2)
    position_ids = torch.arange(total_start + total_tokens, q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    cos, sin = self.rotary_emb(text_states, position_ids)
    text_states = apply_rotary_pos_emb_onedim(text_states, cos, sin)
    bsz, num_head, text_len, C = text_states.shape

    #need to be fixed for internvl2.5 
    adaptive_max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
    q_max = adaptive_max_pool(text_states.view(num_head * bsz, text_len, C).transpose(1, 2).float())
    q_max = q_max.transpose(1, 2).view(bsz, num_head, 1, C).bfloat16()
    
    vision_states = self.self_attn.k_proj(vision_tokens_norm_origin).view(bsz, -1, self.attention.num_key_value_heads, self.attention.head_dim).transpose(1, 2)
    position_ids = torch.arange(total_start, total_start + total_tokens, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    cos, sin = self.rotary_emb(vision_states, position_ids)
    vision_states = apply_rotary_pos_emb_onedim(vision_states, cos, sin)
    vision_states = repeat_kv(vision_states, self.num_key_value_groups)
    #need to be fixed
    vision_states = torch.cat([vision_states, q_max], dim=2)
    text_guide_score = (torch.matmul(text_states, vision_states.transpose(-1, -2)) / math.sqrt(self.attention.head_dim))
    text_guide_score = torch.nn.functional.softmax(text_guide_score, dim=-1)[:, :, :, :-1]
    text_guide_score = text_guide_score.max(dim=1)[0].max(dim=1)[0]
    selected_tokens = torch.zeros(total_tokens, dtype=torch.bool, device=hidden_states.device)
    _, indexes = text_guide_score.topk(k=tkn_number, dim=1)
    
    selected_tokens[indexes] = True
    
    return text_guide_score, selected_tokens
       
        

class InternLM2ForSelector_FromSelf(InternLM2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """
    def __init__(self, config: InternLM2Config, bigger_model, drop_func_name):
        super().__init__(config)
        self._attn_implementation = config._attn_implementation
        drop_layers = config.token_selector_layer + 1
        self.layers = bigger_model.language_model.model.layers[:drop_layers]
        self.mlp1 = bigger_model.mlp1
        self.padding_idx = config.pad_token_id
        # drop_func = getattr(InternLM2DecoderLayer_Selector, drop_func_name)
        drop_func = globals()[drop_func_name]
        
        self.layers[-1].token_selection_pe = types.MethodType(drop_func,  self.layers[-1])
        self.gradient_checkpointing = False
    
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_embedding_pos: Optional[torch.LongTensor] = None,
        # output_attentions: Optional[bool] = False,
        tkn_number: int = 0,
    ):
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
       
       
        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers[:-1]):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

            hidden_states = layer_outputs[0]
        
        text_guide_score = self.layers[-1].token_selection_pe(hidden_states, vision_embedding_pos, tkn_number)
        return text_guide_score
       

class Qwen2ForSelector_EXTERNAL(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """


    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def __init__(self, config, drop_func_name):
        super().__init__(config)
        self.downsample_ratio = config.downsample_ratio
        # Enable Flash Attention if supported, otherwise fall back to eager attention.

       
        self.language_model = Qwen2ForCausalLM(AutoConfig.for_model(**config.llm_config))
        del self.language_model.lm_head
        import pdb; pdb.set_trace()
        vit_hidden_size = config.vision_config["hidden_size"]
        llm_hidden_size = config.llm_config["hidden_size"]

       
        # drop_func = getattr(InternLM2DecoderLayer_Selector, drop_func_name)
        drop_func = globals()[drop_func_name]
        
        self.language_model.model.layers[-1].token_selection_pe = types.MethodType(drop_func,  self.language_model.model.layers[-1])

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        

    
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_embedding_pos: Optional[torch.LongTensor] = None,
        # output_attentions: Optional[bool] = False,
        tkn_number: int = 0,
    ):
        hidden_states = inputs_embeds
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        
        position_embeddings = self.language_model.model.rotary_emb(hidden_states, position_ids)

        for idx, decoder_layer in enumerate(self.language_model.model.layers[:-1]):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

            hidden_states = layer_outputs[0]
        
        text_guide_score = self.language_model.model.layers[-1].token_selection_pe(hidden_states, vision_embedding_pos, tkn_number)
        return text_guide_score

class InternLM2ForSelector_EXTERNAL(InternLM2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """


    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def __init__(self, config, drop_func_name):
        super().__init__(config)
        self.downsample_ratio = config.downsample_ratio
        # Enable Flash Attention if supported, otherwise fall back to eager attention.

        
        self.language_model = InternLM2ForCausalLM(InternLM2Config(**config.llm_config))
        del self.language_model.output
        
            
        vit_hidden_size = config.vision_config["hidden_size"]
        llm_hidden_size = config.llm_config["hidden_size"]

       
        # drop_func = getattr(InternLM2DecoderLayer_Selector, drop_func_name)
        drop_func = globals()[drop_func_name]
        
        self.language_model.model.layers[-1].token_selection_pe = types.MethodType(drop_func,  self.language_model.model.layers[-1])

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        

    
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_embedding_pos: Optional[torch.LongTensor] = None,
        # output_attentions: Optional[bool] = False,
        tkn_number: int = 0,
    ):
        hidden_states = inputs_embeds
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        
        # position_embeddings = self.language_model.model.rotary_emb(hidden_states, position_ids)

        for idx, decoder_layer in enumerate(self.language_model.model.layers[:-1]):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                # position_embeddings=position_embeddings,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

            hidden_states = layer_outputs[0]
        
        text_guide_score = self.language_model.model.layers[-1].token_selection_pe(hidden_states, vision_embedding_pos, tkn_number)
        return text_guide_score