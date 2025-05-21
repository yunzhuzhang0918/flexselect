# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,Qwen2ForCausalLM,
                          LlamaTokenizer, AutoConfig, AutoTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .modeling_internlm2 import InternLM2ForCausalLM
from .token_selector import InternLM2ForSelector_FromSelf, InternLM2ForSelector_EXTERNAL, Qwen2ForSelector_EXTERNAL
from .configuration_internlm2 import InternLM2Config
import math
import os

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')
       
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        

        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
    
    def convert_ids_from_internvl_to_qwen(self, ids):
        return self.tokenizer_selector.encode(self.tokenizer_teacher.decode(ids, spaces_between_special_tokens=False))
    def load_token_selector(self, config, tokenizer_teacher):
        if config.token_selector_path == "self":
            self.token_selector_self = InternLM2ForSelector_FromSelf(config=config, bigger_model=self, drop_func_name=config.drop_func_name).to(self.dtype)
        else:
            if config.token_selector_type == 'internlm':
                # token_selector_config._attn_implementation = "flash_attention_2"
                self.token_selector = InternLM2ForSelector_EXTERNAL.from_pretrained(config.token_selector_path, drop_func_name=config.drop_func_name).to(self.dtype)
                
            elif config.token_selector_type == 'qwen':
                self.token_selector = Qwen2ForSelector_EXTERNAL.from_pretrained(config.token_selector_path, drop_func_name=config.drop_func_name).to(self.dtype)
                self.tokenizer_teacher = tokenizer_teacher
                self.tokenizer_selector = AutoTokenizer.from_pretrained(config.token_selector_path, trust_remote_code=True)
                self.img_start_id = 92544
                self.img_end_id = 92545
                self.img_context_token_id = 92546
                self.img_context_token_id_for_selector = self.convert_ids_from_internvl_to_qwen(self.img_context_token_id)[0]
                self.img_start_id_for_selector = self.convert_ids_from_internvl_to_qwen(self.img_start_id)[0]
                self.img_end_id_for_selector = self.convert_ids_from_internvl_to_qwen(self.img_end_id)[0]

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # image_flags = image_flags.squeeze(-1)
       
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()


        input_ids_for_token_selector = torch.tensor(self.convert_ids_from_internvl_to_qwen(input_ids[0][1:])).unsqueeze(dim=0).to(input_ids.device)
        input_embeds_for_token_selector = self.token_selector.language_model.get_input_embeddings()(input_ids_for_token_selector).clone()

        vit_embeds = self.extract_feature(pixel_values)
        # vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        # vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, tkn_budget=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        self.img_start_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')
        
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tkn_budget=tkn_budget,
            **generation_config
        )
     
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response


    def compress_vision_embeddings(self, vision_embedding_pos, inputs_embeds, tokens_per_frame=258, max_frames_per_group=64, tkn_budget=32*258, **selector_args):
        """
        压缩长视频序列的视觉token
        参数:
            vision_embedding_pos: 视觉嵌入位置信息 [[(start_pos, length)]]
            inputs_embeds: 完整的输入嵌入 [batch, seq_len, dim]
            frame_selector: 帧选择模型
            tokens_per_frame: 每帧对应的token数量
            max_frames_per_group: 每组最大帧数
            selector_args: 传递给frame_selector的额外参数
        返回:
            压缩后的输入嵌入 [batch, new_seq_len, dim]
        """
        # 解析视觉嵌入位置信息
        total_start = vision_embedding_pos[0][0][0]
        total_tokens = sum(length for _, length in vision_embedding_pos[0])
        

        batch_size, seq_len, embed_dim = inputs_embeds.shape
        # 分割前缀和后缀嵌入
        prefix = inputs_embeds[:, :total_start, :]
        post = inputs_embeds[:, total_start+total_tokens:, :]
        vision_tokens = inputs_embeds[:, total_start:total_start+total_tokens, :]

        # 计算帧分组信息
        num_frames = total_tokens // tokens_per_frame
        num_groups = math.ceil(num_frames / max_frames_per_group)
        print("max selected tokens:", tkn_budget)
        tkn_number = int(tkn_budget / num_groups)
        if tkn_number >= total_tokens:
            return torch.ones(total_tokens).to(inputs_embeds.device, dtype=bool)
        selected_indices = set()

        # 并行处理每个帧组（实际可根据硬件情况调整并行度）
        token_selector = self.token_selector
        # next_start=0
        for group_idx in range(num_groups):
            # 生成跨步采样的帧索引 (0,7,15...)
            frame_indices = [group_idx + i*num_groups for i in range(max_frames_per_group) 
                            if (group_idx + i*num_groups) < num_frames]
            # frame_indices = [next_start + i for i in range(max_frames_per_group) 
            #                 if (next_start + i) < min((math.ceil(num_frames/num_groups)) * (group_idx + 1), num_frames)]
            # next_start = min((math.ceil(num_frames/num_groups)) * (group_idx + 1), num_frames)
            
            # 提取当前组的视觉token [batch, group_tokens, dim]
            group_tokens = torch.cat([
                vision_tokens[:, idx*tokens_per_frame:(idx+1)*tokens_per_frame, :] 
                for idx in frame_indices
            ], dim=1)

            # 构建当前组的完整输入序列
            current_embeds = torch.cat([prefix, group_tokens, post], dim=1)
            
            # 准备位置信息
            current_vision_pos = [[(total_start, group_tokens.shape[1])]]
           

            # 调用选择器
            _, token_indices = token_selector(
                inputs_embeds=current_embeds,
                vision_embedding_pos=current_vision_pos,
                tkn_number=tkn_number
            )
            # 转换局部索引到全局索引
            
            for local_idx in token_indices.nonzero(as_tuple=True)[0]:
                frame_in_group = local_idx.item() // tokens_per_frame
                global_frame = frame_indices[frame_in_group]
                global_idx = global_frame * tokens_per_frame + (local_idx % tokens_per_frame)
                selected_indices.add(global_idx.item())

        # 按原始顺序排序
        sorted_indices = sorted(selected_indices)
       

        # 构建最终序列
        return sorted_indices

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            tkn_budget: Optional[int] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            
            vit_embeds_for_teacher = self.mlp1(vit_embeds)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

            B, N, C1 = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C1)
            input_ids = input_ids.reshape(B * N)
            
            
            
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            
            input_embeds[selected] = vit_embeds_for_teacher.reshape(-1, C1).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C1)

            ### drop vision tokens
            
            if hasattr(self, "token_selector"):
                token_selector = self.token_selector
                if isinstance(token_selector, Qwen2ForSelector_EXTERNAL):
                    input_ids_for_token_selector = torch.tensor(self.convert_ids_from_internvl_to_qwen(input_ids[1:])).to(input_ids.device)
                    selected_for_token_selector = (input_ids_for_token_selector == self.img_context_token_id_for_selector)
                elif isinstance(token_selector, InternLM2ForSelector_EXTERNAL):
                    input_ids_for_token_selector = input_ids
                    selected_for_token_selector = selected
                elif isinstance(token_selector, InternLM2ForSelector_FromSelf):
                    input_embeds_for_token_selector = input_embeds
                    
                start = torch.where(input_ids_for_token_selector == self.img_start_id_for_selector)[0].min().item()
                end = torch.where(input_ids_for_token_selector == self.img_end_id_for_selector)[0].max().item() + 1
                vision_embed_pos = [[[start, end - start]]]
                input_embeds_for_token_selector = token_selector.language_model.get_input_embeddings()(input_ids_for_token_selector)
                vit_embeds_for_token_selector = token_selector.mlp1(vit_embeds)
                N2, C2 = input_embeds_for_token_selector.shape[-2:]
                input_embeds_for_token_selector = input_embeds_for_token_selector.reshape(B * N2, C2)
                input_embeds_for_token_selector[selected_for_token_selector] = vit_embeds_for_token_selector.reshape(-1, C2).to(input_embeds.device)
                input_embeds_for_token_selector = input_embeds_for_token_selector.reshape(B, N2, C2)

                prefix_embedding = input_embeds[:, :start, :]
                vision_embedding = input_embeds[:, start : end, :]
                suffix_embedding = input_embeds[:, end:, :]
                
                selected_index = self.compress_vision_embeddings(vision_embed_pos, input_embeds_for_token_selector, tkn_budget=tkn_budget)
                selected_embedding = vision_embedding[:, selected_index, :]
                input_embeds = torch.cat([prefix_embedding, selected_embedding, suffix_embedding], dim=1)
                attention_mask = torch.ones(1, input_embeds.shape[1]).to(dtype=torch.int64, device=input_embeds.device)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        
        
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )
        return outputs