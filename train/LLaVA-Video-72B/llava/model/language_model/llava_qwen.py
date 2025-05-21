#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig

def soft_sort(x, temperature=0.0001):
    """
    对输入序列 x 进行可导的软排序。
    
    参数:
    x (torch.Tensor): 输入序列，形状为 (n,)
    temperature (float): 温度参数，控制排序的平滑程度
    
    返回:
    ranks (torch.Tensor): 软排序后的排名，形状为 (n,)
    """
    n = len(x)
    
    # 计算两两元素之间的差值
    pairwise_diff = x.unsqueeze(1) - x.unsqueeze(0)
    
    # 使用 sigmoid 函数计算每对元素之间的比较结果
    # sigmoid((xj - xi)/temperature) 表示 xj > xi 的概率
    pairwise_comp = torch.sigmoid(pairwise_diff / temperature)
    
    # 计算每个元素的排名
    # 对每一行求和（减去0.5是为了处理自身与自身的比较）
    ranks = torch.sum(pairwise_comp, dim=1) - 0.5
    
    return ranks

def spearmanr(pred, target):
    #pred使用softrank，保持可导
    pred = soft_sort(pred.squeeze())
    #target直接正常排序
    target = target.squeeze().argsort().argsort().to(pred.dtype)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return 1 - (pred * target).sum()

def keep_topk_multiply(tensor, k=1000):
    flattened = tensor.flatten()
    k = min(k, flattened.size(0))
    _, indices = torch.topk(flattened, k)
    mask = torch.zeros_like(flattened, dtype=torch.bool)
    mask[indices] = True
    mask = mask.view_as(tensor)
    return tensor * mask

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        tome_layers: Optional[List[int]] = None,
        hidden_states_shape: Optional[List[int]] = None,
        vision_embedding_pos: Optional[torch.LongTensor] = None,
        cache_position=None,
        teacher_rank: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, vision_embedding_pos, text_guide_score) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
        # with torch.no_grad():
        #     text_guide_score_teacher = self.model.frame_selectors(inputs_embeds, vision_embedding_pos)
        #     torch.npu.empty_cache()
        # loss = ((text_guide_score_teacher - text_guide_score) ** 2).mean()
        # text_guide_score = keep_topk_multiply(text_guide_score)
        # text_guide_score_teacher = keep_topk_multiply(text_guide_score_teacher)
       
        probs = text_guide_score.softmax(dim=-1)
        # h = -torch.sum(probs * torch.log2(probs), dim=-1).squeeze().float()
        # h = 1 - probs.topk(k=500, dim=-1)[0].sum()
        # if teacher_rank[0].shape[1] != text_guide_score.shape[1]:
        #     import pdb; pdb.set_trace()
        l2_loss = ((teacher_rank[0] - text_guide_score) ** 2).sum()
        spearman_loss = spearmanr(target=teacher_rank[0], pred=text_guide_score).float() #+ h.float()
        loss =  l2_loss + spearman_loss
        return CausalLMOutputWithPast(
            loss=loss,
            logits=spearman_loss,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        # # text_guide_score_student = self.
        # if dpo_forward:
        #     outputs = self.model(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         inputs_embeds=inputs_embeds,
        #         use_cache=use_cache,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )

        #     hidden_states = outputs[0]
        #     logits = self.lm_head(hidden_states)
        #     return logits, labels

        # else:
        #     return super().forward(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         inputs_embeds=inputs_embeds,
        #         labels=labels,
        #         use_cache=use_cache,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         tome_layers=tome_layers,
        #         vision_embedding_pos=vision_embedding_pos,
        #         hidden_states_shape=hidden_states_shape,
        #         return_dict=return_dict,
        #     )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, vision_embedding_pos) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
            kwargs["vision_embedding_pos"] = vision_embedding_pos
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        vision_embedding_pos = kwargs.pop("vision_embedding_pos", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if vision_embedding_pos is not None:
            inputs["vision_embedding_pos"] = vision_embedding_pos
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
