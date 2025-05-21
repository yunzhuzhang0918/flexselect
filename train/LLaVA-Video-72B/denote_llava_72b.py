import os
import json
import torch
from tqdm import tqdm
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria,get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
import numpy as np
# import torch_npu
# from torch_npu.contrib import transfer_to_npu
from decord import VideoReader, cpu
import random
import yaml
import math
import transformers
import decord
from llava.constants import IGNORE_INDEX
import copy
from typing import Dict
def create_device_map(process_ids):
    device_map = {
        # 前40层分配到 process_ids + 4
        **{f"model.layers.{i}": process_ids + 4 for i in range(40)},
        # 后40层分配到 process_ids
        **{f"model.layers.{i}": process_ids for i in range(40, 80)},
        # 后40层分配到 process_ids
        # 其他所有模块显式分配到 process_ids
        "lm_head.weight": process_ids,
        "model.embed_tokens.weight": process_ids,
        **{f"model.token_selector.layers.{i}": process_ids + 4 for i in range(40)},
        **{f"model.token_selector.layers.{i}": process_ids for i in range(40, 61)},
        "model.image_newline": process_ids,
        "model.mm_projector": process_ids,
        "model.norm.weight": process_ids,
        "model.vision_tower": process_ids,
        "model.token_selector.embed_tokens.weight": process_ids,
        "model.token_selector.image_newline": process_ids,
        "model.token_selector.mm_projector": process_ids,
        "model.token_selector.norm.weight": process_ids,
        "model.token_selector.vision_tower": process_ids
    }

    return device_map


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)
       
        for conv in source[:2]:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv, add_generation_prompt=False)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

class LlavaVideoInference:
    def __init__(self, model_path, jsonl_data, data_root, batch_size=1, max_frames=3):
        # 初始化加速器
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        
        # 设置每个进程可见的GPU（每个进程使用2个GPU）
        # 计算设备映射
        # self.device_map = {
        #     "model.vision_tower.vision_tower.vision_model": 0,
        #     "model.mm_projector": 0,
        #     "model.language_model": 1
        # }
        
        # 加载模型
        self.device_map = create_device_map(self.accelerator.local_process_index)
        # process_id = self.accelerator.local_process_index  # 0~3
        # visible_gpus = f"{2*process_id},{2*process_id+1}"  # 进程0:0,1; 进程1:2,3..
        # os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
        overwrite_config = {"mm_spatial_pool_mode": "bilinear", "use_token_selector": True, "token_selector_path": "self", "token_selector_layer":60, "drop_func_name":"token_selection"}
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, get_model_name_from_path(model_path),
            device_map=self.device_map,  # 使用自定义设备映射
            torch_dtype="bfloat16",
            overwrite_config=overwrite_config,
            attn_implementation="sdpa"
        )
        
        # 将模型移动到加速器设备
        # self.model = self.accelerator.prepare(self.model)
        # 准备数据
        self.jsonl_data = self.load_json_or_jsonl(jsonl_data)
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.data_root = data_root

    def load_jsonl(self, path):
        """加载JSONL文件"""
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]

    def load_json_or_jsonl(self, path):
        """加载 JSON 或 JSONL 文件"""
        with open(path, 'r') as f:
            first_line = f.readline().strip()
            f.seek(0)  # 重置文件指针到开头
            
            if first_line.startswith('[') or first_line.startswith('{'):
                # 普通 JSON 文件
                return json.load(f)
            else:
                # JSONL 文件
                return [json.loads(line) for line in f]

    # def load_data_from_yaml(self, yaml_path):
    #     """从 YAML 文件加载所有 JSON 或 JSONL 数据"""
    #     with open(yaml_path, 'r') as f:
    #         config = yaml.safe_load(f)
        
    #     all_data = []
    #     for dataset in config['datasets']:
    #         json_path = dataset['json_path']
    #         sampling_strategy = dataset["sampling_strategy"]
    #         cur_data_dict = self.load_json_or_jsonl(json_path)
    #         if ":" in sampling_strategy:
    #             sampling_strategy, sampling_number = sampling_strategy.split(":")
    #             if "%" in sampling_number:
    #                 sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
    #             else:
    #                 sampling_number = int(sampling_number)

    #         # Apply the sampling strategy
    #         if sampling_strategy == "first" and sampling_number is not None:
    #             cur_data_dict = cur_data_dict[:sampling_number]
    #         elif sampling_strategy == "end" and sampling_number is not None:
    #             cur_data_dict = cur_data_dict[-sampling_number:]
    #         elif sampling_strategy == "random" and sampling_number is not None:
    #             random.shuffle(cur_data_dict)
    #             cur_data_dict = cur_data_dict[:sampling_number]
    #         all_data.extend(cur_data_dict)
    #     for i, item in enumerate(all_data):
    #         item["conversation_ids"] = i
    #         with open("./data/random10.jsonl", "a") as f:
    #             f.write(json.dumps(item, ensure_ascii=False) + "\n")
    #             f.flush()
    #     return all_data
    def load_video(self, video_path, max_frames_num, fps, force_sample=False):
        print(video_path, self.accelerator.process_index)#/mnt/csp/mmvision/data/videollm/benchmarks/videomme/data/mAwgdX5VxGc.mp4
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
    def process_video(self, video_path):
        """视频处理逻辑"""
        
        frames, _, _ = self.load_video(video_path, max_frames_num=64, fps=1, force_sample=True)
        
        # 预处理视频帧
        return self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(device=f"cuda:{self.accelerator.local_process_index}", dtype=torch.bfloat16)
    def generate_answer(self, input_ids, video_tensor):
        """生成回答的核心逻辑"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        with torch.inference_mode():
            text_guide_score = model.get_selected_token_index(
                input_ids,
                position_ids=None,
                attention_mask=None, 
                past_key_values=None,
                labels=None,
                images = video_tensor,
                modalities = ["video"],
                image_sizes=None,
            )
        return text_guide_score.cpu()

    def run_inference(self):
        """分布式推理主循环"""

        os.makedirs("./temp", exist_ok=True)
    
        # 每个进程的独立文件
        rank = self.accelerator.process_index
        total = len(self.jsonl_data)
        num_gpus = self.accelerator.num_processes  # 替换原来的硬编码
        per_gpu = total // num_gpus
        start = self.accelerator.process_index * per_gpu
        end = start + per_gpu
        if self.accelerator.process_index == num_gpus - 1:
            end = total
        data_shard = self.jsonl_data[start:end]
        
        
        processed_ids = set()
        if os.path.exists(f"./data/text_guide_scores_rst_64/processes_{self.accelerator.local_process_index}.txt"):
            with open(f"./data/text_guide_scores_rst_64/processes_{self.accelerator.local_process_index}.txt", "r") as f:
                for line in f:
                    pid = line.strip()
                    if pid:
                        processed_ids.add(int(pid))
        # print(processed_ids)
        for item in tqdm(data_shard, disable=not self.accelerator.is_main_process):
            if item["conversation_ids"] in processed_ids:
                # print("pass:", item["conversation_ids"])
                continue
            try:
            # if item["video"] in ["liwei_youtube_videos/videos/youtube_video_2024/ytb_lwM2tdhKmQk.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_0DjruPc47fA.mp4", "academic_source/youcook2/202/pCTdsgv1wZ4/split_5.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_qJzoKBWcq28.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_wPwrD9bLVkY.mp4", "liwei_youtube_videos/videos/ytb_rBXyFGfMsp8", "liwei_youtube_videos/videos/youtube_video_2024/ytb_9QKWwOQlt9Q.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_khCwOGsO0dI.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_n0Y30sXTsIQ.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_e9GcYw6EkFE.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_3teeAnNXS1k.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_e_3PfEonyDA.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_l16aF-dLnZU.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_1ZhKRkCIbaQ.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_gWvyB-xs0Fk.mp4", "liwei_youtube_videos/videos/youtube_video_2024/ytb_x6coGFNfShI.mp4"]:
            #     continue
                video_path = os.path.join(self.data_root, item["video"])
                if not os.path.exists(video_path):
                    print(f"File not found: {video_path}")
                    continue
                video_tensor = self.process_video(video_path)  # (32,3,384,384)
                
                if video_tensor is None:
                    continue
                if video_tensor.shape[0] <= 32:
                    continue
                sources = [item["conversations"]]
                
                input_ids = preprocess_qwen(sources, self.tokenizer)['input_ids']
                # import pdb; pdb.set_trace()
                print("len:", len(input_ids[0]))
                if len(input_ids[0]) > 2000:
                    continue
                video_tensor = video_tensor.to(f"cuda:{self.accelerator.local_process_index}")
                # 生成完整视频的答案
                input_ids = input_ids.to(f"cuda:{self.accelerator.local_process_index}")
                text_guide_score = self.generate_answer(input_ids, [video_tensor])
                assert text_guide_score.shape[-1] == video_tensor.shape[0] * 210, f"expect ext_guide_score.shape[-1] == video_tensor.shape[0] * 210, but got text_guide_score shape : {text_guide_score.shape}, video_tensor: {video_tensor.shape}"
                torch.save(text_guide_score, f"./data/text_guide_scores_64/{item['conversation_ids']}.pt")
                with open(f"./data/text_guide_scores_rst_64/processes_{self.accelerator.local_process_index}.txt", "a") as f:
                    f.write(f"{item['conversation_ids']}\n")
            except BaseException as e:
                print(f"Process {self.accelerator.process_index} error: {e}")
                continue
        # 等待所有进程完成

import argparse

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Run LLaVA Video Inference")
    
    # 添加命令行参数，并设置默认值
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/mnt/csp/mmvision/home/yunzhuzhang/huggingface/lmms-lab/LLaVA-Video-72B-Qwen2", 
        help="Path to the model"
    )
    parser.add_argument(
        "--jsonl_path", 
        type=str, 
        default="/mnt/sh/mmvision/home/yunzhuzhang/eval_dev/data/llava_video_178k_5_percent.json", 
        help="Path to the JSONL data file"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="/mnt/csp/mmvision/data/video/public/lmms-lab/LLaVA-Video-178K/data", 
        help="Root directory of the data"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 使用解析后的参数
    inferencer = LlavaVideoInference(
        model_path=args.model_path,
        jsonl_data=args.jsonl_path,
        batch_size=1,
        max_frames=64,
        data_root=args.data_root
    )
    
    # 运行推理
    inferencer.run_inference()

if __name__ == "__main__":
    main()