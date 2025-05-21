import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Any
import json

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_token_selector: bool = field(default=False)
    token_selector_path :  Optional[str] = field(default=None)
    # token_selector_config: Optional[str] = field(default=None)
    drop_func_name : Optional[str] = field(default=None)
    token_selector_layer: Optional[str] = field(default=None)
    
@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    # lr_scheduler_kwargs: Dict[str, Any] = field(
    #     default_factory=lambda: {"num_cycles": 3},  # 默认值
    #     metadata={
    #         "help": "Extra scheduler args. Default: {'num_cycles':3} for cosine_with_restarts",
    #         "type": json.loads  # 自动将JSON字符串转为字典
    #     }
    # )

