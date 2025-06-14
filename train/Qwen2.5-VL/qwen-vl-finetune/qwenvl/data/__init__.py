import re

# Define placeholders for dataset paths
# CAMBRIAN_737K = {
#     "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
#     "data_path": "",
# }

# MP_DOC = {
#     "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
#     "data_path": "PATH_TO_MP_DOC_DATA",
# }

# CLEVR_MC = {
#     "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
#     "data_path": "PATH_TO_CLEVR_MC_DATA",
# }

# VIDEOCHATGPT = {
#     "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
#     "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
# }

# data_dict = {
#     "cambrian_737k": CAMBRIAN_737K,
#     "mp_doc": MP_DOC,
#     "clevr_mc": CLEVR_MC,
#     "videochatgpt": VIDEOCHATGPT,
# }

LLAVA_VIDEO_178K = {
    "annotation_path" : "/mnt/sh/mmvision/home/yunzhuzhang/Qwen2.5-VL/llava_video_178k.json",
    "data_path" : "/mnt/sh/mmvision/data/video/public/lmms-lab/LLaVA-Video-178K/data"
}

VPRIT_LONG = {
    "annotation_path" : "/mnt/sh/mmvision/home/yunzhuzhang/huggingface/OpenGVLab/VideoChat-Flash-Training-Data/vprit_long.json",
    "data_path" : "/mnt/sh/mmvision/home/yunzhuzhang/huggingface/Mutonix/Vript/long_videos"
}

data_dict = {
    "llava_video_178k" : LLAVA_VIDEO_178K,
    "vprit_long": VPRIT_LONG
}
def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
