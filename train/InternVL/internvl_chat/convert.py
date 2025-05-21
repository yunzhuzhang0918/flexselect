import json

# 输入文件（标准JSON）
input_file = '/mnt/sh/mmvision/home/yunzhuzhang/Qwen2.5-VL/llava_video_178k.json'
# 输出文件（JSONL）
output_file = 'llava_video_178k.jsonl'

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    data = json.load(f_in)  # 读取原始JSON
    
    if isinstance(data, list):
        # 如果JSON是数组，逐行写入每个对象
        for item in data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    elif isinstance(data, dict):
        # 如果JSON是单个对象，直接写入
        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    else:
        raise ValueError("Invalid JSON format: Expected array or object")

print(f"转换完成 -> {output_file}")