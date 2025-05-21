import yaml
import json
from pathlib import Path
import random

def process_data(yaml_path, output_path):
    # 设置随机种子
    random.seed(1234)
    
    # 1. 读取 data.yaml 文件
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    all_data = []
    
    # 2. 处理每个 json 文件
    for dataset in config['datasets']:
        json_path = dataset['json_path']
        print(f"Processing {json_path}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 2.1 随机选择5%的数据
        if "academic" in json_path:
            random_ratio = 1
        else:
            random_ratio = 0.2
        sample_size = max(1, int(len(data) * random_ratio))  # 确保至少选1个
        sampled_data = random.sample(data, sample_size)
        
        for item in sampled_data:
            # 2.2 只保留前两个对话
            if len(item['conversations']) > 2:
                item['conversations'] = item['conversations'][:2]
            
            # 2.3 替换 <image> 为 <video>
            for conv in item['conversations']:
                if conv['from'] == 'human':
                    conv['value'] = conv['value'].replace('<image>', '<video>')
            
            all_data.append(item)
    
    # 3. 保存合并后的文件
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Done! Merged data saved to {output_path}")
    print(f"Total samples: {len(all_data)}")

if __name__ == '__main__':
    # 输入文件路径
    data_yaml_path = "/mnt/sh/mmvision/data/video/public/lmms-lab/output_rnd20.yaml"
    output_json_path = "llava_video_178k_long.json"
    
    process_data(data_yaml_path, output_json_path)