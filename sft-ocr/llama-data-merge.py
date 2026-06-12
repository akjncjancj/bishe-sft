from tqdm import tqdm
from datasets import load_dataset
import os
import json
from PIL import Image

def save_merged_datasets(output_dir="OCR_data_7000"):

    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)        

    all_data = []

    datasets = [
        {
            "name": "CTW",
            "loader": load_dataset("./data/CTW", cache_dir="./cache", split='train'), 
            "prefix": "CTW"
        },
        {
            "name": "CASIA",
            "loader": load_dataset("./data/CASIA", cache_dir="./cache", split='train'),
            "prefix": "CASIA"
        }
    ]


    for ds_info in datasets: 
        ds_name = ds_info["name"]
        ds = ds_info["loader"]
        prefix = ds_info["prefix"]
        
        print(f"\n开始处理{ds_name}数据集...")
        for idx, item in tqdm(enumerate(ds), total=len(ds), desc=f"处理{ds_name}数据,将hugging face数据转化为更适配的图片+json格式"):
            if ds_name == "CTW" and idx >= 3000:
                break;
            if ds_name == "CASIA" and idx >= 4000:
                break;
            img_filename = f"{prefix}_{idx+1}.jpg"
            img_path = f"{output_dir}/{img_filename}"
            image = item["image"] 

            image.save(img_path)

            all_data.append(
                {
                    "messages": [
                        {
                            "content": "<image>识别图片中的文本内容",
                            "role": "user",
                        },
                        {
                            "content": item["text"], 
                            "role": "assistant",
                        },
                    ],
                    "images": [img_path], 
                }
            )


    json_file_path = f"{output_dir}/OCR_data_7000.json"
    with open(json_file_path, "w", encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False)

    print(f"\n合并完成！")
    print(f"总样本数：{len(all_data)}")
    print(f"图片和JSON文件已保存至：{output_dir}")

if __name__ == "__main__":
    save_merged_datasets(output_dir="OCR_data_7000")
