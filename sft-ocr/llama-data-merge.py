from tqdm import tqdm
from datasets import load_dataset
import os
import json
from PIL import Image

def save_merged_datasets(output_dir="OCR_data_7000"):

    if not os.path.exists(output_dir):  #判断指定路径（文件）是否真实存在，返回True/Fasle
        os.makedirs(output_dir)         #创建指定的目录

    all_data = []

    datasets = [
        {
            "name": "CTW",
            "loader": load_dataset("./data/CTW", cache_dir="./cache", split='train'), #只取训练集，因为进行lora微调训练
            "prefix": "CTW"
        },
        {
            "name": "CASIA",
            "loader": load_dataset("./data/CASIA", cache_dir="./cache", split='train'),
            "prefix": "CASIA"
        }
    ]


    for ds_info in datasets:    #依次遍历两个字典
        ds_name = ds_info["name"]
        ds = ds_info["loader"]
        prefix = ds_info["prefix"]
        
        print(f"\n开始处理{ds_name}数据集...")
        #enumerate用于同时拿到索引 + 原数据
        for idx, item in tqdm(enumerate(ds), total=len(ds), desc=f"处理{ds_name}数据,将hugging face数据转化为更适配的图片+json格式"):
            if ds_name == "CTW" and idx >= 3000:
                break;
            if ds_name == "CASIA" and idx >= 4000:
                break;
            img_filename = f"{prefix}_{idx+1}.jpg"
            img_path = f"{output_dir}/{img_filename}"
            image = item["image"]  # image是PIL的 PIL.Image.Image 类对象

            #调用 PIL Image对象的 .save() 方法，把内存里的图片保存到硬盘上
            image.save(img_path)

            # 构造微调数据格式（指令适配英文/数字OCR场景）
            all_data.append(
                {
                    "messages": [
                        {
                            "content": "<image>识别图片中的文本内容",
                            "role": "user",
                        },
                        {
                            "content": item["text"],  # 标注文本
                            "role": "assistant",
                        },
                    ],
                    "images": [img_path],  # 图片路径
                }
            )

    # 保存合并后的JSON文件
    json_file_path = f"{output_dir}/OCR_data_7000.json"
    #以写入模式打开指定路径的文件，将打开的文件对象赋值给变量 f
    with open(json_file_path, "w", encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False)
        #json.dump是 json模块的函数，核心作用：把 Python 结构化数据（列表）直接序列化成 JSON 格式字符串，并写入到已打开的文件对象中

    print(f"\n合并完成！")
    print(f"总样本数：{len(all_data)}")
    print(f"图片和JSON文件已保存至：{output_dir}")

if __name__ == "__main__":
    save_merged_datasets(output_dir="OCR_data_7000")
