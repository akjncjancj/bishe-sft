from modelscope import AutoModel, AutoTokenizer
import torch
from datasets import load_dataset
import Levenshtein
import warnings
warnings.filterwarnings("ignore")

device = "cuda"
model = AutoModel.from_pretrained(
    "OpenBMB/MiniCPM-V-2_6",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("OpenBMB/MiniCPM-V-2_6",trust_remote_code=True)


ds = load_dataset("./data/ICDAR2013", cache_dir="./cache")
data = ds["test"]


def calculate_accuracy(test_data):
    total_samples = len(test_data)
    correct = 0                   
    total_edit_distance = 0
    
    for idx, sample in enumerate(test_data):
        if idx % 50 == 0:
            print(f"正在处理第 {idx}/{total_samples} 个样本...")
            
        image = sample["image"].convert("RGB")
        question = "识别图片中的文字"

        msgs = [{'role': 'user', 'content': [image, question]}]

        with torch.no_grad():
            pred_text = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                do_sample=False
            )
        
        groundtruth = sample["text"].strip().replace(" ", "").replace("\n", "")
        pred_text = pred_text.replace(" ", "").replace("\n", "")

        if pred_text == groundtruth:
            correct += 1
        total_edit_distance += Levenshtein.distance(pred_text, groundtruth)
   
    accuracy = (correct / total_samples) * 100
    avg_edit_distance = total_edit_distance / total_samples
    
    return {
        "总样本数": total_samples,
        "精确匹配正确数": correct,
        "精确匹配准确率(%)": round(accuracy, 2), 
        "平均编辑距离": round(avg_edit_distance, 2)
    }

if __name__ == "__main__":
    
    print("开始OCR精度评估...")
    results = calculate_accuracy(data)
    print("\n===== OCR推理精度评估结果 =====")
    for k, v in results.items():
        print(f"{k}: {v}")
