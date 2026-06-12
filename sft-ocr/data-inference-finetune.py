from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from datasets import load_dataset
import Levenshtein
import warnings
warnings.filterwarnings("ignore")
from peft import PeftModel 


device = "cuda"
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/lora-qkvo-model", 
    device_map="auto",
    torch_dtype=torch.float16
)
lora_model_path = "finetune/lora-GRPO-model-w0.5"
model = PeftModel.from_pretrained(
    base_model, 
    lora_model_path,
    device_map="auto",
    dtype=torch.float16 
)


processor = AutoProcessor.from_pretrained("Qwen/lora-qkvo-model")


ds = load_dataset("./data/CASIA", cache_dir="./cache")
data = ds["test"].select(range(500))


def calculate_accuracy(test_data):
    total_samples = len(test_data)
    correct = 0                   
    total_edit_distance = 0
    
    config = {
        "max_new_tokens": 512,     
        "do_sample": False,           
        "early_stopping": True   
    }


    for idx, sample in enumerate(test_data):
        if idx % 50 == 0:
            print(f"正在处理第 {idx}/{total_samples} 个样本...")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": "识别图片中的文字"},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,              
            add_generation_prompt=True, 
            return_dict=True,           
            return_tensors="pt"      
        ).to(device)
        
        with torch.no_grad():
            response = model.generate(**inputs, **config)
            

        input_len = inputs.input_ids.shape[1]
        response2 = response[0][input_len:]  
        pred_text = processor.decode(response2, skip_special_tokens=True).strip()
        
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
