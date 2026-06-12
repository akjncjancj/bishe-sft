import os 
from datasets import load_dataset,Dataset,concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from qwen_vl_utils import process_vision_info
from transformers import ( 
AutoTokenizer, 
Qwen3VLForConditionalGeneration,
AutoProcessor,
BitsAndBytesConfig
)
import torch
import json
from PIL import Image
import pandas as pd
from tqdm import tqdm
from trl import GRPOConfig
from grpo_trainer import Qwen2VLGRPOTrainer

compute_dtype = getattr(torch, "float16")

processor = AutoProcessor.from_pretrained("/root/autodl-tmp/Qwen/lora-qkvo-model",use_fast=True)
model = Qwen3VLForConditionalGeneration.from_pretrained("/root/autodl-tmp/Qwen/lora-qkvo-model", 
                                             device_map='cuda', 
                                            torch_dtype=compute_dtype
                                                          )

if not hasattr(model, 'warnings_issued'):
    model.warnings_issued = {}


ds_ctw = load_dataset("./data/CTW", cache_dir="./cache", split='train').select(range(3000))
ds_casia = load_dataset("./data/CASIA", cache_dir="./cache", split='train').select(range(3000))
ds = concatenate_datasets([ds_ctw, ds_casia])

print(f"合并后数据集总条数：{len(ds)}") 
 
def get_prompt_rft_simple(example):
    
    img_pil = example['image'].convert('RGB')

    SYSTEM_PROMPT = r'''你是一个专业的OCR识别专家。请仔细识别图像中的文本内容，并准确输出。且仅输出最终识别的答案即可，不要多余解释和字符'''
    
    user_instruction = "请识别并输出这张图片中的文本内容。"
    
    ground_truth = example['text']
    
    return {
        'prompt': [
            {'role': 'system', 'content': [{"type": "text", "text": SYSTEM_PROMPT}]},
            {'role': 'user', 'content': [
                {"type": "image"},  
                {"type": "text", "text": user_instruction},    
            ]}
        ],
        'image': img_pil,
        'solution': ground_truth,
    }
 
def dataset_gen_simple():

    for item in ds:
        try:
            yield get_prompt_rft_simple(item)
        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue

dataset_train = Dataset.from_generator(dataset_gen_simple)



import re

from Levenshtein import ratio as levenshtein_ratio
def accuracy_reward_func(completions, solution, **kwargs):
    rewards = []
    
    print("\n[准确率奖励]")
    for i, (completion, true_text) in enumerate(zip(completions, solution)):
        pred_text = completion[0]['content'].strip()
        true_text = true_text.strip()
        
        is_correct = pred_text == true_text
        reward = 1.0 if is_correct else 0.0
        rewards.append(reward)
        
        print(f"样本{i}: 预测='{pred_text}' | 真实='{true_text}' | 奖励={reward}")
    
    return rewards

def levenshtein_reward_func(completions, solution, **kwargs):
    res = []
    weight = 0.5
    
    print("\n[相似度奖励]")
    for i, (completion, sol) in enumerate(zip(completions, solution)):
        pred_text = completion[0]['content'].strip()
        
        raw_sim = levenshtein_ratio(pred_text, sol)
        reward = raw_sim * weight
        res.append(reward)
        
        print(f"样本{i}: 相似度={raw_sim:.3f} × {weight} = {reward:.3f}")
    
    return res


output_dir="./finetune/lora-GRPO-model-w0.5"

model.train()
peft_config = LoraConfig(
    r=4, #Rank
    lora_alpha=8,
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "gate_proj", 
        "up_proj", 
        "down_proj"
    ],
    bias="none",
    lora_dropout=0.1, 
)
 
training_args = GRPOConfig(
    learning_rate = 5e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 10,
    bf16 = False,
    fp16 = True,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    num_generations = 2, 
    max_completion_length = 1024,
    num_train_epochs = 1, 
    save_steps = 5000,

    max_grad_norm = 0.1,
    report_to = "none", 
    output_dir = "./finetune/lora-GRPO-model-w0.5",
    disable_tqdm=False, 
)
training_args.max_prompt_length = 1024

trainer = Qwen2VLGRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[
        accuracy_reward_func,
        levenshtein_reward_func],
    args=training_args,
    train_dataset=dataset_train,
    peft_config = peft_config,
)
 
trainer.train()
 
trainer.save_model(output_dir)
