# bishe-sft
📌 Project Overview
This project optimizes the general multimodal large model for downstream OCR tasks, based on Qwen3-VL-4B. We adopt three core optimization schemes: LoRA fine-tuning, GRPO reinforcement learning and In-Context Learning (ICL).
This repository is my outstanding undergraduate graduation project. You can replace the dataset, base model and experimental methods based on this complete project framework to complete your own graduation design independently.

## Datasets
We use four public open-source OCR datasets for model evaluation. The download methods are shown below:
```bash
from datasets import load_dataset
ds = load_dataset("MiXaiLL76/CTW1500_OCR", cache_dir='./data/CTW')
ds = load_dataset("MiXaiLL76/ICDAR2013_OCR", cache_dir='./data/ICDAR2013')
ds = load_dataset("MiXaiLL76/ICDAR2015_OCR", cache_dir='./data/ICDAR2015')
ds = load_dataset("Teklia/CASIA-HWDB2-line", cache_dir='./data/CASIA')
```

## Model(baseline)
We take Qwen3-VL-4B as our primary baseline model. In addition, we also support gemma-3-4b and MiniCPM-V-2_6, Dataset download scripts can be found in download.py:```bash
```bash

```
