import sys
import os
import json

# 로컬 datasets 폴더와 HuggingFace datasets 충돌 해결
# sys.path에서 현재 디렉토리를 제거하고 HuggingFace datasets를 import
original_path = sys.path.copy()
sys.path = [p for p in sys.path if not p.startswith(os.getcwd())]

import datasets as hf_datasets

# 원래 경로 복원
sys.path = original_path

# OpenAI GSM8K 데이터셋 로드
dataset = hf_datasets.load_dataset("openai/gsm8k", "main", cache_dir="./datasets")

# 저장 디렉토리 생성
os.makedirs("datasets/gsm8k", exist_ok=True)

# test 데이터를 gsm8k.jsonl로 저장
with open("datasets/gsm8k/gsm8k.jsonl", "w", encoding="utf-8") as f:
    for item in dataset["test"]:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

# train 데이터를 train.jsonl로 저장
with open("datasets/gsm8k/train.jsonl", "w", encoding="utf-8") as f:
    for item in dataset["train"]:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print("datasets/gsm8k/gsm8k.jsonl 생성 완료")
print("datasets/gsm8k/train.jsonl 생성 완료")
