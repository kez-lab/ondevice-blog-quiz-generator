#!/usr/bin/env python3
"""
Response-Only Loss Masking & 고용량 LoRA 파인튜닝 스크립트
- 1,000개 대규모 데이터셋 기반
- 입력 프롬프트 Loss 마스킹 (Label = -100) -> 오직 JSON 퀴즈 생성에만 100% 가중치 집중
- LoRA (r=32, alpha=64), Cosine LR Scheduler 적용
"""

import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output" / "qwen2.5-0.5b-blog-quiz-lora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def select_device():
    if torch.backends.mps.is_available():
        print("⚡ Apple Silicon Metal Performance Shaders (MPS) 가속을 사용합니다.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("⚡ NVIDIA CUDA GPU 가속을 사용합니다.")
        return torch.device("cuda")
    else:
        print("⚠️ GPU를 찾지 못하여 CPU 모드로 실행합니다.")
        return torch.device("cpu")

def train_advanced():
    device = select_device()
    
    print(f"📦 베이스 모델 및 토크나이저 로드 중: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        trust_remote_code=True
    )

    # 고용량 LoRA 구성 (r=32, alpha=64)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_file = str(DATA_DIR / "train_large_1k.jsonl")
    val_file = str(DATA_DIR / "val_large_1k.jsonl")
    
    raw_datasets = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file}
    )

    # Response-Only Loss 마스킹 전처리 함수
    assistant_prefix = "<|im_start|>assistant\n"
    assistant_prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)

    def preprocess_with_response_masking(examples):
        input_ids_list = []
        labels_list = []
        
        for prompt_text, full_text in zip(examples["prompt"], examples["full_text"]):
            full_encoded = tokenizer(full_text, max_length=2048, truncation=True)
            input_ids = full_encoded["input_ids"]
            labels = list(input_ids)
            
            prompt_encoded = tokenizer(prompt_text, max_length=2048, truncation=True)
            prompt_len = len(prompt_encoded["input_ids"])
            
            # 프롬프트 부분(User 텍스트)은 Loss 계산에서 제외 (-100 마스킹)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
                
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            
        return {"input_ids": input_ids_list, "labels": labels_list}

    print("✂️ Response-Only Loss 마스킹 토크나이징 중...")
    tokenized_datasets = raw_datasets.map(
        preprocess_with_response_masking,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        use_cpu=(device.type == "cpu")
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    print("🚀 1,000개 대규모 데이터셋 & LoRA (r=32) 풀 트레이닝 시작...")
    trainer.train()

    print(f"💾 학습 완료! 고성능 LoRA 어댑터 저장 중: {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("✅ 모델 가중치와 토크나이저 저장이 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    train_advanced()
