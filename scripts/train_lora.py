#!/usr/bin/env python3
"""
Qwen2.5-0.5B-Instruct LoRA 파인튜닝 스크립트 (장문 1만자 & 다중 퀴즈 지원)
- Mac M4 Pro Apple Silicon (MPS) 가속
- 4096 max_length 컨텍스트 지원
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

def train():
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

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_file = str(DATA_DIR / "train_quiz_dataset.jsonl")
    val_file = str(DATA_DIR / "val_quiz_dataset.jsonl")
    
    raw_datasets = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file}
    )

    def preprocess_function(examples):
        inputs = examples["full_text"]
        model_inputs = tokenizer(inputs, max_length=4096, truncation=True, padding=False)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=12,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        warmup_steps=2,
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
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

    print("🚀 장문 & 다중 퀴즈 LoRA 파인튜닝 시작...")
    trainer.train()

    print(f"💾 학습 완료! LoRA 어댑터 저장 중: {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("✅ 모든 가중치와 토크나이저가 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    train()
