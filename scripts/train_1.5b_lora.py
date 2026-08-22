#!/usr/bin/env python3
"""
🚀 Qwen2.5-1.5B-Instruct LoRA 파인튜닝 & 완전체 가중치 병합(Merge) 파이프라인
- Target Repo: kez-lab/qwen2.5-1.5b-quiz-korean
- Mac M4 Pro Apple Silicon (MPS) 안전 가속
- Response-Only Loss 마스킹 적용
"""

import os
import gc
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

# 메모리 안전 설정
for k in ["PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"]:
    os.environ.pop(k, None)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output" / "qwen2.5-1.5b-quiz-korean-lora"
MERGED_DIR = Path(__file__).parent / "output" / "qwen2.5-1.5b-quiz-korean-merged"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)

def select_device():
    if torch.backends.mps.is_available():
        print("⚡ Apple Silicon Metal Performance Shaders (MPS) 가속을 사용합니다.")
        return torch.device("mps")
    return torch.device("cpu")

def train_1_5b():
    device = select_device()
    
    print(f"📦 [1/3] 베이스 모델 및 토크나이저 로드 중: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        trust_remote_code=True
    )

    # 1.5B 맞춤형 고용량 LoRA 설정 (r=32, alpha=64)
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

    def preprocess_with_response_masking(examples):
        input_ids_list = []
        labels_list = []
        
        for prompt_text, full_text in zip(examples["prompt"], examples["full_text"]):
            full_encoded = tokenizer(full_text, max_length=1536, truncation=True)
            input_ids = full_encoded["input_ids"]
            labels = list(input_ids)
            
            prompt_encoded = tokenizer(prompt_text, max_length=1536, truncation=True)
            prompt_len = len(prompt_encoded["input_ids"])
            
            # 사용자 질문 부분은 Loss 마스킹 (-100)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
                
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            
        return {"input_ids": input_ids_list, "labels": labels_list}

    print("✂️ [2/3] Response-Only Loss 마스킹 토크나이징 중...")
    tokenized_datasets = raw_datasets.map(
        preprocess_with_response_masking,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    # 2 Epoch 학습 설정 (1,000개 데이터셋 기준 약 10~15분)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_pin_memory=False,
        use_cpu=(device.type == "cpu")
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    print("🚀 [3/3] Qwen2.5-1.5B 파인튜닝 시작...")
    trainer.train()

    print(f"💾 LoRA 어댑터 저장 중: {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    # 가중치 완전 병합 (Weight Merge) 실행 -> 독립형 1.5B 완전체 모델 생성
    print("🧬 [Merge] LoRA 어댑터를 원본 1.5B 베이스 모델과 영구 결합 중...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        trust_remote_code=True
    )
    merged_model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR))
    merged_model = merged_model.merge_and_unload()

    print(f"🎉 완전체 모델 저장 중: {MERGED_DIR}")
    merged_model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print("✅ 모든 파인튜닝 및 가중치 병합이 완벽하게 완료되었습니다!")

if __name__ == "__main__":
    train_1_5b()
