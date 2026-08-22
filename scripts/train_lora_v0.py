#!/usr/bin/env python3
"""
🚀 SFT LoRA V0 훈련 및 가중치 병합 러너 (Qwen2.5-1.5B)
- Data: train_v0_270.jsonl / val_v0_30.jsonl (Evidence 스키마 적용)
- Architecture: LoRA (r=16, alpha=32), bfloat16 MPS 가속
- Loss: Response-Only Loss Masking (Assistant JSON만 학습)
- Post-Training: Standalone Full Weight Merge
"""

import os
import gc
import json
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 워터마크 충돌 방지
for k in ["PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"]:
    os.environ.pop(k, None)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "scripts" / "data"
TRAIN_FILE = str(DATA_DIR / "train_v0_270.jsonl")
VAL_FILE = str(DATA_DIR / "val_v0_30.jsonl")
OUTPUT_DIR = BASE_DIR / "scripts" / "output" / "qwen2.5-1.5b-v0-lora"
MERGED_DIR = BASE_DIR / "scripts" / "output" / "qwen2.5-1.5b-v0-merged"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ [1/4] Device: {device} (Apple Silicon Metal bfloat16 가속)")

    print(f"📦 [2/4] 모델 로드 중: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == "mps" else torch.float32,
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

    raw_datasets = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE}
    )

    def preprocess_conversations(examples):
        input_ids_list = []
        labels_list = []

        for messages in examples["messages"]:
            # ChatML 포맷팅
            prompt_msgs = messages[:-1] # system + user
            full_msgs = messages        # system + user + assistant

            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

            prompt_enc = tokenizer(prompt_text, max_length=768, truncation=True)
            full_enc = tokenizer(full_text, max_length=768, truncation=True)

            input_ids = full_enc["input_ids"]
            labels = list(input_ids)

            prompt_len = len(prompt_enc["input_ids"])
            # Assistant 이전 프롬프트 토큰은 Loss 계산 제외 (-100)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    print("✂️ [3/4] Conversational Loss 마스킹 데이터셋 토크나이징 중...")
    tokenized_datasets = raw_datasets.map(
        preprocess_conversations,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    # 2 Epoch 초고속 최적화 (270개 샘플 기준 약 2~3분)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    print("🚀 [4/4] LoRA V0 SFT 파인튜닝 시작...")
    trainer.train()

    print(f"💾 LoRA 어댑터 저장: {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    print("🧬 [Merge] LoRA 가중치 완전 병합 -> 완전체 모델 생성 중...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR))
    merged_model = merged.merge_and_unload()

    print(f"🎉 완전체 모델 저장: {MERGED_DIR}")
    merged_model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print("✅ LoRA V0 훈련 및 가중치 병합이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    train()
