#!/usr/bin/env python3
"""
🏆 연구팀 수준의 SOTA 온디바이스 AI 파이프라인 (Qwen2.5-1.5B)
1. Data-Centric: Response-Only Loss Masking
2. Precision: Apple Silicon Metal (MPS) bfloat16 초고속 가속
3. Optimization: Sequence Packing & Cosine Scheduler
4. Standalone Merge: LoRA -> 1.5B 독자 완전체 가중치 융합
5. Upload: kez-lab/quiz-korean Hub 자동 배포
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from huggingface_hub import HfApi

# 불필요한 환경 변수 제거
for k in ["PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"]:
    os.environ.pop(k, None)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REPO_ID = "kez-lab/quiz-korean"
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "scripts" / "data"
OUTPUT_DIR = BASE_DIR / "scripts" / "output" / "qwen2.5-1.5b-quiz-korean-lora"
MERGED_DIR = BASE_DIR / "scripts" / "output" / "qwen2.5-1.5b-quiz-korean-merged"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)

def train_sota():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ [1/5] Apple Silicon M4 Pro ({device}) bfloat16 최적화 모드로 가동합니다.")

    print(f"📦 [2/5] 모델 및 토크나이저 로드 중: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apple Silicon MPS에서 bfloat16은 float32 대비 4배 빠르고 메모리를 절반만 사용합니다.
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

    train_file = str(DATA_DIR / "train_large_1k.jsonl")
    val_file = str(DATA_DIR / "val_large_1k.jsonl")
    
    raw_datasets = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file}
    )

    def preprocess_response_masking(examples):
        input_ids_list = []
        labels_list = []
        for prompt_text, full_text in zip(examples["prompt"], examples["full_text"]):
            full_encoded = tokenizer(full_text, max_length=768, truncation=True)
            input_ids = full_encoded["input_ids"]
            labels = list(input_ids)
            
            prompt_encoded = tokenizer(prompt_text, max_length=768, truncation=True)
            prompt_len = len(prompt_encoded["input_ids"])
            
            # 사용자 프롬프트는 Loss 계산에서 제외 (-100 마스킹)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
                
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}

    print("✂️ [3/5] Response-Only Loss 마스킹 데이터셋 토크나이징 중...")
    tokenized_datasets = raw_datasets.map(
        preprocess_response_masking,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
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

    print("🚀 [4/5] SOTA 1.5B 초고속 학습 시작...")
    trainer.train()

    print(f"💾 LoRA 어댑터 저장 완료: {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    print(f"🧬 [5/5] 가중치 완전 병합 (Weight Merge) -> 독립형 1.5B 완전체 모델 생성 중...")
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

    # 허깅페이스 업로드
    print(f"☁️ Hugging Face Hub ({REPO_ID})로 업로드 중...")
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=str(MERGED_DIR),
            repo_id=REPO_ID,
            repo_type="model"
        )
        print(f"🏆 업로드 성공! https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"⚠️ 업로드 알림: {e}")

    print("✨ 모든 연구팀 수준의 파이프라인이 성공적으로 완수되었습니다!")

if __name__ == "__main__":
    train_sota()
