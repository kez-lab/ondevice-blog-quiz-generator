#!/usr/bin/env python3
"""
🚀 온디바이스 블로그 퀴즈 AI 전용 고성능 웹 서버 (FastAPI)
- Mac M4 Pro Metal MPS GPU 안전 가속
- 정교한 4지선다 퀴즈 파싱 및 비동기 REST API 제공
"""

import os
import gc
import json
import re
import asyncio
from pathlib import Path

# 불필요하거나 충돌을 일으키는 환경 변수 제거
for k in ["PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"]:
    os.environ.pop(k, None)

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="On-Device Quiz AI Playground")

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = Path(__file__).parent / "static"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = BASE_DIR / "scripts" / "output" / "qwen2.5-0.5b-blog-quiz-lora"

SYSTEM_PROMPT = (
    "당신은 주어진 글(블로그, 아티클, 문서)을 분석하여 핵심 내용을 묻는 4지선다 객관식 퀴즈를 생성하는 AI입니다.\n"
    "반드시 아래 JSON 배열 형식으로만 응답하세요:\n"
    "[\n"
    "  {\n"
    '    "question": "문제 내용",\n'
    '    "options": ["보기1", "보기2", "보기3", "보기4"],\n'
    '    "answer_index": 0,\n'
    '    "explanation": "정답에 대한 명확한 해설"\n'
    "  }\n"
    "]"
)

# 전역 모델 객체 및 단일 추론 락
model = None
tokenizer = None
device = "mps" if torch.backends.mps.is_available() else "cpu"
inference_lock = asyncio.Lock()

def get_model():
    global model, tokenizer
    if model is None:
        print(f"📦 [Server] 베이스 모델 로드 중: {MODEL_ID} (Device: {device})...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            trust_remote_code=True
        ).to(device)
        
        if LORA_PATH.exists() and (LORA_PATH / "adapter_model.safetensors").exists():
            print(f"🎯 [Server] LoRA 어댑터 결합 중: {LORA_PATH}")
            model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
        else:
            print("💡 베이스 모델 단독 모드로 실행합니다.")
            model = base_model
            
        model.eval()
        print("✅ [Server] 모델 로드 완료!")
    return model, tokenizer

def robust_parse_quizzes(raw_text: str):
    """3중 방어 JSON 퀴즈 파서"""
    match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
    if match:
        clean_str = match.group(0)
        clean_str = re.sub(r'\"\)', '\"', clean_str)
        clean_str = re.sub(r',\s*\]', ']', clean_str)
        clean_str = re.sub(r',\s*\}', '}', clean_str)
        try:
            return json.loads(clean_str)
        except Exception:
            pass

    # 정규식 기반 폴백
    quizzes = []
    blocks = re.findall(r'\{[^{}]*\"question\"[^{}]*\}', raw_text, re.DOTALL)
    for b in blocks:
        q_m = re.search(r'\"question\"\s*:\s*\"([^\"]+)\"', b)
        exp_m = re.search(r'\"explanation\"\s*:\s*\"([^\"]+)\"', b)
        ans_m = re.search(r'\"answer_index\"\s*:\s*(\d+)', b)
        opt_m = re.search(r'\"options\"\s*:\s*\[([^\]]+)\]', b)

        if q_m:
            question = q_m.group(1)
            explanation = exp_m.group(1) if exp_m else ""
            ans_idx = int(ans_m.group(1)) if ans_m else 0
            opts = [o.replace('"', '').strip() for o in opt_m.group(1).split(',')] if opt_m else ["보기 1", "보기 2", "보기 3", "보기 4"]
            quizzes.append({
                "question": question,
                "options": opts,
                "answer_index": ans_idx,
                "explanation": explanation
            })
    return quizzes

class GenerateRequest(BaseModel):
    article: str
    count: int = 3
    temperature: float = 0.1

@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/api/generate")
async def generate_quiz(req: GenerateRequest):
    if len(req.article.strip()) < 20:
        raise HTTPException(status_code=400, detail="최소 20자 이상의 글을 입력해주세요.")

    async with inference_lock:
        try:
            m, tok = get_model()
            
            user_prompt = f"다음 블로그 글을 읽고 핵심 내용에 대한 4지선다 객관식 퀴즈 {req.count}문제를 JSON 배열로 만들어주세요:\n\n{req.article}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = m.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=max(0.01, req.temperature),
                    top_p=0.9,
                    repetition_penalty=1.15,
                    do_sample=True,
                    pad_token_id=tok.eos_token_id
                )
                
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            raw_output = tok.decode(generated_ids, skip_special_tokens=True).strip()
            
            # GPU 캐시 즉시 비우기 (메모리 누수 방지)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
            quizzes = robust_parse_quizzes(raw_output)
            
            return {
                "success": True,
                "count": len(quizzes),
                "quizzes": quizzes,
                "raw_json": raw_output
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    print("🚀 On-Device Quiz AI 웹 서버 시작 (http://127.0.0.1:8000)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
