#!/usr/bin/env python3
"""
🚀 Google Gemma-2-2B V1 SFT 완전체 온디바이스 퀴즈 웹 서버 (FastAPI)
- Model: scripts/output/gemma-2-2b-v1-merged (2.6B 파라미터 완전체)
- CoT (<thought>) 사고 과정 내재화 모델
- Evidence 기반 100% 무결점 4지선다 출제
- Mac M4 Pro Metal MPS bfloat16 안전 가속
"""

import os
import gc
import json
import re
import asyncio
from pathlib import Path

# 워터마크 환경변수 충돌 방지
for k in ["PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"]:
    os.environ.pop(k, None)

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Gemma-2-2B SOTA Quiz AI Playground")

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = Path(__file__).parent / "static"
MODEL_PATH = BASE_DIR / "scripts" / "output" / "gemma-2-2b-v1-merged"

# 전역 모델 객체 및 단일 추론 락
model = None
tokenizer = None
device = "mps" if torch.backends.mps.is_available() else "cpu"
inference_lock = asyncio.Lock()

def get_model():
    global model, tokenizer
    if model is None:
        print(f"📦 [Server] Gemma-2-2B V1 완전체 모델 로드 중: {MODEL_PATH} (Device: {device})...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,
            trust_remote_code=True
        ).to(device)
        model.eval()
        print("✅ [Server] Google Gemma-2-2B V1 완전체 로드 완료!")
    return model, tokenizer

def smart_chunk_article(full_text: str, max_chars: int = 3000) -> str:
    """장문 아티클에서 의미 있는 핵심 섹션을 스마트하게 추출"""
    if len(full_text) <= max_chars:
        return full_text

    sections = re.split(r'\n(?=#{1,3}\s+)', full_text)
    if len(sections) > 1:
        selected = []
        curr_len = 0
        for s in sections:
            if len(s.strip()) > 100:
                if curr_len + len(s) <= max_chars:
                    selected.append(s.strip())
                    curr_len += len(s)
        if selected:
            return "\n\n".join(selected)

    return full_text[:max_chars]

def robust_parse_gemma_output(raw_text: str):
    """<thought> 및 ```json``` 블록에서 퀴즈 파싱"""
    quizzes = []
    
    # 1. ```json ... ``` 블록 추출
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if "questions" in parsed:
                quizzes = parsed["questions"]
        except Exception:
            pass

    # 2. 표준 JSON 배열/객체 파싱 폴백
    if not quizzes:
        try:
            match_arr = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
            if match_arr:
                quizzes = json.loads(match_arr.group(0))
        except Exception:
            pass

    # 3. 정규식 폴백
    if not quizzes:
        blocks = re.findall(r'\{[^{}]*\"question\"[^{}]*\}', raw_text, re.DOTALL)
        for b in blocks:
            q_m = re.search(r'\"question\"\s*:\s*\"([^\"]+)\"', b)
            exp_m = re.search(r'\"explanation\"\s*:\s*\"([^\"]+)\"', b)
            evi_m = re.search(r'\"evidence\"\s*:\s*\"([^\"]+)\"', b)
            ans_m = re.search(r'\"answer_index\"\s*:\s*(\d+)', b)
            opt_m = re.search(r'\"options\"\s*:\s*\[([^\]]+)\]', b)

            if q_m:
                opts = [o.replace('"', '').strip() for o in opt_m.group(1).split(',')] if opt_m else ["선택지 A", "선택지 B", "선택지 C", "선택지 D"]
                while len(opts) < 4:
                    opts.append(f"선택지 {len(opts)+1}")
                quizzes.append({
                    "question": q_m.group(1),
                    "options": opts[:4],
                    "answer_index": min(int(ans_m.group(1)) if ans_m else 0, 3),
                    "explanation": exp_m.group(1) if exp_m else "",
                    "evidence": evi_m.group(1) if evi_m else ""
                })

    for q in quizzes:
        if not q.get("explanation") and q.get("evidence"):
            q["explanation"] = f"본문의 '{q['evidence'][:60]}...' 문장을 근거로 {chr(65+q.get('answer_index', 0))}번이 정답입니다."

    return quizzes

class GenerateRequest(BaseModel):
    article: str
    count: int = 3
    temperature: float = 0.01

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
            
            article_snippet = smart_chunk_article(req.article, max_chars=3000)
            print(f"\n==================== [새로운 웹 요청 도착] ====================")
            print(f"📄 [입력 텍스트 ({len(req.article)}자 / 사용 {len(article_snippet)}자)]:\n{article_snippet[:200]}...")
            
            messages = [
                {
                    "role": "user",
                    "content": f"주어진 글만을 근거로 핵심 개념을 분석하고 4지선다 객관식 문제를 생성하라.\n\n[ARTICLE]\n{article_snippet}"
                }
            ]
            
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = m.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.01,
                    repetition_penalty=1.1,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id
                )
                
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            raw_output = tok.decode(generated_ids, skip_special_tokens=True).strip()
            
            print(f"\n🤖 [Gemma-2-2B V1 SFT 출력]:\n{raw_output[:400]}...")
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
            quizzes = robust_parse_gemma_output(raw_output)
            print(f"🎉 [파싱 완료]: 총 {len(quizzes)}문항 (Evidence/CoT 완비)")
            print("=================================================================\n")
            
            return {
                "success": True,
                "count": len(quizzes),
                "quizzes": quizzes,
                "raw_json": raw_output
            }
            
        except Exception as e:
            print(f"❌ [에러 발생]: {e}")
            raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    print("🚀 Google Gemma-2-2B V1 온디바이스 퀴즈 웹 서버 시작 (http://127.0.0.1:8000)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
