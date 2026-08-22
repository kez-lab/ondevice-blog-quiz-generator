#!/usr/bin/env python3
"""
🚀 Google Gemma-2-2B V1 SFT 완전체 온디바이스 퀴즈 웹 서버 (FastAPI)
- Model: scripts/output/gemma-2-2b-v1-merged (2.6B 파라미터 완전체)
- CoT (<thought>) 사고 과정 내재화 모델
- 다문항 (N문제) 섹션별 독립 심층 생성 지원
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

def split_into_coherent_sections(full_text: str, target_count: int = 3):
    """장문 아티클을 의미 있는 target_count개의 서브 섹션으로 분할"""
    full_text = full_text.strip()
    
    # 1. 마크다운 헤더(##, ###) 또는 번호 매김(1., 2.) 기준 분할
    header_split = [s.strip() for s in re.split(r'\n(?=(?:#{1,3}\s+|\d+\.\s+))', full_text) if len(s.strip()) > 80]
    if len(header_split) >= target_count:
        # 균등 간격으로 target_count개 선택
        step = len(header_split) // target_count
        return [header_split[i * step][:2000] for i in range(target_count)]
    
    # 2. 문단(\n\n) 기준 분할
    paras = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 60]
    if len(paras) >= target_count:
        chunk_size = len(paras) // target_count
        sections = []
        for i in range(target_count):
            start = i * chunk_size
            end = start + chunk_size if i < target_count - 1 else len(paras)
            sections.append("\n\n".join(paras[start:end])[:2000])
        return sections

    # 3. 단순 텍스트 길이 기준 분할
    chunk_len = len(full_text) // target_count
    if chunk_len < 100:
        return [full_text]
    return [full_text[i*chunk_len : (i+1)*chunk_len] for i in range(target_count)]

def robust_parse_single_quiz(raw_text: str):
    """단일 생성 출력에서 퀴즈 객체 추출"""
    # 1. ```json ... ``` 블록 추출
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if "questions" in parsed and len(parsed["questions"]) > 0:
                return parsed["questions"][0]
        except Exception:
            pass

    # 2. JSON 배열
    match_arr = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
    if match_arr:
        try:
            arr = json.loads(match_arr.group(0))
            if len(arr) > 0:
                return arr[0]
        except Exception:
            pass

    # 3. 정규식 추출
    q_m = re.search(r'\"question\"\s*:\s*\"([^\"]+)\"', raw_text)
    exp_m = re.search(r'\"explanation\"\s*:\s*\"([^\"]+)\"', raw_text)
    evi_m = re.search(r'\"evidence\"\s*:\s*\"([^\"]+)\"', raw_text)
    ans_m = re.search(r'\"answer_index\"\s*:\s*(\d+)', raw_text)
    opt_m = re.search(r'\"options\"\s*:\s*\[([^\]]+)\]', raw_text)

    if q_m:
        opts = [o.replace('"', '').strip() for o in opt_m.group(1).split(',')] if opt_m else ["보기 A", "보기 B", "보기 C", "보기 D"]
        while len(opts) < 4:
            opts.append(f"선택지 {len(opts)+1}")
        return {
            "question": q_m.group(1),
            "options": opts[:4],
            "answer_index": min(int(ans_m.group(1)) if ans_m else 0, 3),
            "explanation": exp_m.group(1) if exp_m else "",
            "evidence": evi_m.group(1) if evi_m else ""
        }
    return None

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

    target_count = max(1, min(req.count, 5))

    async with inference_lock:
        try:
            m, tok = get_model()
            sections = split_into_coherent_sections(req.article, target_count=target_count)
            print(f"\n==================== [새로운 웹 요청 도착] ====================")
            print(f"📄 [입력 텍스트 ({len(req.article)}자) ➡️ {len(sections)}개 섹션으로 분할하여 {target_count}문제 심층 생성]")

            all_quizzes = []
            raw_traces = []

            for s_idx, sec_text in enumerate(sections, 1):
                print(f"🎯 [섹션 {s_idx}/{len(sections)}] CoT 추론 진행 중 ({len(sec_text)}자)...")
                messages = [
                    {
                        "role": "user",
                        "content": f"주어진 글만을 근거로 핵심 개념을 분석하고 4지선다 객관식 문제를 생성하라.\n\n[ARTICLE]\n{sec_text}"
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

                gen_ids = outputs[0][inputs.input_ids.shape[1]:]
                raw_out = tok.decode(gen_ids, skip_special_tokens=True).strip()
                raw_traces.append(raw_out)

                q_obj = robust_parse_single_quiz(raw_out)
                if q_obj:
                    if not q_obj.get("explanation") and q_obj.get("evidence"):
                        q_obj["explanation"] = f"본문의 '{q_obj['evidence'][:60]}...' 문장을 근거로 {chr(65+q_obj.get('answer_index', 0))}번이 정답입니다."
                    all_quizzes.append(q_obj)

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

            print(f"🎉 [성공] 총 {len(all_quizzes)}개의 다문항 퀴즈 생성 완료!")
            print("=================================================================\n")

            return {
                "success": True,
                "count": len(all_quizzes),
                "quizzes": all_quizzes,
                "raw_json": "\n\n---\n\n".join(raw_traces)
            }

        except Exception as e:
            print(f"❌ [에러 발생]: {e}")
            raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    print("🚀 Google Gemma-2-2B V1 온디바이스 퀴즈 웹 서버 시작 (http://127.0.0.1:8000)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
