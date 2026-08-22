#!/usr/bin/env python3
"""
🚀 Google Gemma-2-2B V1 SFT 온디바이스 퀴즈 웹 서버 (FastAPI + SSE 스트리밍 + 영구 로그 저장소)
- Model: scripts/output/gemma-2-2b-v1-merged (2.6B 파라미터 완전체)
- Logs: logs/server_activity.log, logs/inference_history.jsonl, logs/errors.log
- SSE (Server-Sent Events) 실시간 문항별 즉시 스트리밍
- Thought 기반 근거(Evidence) 및 해설(Explanation) 자동 보강
- Mac M4 Pro Metal MPS bfloat16 안전 가속
"""

import os
import gc
import json
import re
import time
import logging
import traceback
import asyncio
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import AsyncGenerator

# 워터마크 환경변수 충돌 방지
for k in ["PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"]:
    os.environ.pop(k, None)

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = Path(__file__).parent / "static"
LOGS_DIR = BASE_DIR / "logs"
MODEL_PATH = BASE_DIR / "scripts" / "output" / "gemma-2-2b-v1-merged"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
SERVER_LOG_FILE = LOGS_DIR / "server_activity.log"
INFERENCE_JSONL_FILE = LOGS_DIR / "inference_history.jsonl"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"

# ==================== [로깅 시스템 구축] ====================
logger = logging.getLogger("QuizServer")
logger.setLevel(logging.INFO)

# 1. 파일 회전 로거 (최대 10MB, 5개 보관)
file_handler = RotatingFileHandler(
    SERVER_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
file_handler.setFormatter(
    logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(file_handler)

# 2. 콘솔 출력 로거
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
)
logger.addHandler(console_handler)

def log_inference_event(event_type: str, data: dict):
    """모든 추론 요청, 입력 글, 원본 CoT 출력, 파싱 결과를 JSONL로 영구 저장"""
    try:
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data
        }
        with open(INFERENCE_JSONL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"로그 파일 기록 실패: {e}")

def log_error_event(context_msg: str, exc: Exception):
    """에러 발생 시 상세 트레이스백을 errors.log에 영구 기록"""
    tb_str = traceback.format_exc()
    logger.error(f"{context_msg}: {exc}\n{tb_str}")
    try:
        err_entry = (
            f"\n{'='*60}\n"
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {context_msg}\n"
            f"Exception: {str(exc)}\n"
            f"Traceback:\n{tb_str}\n"
            f"{'='*60}\n"
        )
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(err_entry)
    except Exception as e:
        logger.error(f"에러 로그 파일 기록 실패: {e}")

# ==================== [FastAPI 앱 및 모델 엔진] ====================
app = FastAPI(title="Gemma-2-2B SOTA Quiz AI Playground with Persistent Logging")

model = None
tokenizer = None
device = "mps" if torch.backends.mps.is_available() else "cpu"
inference_lock = asyncio.Lock()

def get_model():
    global model, tokenizer
    if model is None:
        logger.info(f"📦 [Server] Gemma-2-2B V1 완전체 모델 로드 중: {MODEL_PATH} (Device: {device})...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,
            trust_remote_code=True
        ).to(device)
        model.eval()
        logger.info("✅ [Server] Google Gemma-2-2B V1 완전체 로드 완료!")
    return model, tokenizer

# 서버 시작 시 모델 사전 적재
get_model()

def split_into_coherent_sections(full_text: str, target_count: int = 3):
    """장문 아티클을 500~800자 핵심 섹션으로 스마트 분할"""
    full_text = full_text.strip()
    
    header_split = [s.strip() for s in re.split(r'\n(?=(?:#{1,3}\s+|\d+\.\s+))', full_text) if len(s.strip()) > 80]
    if len(header_split) >= target_count:
        step = len(header_split) // target_count
        return [header_split[i * step][:800] for i in range(target_count)]
    
    paras = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 60]
    if len(paras) >= target_count:
        chunk_size = len(paras) // target_count
        sections = []
        for i in range(target_count):
            start = i * chunk_size
            end = start + chunk_size if i < target_count - 1 else len(paras)
            sections.append("\n\n".join(paras[start:end])[:800])
        return sections

    chunk_len = len(full_text) // target_count
    if chunk_len < 100:
        return [full_text[:800]]
    return [full_text[i*chunk_len : (i+1)*chunk_len][:800] for i in range(target_count)]

def robust_parse_single_quiz(raw_text: str):
    """출력 문자열에서 퀴즈 객체 추출 및 thought 기반 근거 보강"""
    quiz_obj = None

    # 1. ```json ... ``` 블록 추출
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if "questions" in parsed and len(parsed["questions"]) > 0:
                quiz_obj = parsed["questions"][0]
        except Exception:
            pass

    # 2. JSON 배열
    if not quiz_obj:
        match_arr = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
        if match_arr:
            try:
                arr = json.loads(match_arr.group(0))
                if len(arr) > 0:
                    quiz_obj = arr[0]
            except Exception:
                pass

    # 3. 정규식 추출 (이스케이프된 따옴표 \" 완벽 지원)
    if not quiz_obj:
        q_m = re.search(r'\"question\"\s*:\s*\"((?:\\.|[^\"\\])*)\"', raw_text)
        exp_m = re.search(r'\"explanation\"\s*:\s*\"((?:\\.|[^\"\\])*)\"', raw_text)
        evi_m = re.search(r'\"evidence\"\s*:\s*\"((?:\\.|[^\"\\])*)\"', raw_text)
        ans_m = re.search(r'\"answer_index\"\s*:\s*(\d+)', raw_text)
        opt_m = re.search(r'\"options\"\s*:\s*\[((?:\\.|[^\]])+)\]', raw_text)

        if q_m:
            raw_opts = opt_m.group(1) if opt_m else ""
            opts = [re.sub(r'^[\"\s]+|[\"\s]+$', '', o).replace('\\"', '"') for o in re.findall(r'\"((?:\\.|[^\"\\])*)\"', raw_opts)]
            if not opts:
                opts = ["선택지 A", "선택지 B", "선택지 C", "선택지 D"]
            while len(opts) < 4:
                opts.append(f"선택지 {len(opts)+1}")
            quiz_obj = {
                "question": q_m.group(1).replace('\\"', '"').replace('\\\\', '\\').strip(),
                "options": opts[:4],
                "answer_index": min(int(ans_m.group(1)) if ans_m else 0, 3),
                "explanation": exp_m.group(1).replace('\\"', '"').strip() if exp_m else "",
                "evidence": evi_m.group(1).replace('\\"', '"').strip() if evi_m else ""
            }

    # 4. <thought> 블록에서 Evidence, 정답, 4개 선택지 및 해설 자동 보강
    if quiz_obj:
        thought_q = re.search(r'2\.\s*출제 질문[^\n]*\n\s*\"((?:\\.|[^\"\\])*)\"', raw_text)
        thought_evi = re.search(r'1\.\s*본문 핵심 근거[^\n]*\n\s*\"((?:\\.|[^\"\\])*)\"', raw_text)
        thought_ans = re.search(r'3\.\s*명확한 단일 정답[^\n]*\n\s*\"((?:\\.|[^\"\\])*)\"', raw_text)
        thought_d1 = re.search(r'오답 1\s*:\s*([^\n]+)', raw_text)
        thought_d2 = re.search(r'오답 2\s*:\s*([^\n]+)', raw_text)
        thought_d3 = re.search(r'오답 3\s*:\s*([^\n]+)', raw_text)
        thought_idx_m = re.search(r'정답을\s*([A-D])\s*번', raw_text)

        # 질문이 잘렸거나 너무 짧을 때 thought의 완성된 질문으로 복원
        if (not quiz_obj.get("question") or len(quiz_obj.get("question", "").strip()) < 10 or quiz_obj.get("question", "").endswith("\\")) and thought_q:
            quiz_obj["question"] = thought_q.group(1).replace('\\"', '"').strip()

        # 선택지가 더미("선택지 A")이거나 잘렸을 경우 thought의 정답 및 오답 3개로 완전 재구성
        has_dummy_opts = any("선택지 " in opt for opt in quiz_obj.get("options", []))
        if has_dummy_opts and thought_ans and thought_d1 and thought_d2 and thought_d3:
            ans_text = thought_ans.group(1).strip()
            d1_text = thought_d1.group(1).strip()
            d2_text = thought_d2.group(1).strip()
            d3_text = thought_d3.group(1).strip()
            
            ans_idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(thought_idx_m.group(1).upper() if thought_idx_m else "B", 1)
            distractors = [d1_text, d2_text, d3_text]
            new_options = []
            for i in range(4):
                if i == ans_idx:
                    new_options.append(ans_text)
                else:
                    new_options.append(distractors.pop(0) if distractors else f"대체 보기 {i+1}")
            quiz_obj["options"] = new_options
            quiz_obj["answer_index"] = ans_idx

        if not quiz_obj.get("evidence") and thought_evi:
            quiz_obj["evidence"] = thought_evi.group(1).strip()

        if not quiz_obj.get("explanation"):
            target_str = thought_ans.group(1).strip() if thought_ans else ""
            if target_str and quiz_obj.get("evidence"):
                quiz_obj["explanation"] = f"본문에서 '{quiz_obj['evidence'][:50]}...'라고 언급되어 있으므로, '{target_str[:50]}'을 설명하는 {chr(65 + quiz_obj.get('answer_index', 0))}번이 정답입니다."
            elif quiz_obj.get("evidence"):
                quiz_obj["explanation"] = f"본문의 '{quiz_obj['evidence'][:60]}...' 문장을 근거로 {chr(65 + quiz_obj.get('answer_index', 0))}번이 정답입니다."
            else:
                quiz_obj["explanation"] = f"본문 내용의 핵심 논리에 따라 {chr(65 + quiz_obj.get('answer_index', 0))}번이 정답입니다."

    return quiz_obj

class GenerateRequest(BaseModel):
    article: str
    count: int = 3
    temperature: float = 0.01

@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/logs/latest")
async def get_latest_logs(lines: int = 50):
    """최근 서버 활동 로그 반환"""
    try:
        if not SERVER_LOG_FILE.exists():
            return {"logs": []}
        with open(SERVER_LOG_FILE, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            return {"logs": [line.strip() for line in all_lines[-lines:]]}
    except Exception as e:
        return {"error": str(e)}

def generate_single_quiz_sync(sec_text: str, s_idx: int, total_count: int):
    """동기식 단일 문항 추론 및 시간 측정"""
    start_t = time.time()
    m, tok = get_model()
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
            max_new_tokens=768,
            temperature=0.01,
            repetition_penalty=1.1,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )

    gen_ids = outputs[0][inputs.input_ids.shape[1]:]
    raw_out = tok.decode(gen_ids, skip_special_tokens=True).strip()
    latency = time.time() - start_t

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    q_obj = robust_parse_single_quiz(raw_out)
    return q_obj, raw_out, latency

@app.post("/api/generate/stream")
async def generate_quiz_stream(req: GenerateRequest, request: Request):
    """SSE (Server-Sent Events) 스트리밍 + 상세 요청 로깅"""
    client_ip = request.client.host if request.client else "unknown"
    article_len = len(req.article.strip())
    
    if article_len < 20:
        logger.warning(f"[{client_ip}] 요청 거부: 본문 길이 부족 ({article_len}자)")
        raise HTTPException(status_code=400, detail="최소 20자 이상의 글을 입력해주세요.")

    target_count = max(1, min(req.count, 5))
    sections = split_into_coherent_sections(req.article, target_count=target_count)

    logger.info(f"📥 [{client_ip}] 신규 SSE 요청: 본문 {article_len}자 ➡️ {len(sections)}개 섹션으로 {target_count}문제 생성")

    async def event_generator() -> AsyncGenerator[str, None]:
        req_start_t = time.time()
        quizzes_generated = []
        raw_outputs = []

        async with inference_lock:
            if await request.is_disconnected():
                logger.info(f"🛑 [{client_ip}] 락 획득 전 브라우저 연결 종료 감지 ➡️ 요청 취소")
                return

            logger.info(f"📥 [{client_ip}] SSE 추론 시작: 본문 {article_len}자 ➡️ {len(sections)}개 섹션으로 {target_count}문제 생성")
            yield f"data: {json.dumps({'type': 'init', 'total': len(sections)})}\n\n"
            
            for s_idx, sec in enumerate(sections, 1):
                if await request.is_disconnected():
                    logger.info(f"🛑 [{client_ip}] [Q{s_idx}/{len(sections)}] 도중 브라우저 연결 종료 감지 ➡️ 잔여 추론 즉시 중단")
                    break

                logger.info(f"🎯 [{client_ip}] [Q{s_idx}/{len(sections)}] CoT 추론 시작 ({len(sec)}자)...")
                yield f"data: {json.dumps({'type': 'progress', 'current': s_idx, 'total': len(sections), 'message': f'Q{s_idx} 생성 중 ({s_idx}/{len(sections)})...'})}\n\n"
                
                try:
                    q_obj, raw_out, q_lat = await asyncio.to_thread(generate_single_quiz_sync, sec, s_idx, len(sections))
                    raw_outputs.append(raw_out)
                    
                    if q_obj:
                        quizzes_generated.append(q_obj)
                        logger.info(f"✅ [{client_ip}] [Q{s_idx}/{len(sections)}] 생성 완료 ({q_lat:.2f}초): \"{q_obj.get('question')[:40]}...\" (정답: {chr(65+q_obj.get('answer_index',0))})")
                        yield f"data: {json.dumps({'type': 'quiz', 'index': s_idx, 'quiz': q_obj, 'raw_json': raw_out})}\n\n"
                    else:
                        logger.warning(f"⚠️ [{client_ip}] [Q{s_idx}/{len(sections)}] 파싱 실패")
                        
                except Exception as e:
                    log_error_event(f"[{client_ip}] Q{s_idx} 생성 도중 에러", e)

                await asyncio.sleep(0.05)

            total_lat = time.time() - req_start_t
            logger.info(f"🎉 [{client_ip}] 모든 문항 생성 완료! 총 {len(quizzes_generated)}개 (소요 시간: {total_lat:.2f}초)")
            
            # JSONL 영구 기록
            log_inference_event("stream_generation", {
                "client_ip": client_ip,
                "article_length": article_len,
                "article_preview": req.article[:200],
                "requested_count": target_count,
                "generated_count": len(quizzes_generated),
                "total_latency_sec": round(total_lat, 2),
                "quizzes": quizzes_generated,
                "raw_outputs": raw_outputs
            })

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/generate")
async def generate_quiz(req: GenerateRequest, request: Request):
    """기존 REST API 폴백 엔드포인트 + 로깅"""
    client_ip = request.client.host if request.client else "unknown"
    article_len = len(req.article.strip())
    
    if article_len < 20:
        raise HTTPException(status_code=400, detail="최소 20자 이상의 글을 입력해주세요.")

    target_count = max(1, min(req.count, 5))
    sections = split_into_coherent_sections(req.article, target_count=target_count)

    logger.info(f"📥 [{client_ip}] 신규 REST 요청: 본문 {article_len}자 ➡️ {len(sections)}개 섹션")
    req_start_t = time.time()

    async with inference_lock:
        all_quizzes = []
        raw_traces = []

        for s_idx, sec in enumerate(sections, 1):
            try:
                q_obj, raw_out, _ = await asyncio.to_thread(generate_single_quiz_sync, sec, s_idx, len(sections))
                raw_traces.append(raw_out)
                if q_obj:
                    all_quizzes.append(q_obj)
            except Exception as e:
                log_error_event(f"[{client_ip}] REST Q{s_idx} 에러", e)

        total_lat = time.time() - req_start_t
        log_inference_event("rest_generation", {
            "client_ip": client_ip,
            "article_length": article_len,
            "article_preview": req.article[:200],
            "requested_count": target_count,
            "generated_count": len(all_quizzes),
            "total_latency_sec": round(total_lat, 2),
            "quizzes": all_quizzes,
            "raw_outputs": raw_traces
        })

        return {
            "success": True,
            "count": len(all_quizzes),
            "quizzes": all_quizzes,
            "raw_json": "\n\n---\n\n".join(raw_traces)
        }

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    logger.info("🚀 Google Gemma-2-2B V1 실시간 스트리밍 & 영구 로깅 퀴즈 웹 서버 시작 (http://127.0.0.1:8000)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
