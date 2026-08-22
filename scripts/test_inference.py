#!/usr/bin/env python3
"""
장문 블로그 글 다중 퀴즈 생성 로컬 추론 테스트 & 견고한 파서
"""

import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_ADAPTER_DIR = Path(__file__).parent / "output" / "qwen2.5-0.5b-blog-quiz-lora"

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

LONG_TEST_ARTICLE = (
    "[1. Kotlin Coroutine의 비동기 메커니즘]\n"
    "Kotlin 코루틴은 OS 네이티브 스레드를 매번 생성하는 대신, 기존 스레드를 차단(blocking)하지 않고 실행을 중단(suspend)했다가 재개(resume)하는 경량 스레드입니다. "
    "Dispatchers.IO는 네트워크 통신과 로컬 파일 I/O 작업에 특화되어 있으며, Dispatchers.Default는 대규모 데이터 정렬 등 CPU 집약적 연산에 최적화되어 있습니다.\n\n"
    "[2. 안드로이드 백그라운드 서비스 정책 (Android 14)]\n"
    "안드로이드 14(API 34)부터는 포그라운드 서비스(Foreground Service) 실행 시 매니페스트에 반드시 foregroundServiceType(예: location, mediaPlayback 등)을 명시해야 합니다. "
    "타입을 명시하지 않으면 SecurityException이 발생하여 앱이 강제 종료됩니다. 한편 지연 가능하고 영속적인 작업에는 WorkManager가 표준 권장 사항입니다.\n\n"
    "[3. 온디바이스 AI와 양자화 기법]\n"
    "온디바이스 AI는 클라우드 서버 없이 기기 내부의 NPU/GPU에서 모델을 직접 구동하여 개인정보 보호와 오프라인 실행을 보장합니다. "
    "32비트 부동소수점(FP32) 가중치를 4비트(INT4) 정수로 압축하는 양자화(Quantization)를 적용하여 모델 용량을 수백 메가바이트로 축소할 수 있습니다."
)

def robust_parse_quizzes(raw_text: str):
    """안드로이드 QuizJsonParser와 동일한 정제 및 복구 로직"""
    match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
    if not match:
        return []
    
    clean_str = match.group(0)
    # 문법 오류 보정 (닫는 괄호, 오탈자 등)
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
            opts = [o.replace('"', '').strip() for o in opt_m.group(1).split(',')] if opt_m else ["1", "2", "3", "4"]
            quizzes.append({
                "question": question,
                "options": opts,
                "answer_index": ans_idx,
                "explanation": explanation
            })
    return quizzes

def test_inference():
    print(f"📦 베이스 모델 및 토크나이저 로드 중: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"⚙️ 실행 디바이스: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        trust_remote_code=True
    ).to(device)

    # LoRA 어댑터 결합
    if LORA_ADAPTER_DIR.exists() and (LORA_ADAPTER_DIR / "adapter_model.safetensors").exists():
        print(f"🎯 방금 학습된 LoRA 어댑터 결합 중: {LORA_ADAPTER_DIR}")
        model = PeftModel.from_pretrained(model, str(LORA_ADAPTER_DIR))

    model.eval()

    user_prompt = f"다음 장문 블로그 글을 읽고 핵심 내용에 대한 4지선다 객관식 퀴즈 3문제를 JSON 배열로 만들어주세요:\n\n{LONG_TEST_ARTICLE}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\n🔍 장문 글 기반 다중 퀴즈(3문항) 추론 생성 중...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\n================ [AI 원본 출력] ================")
    print(response_text)
    print("================================================\n")

    quizzes = robust_parse_quizzes(response_text)
    if quizzes:
        print(f"🎉 [JSON 파싱 성공! 생성된 퀴즈 총 {len(quizzes)}문항]")
        for i, q in enumerate(quizzes, 1):
            print(f"\nQ{i}. {q.get('question')}")
            for idx, opt in enumerate(q.get('options', [])):
                mark = " (✅ 정답)" if idx == q.get('answer_index') else ""
                print(f"  {idx + 1}) {opt}{mark}")
            print(f"  💡 해설: {q.get('explanation')}")
    else:
        print("⚠️ 퀴즈 파싱 실패")

if __name__ == "__main__":
    test_inference()
