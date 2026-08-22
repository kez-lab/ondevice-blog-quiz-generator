#!/usr/bin/env python3
"""
🏆 [Phase 8] LoRA V0 완전체 모델 벤치마크 평가 러너
- 모델: scripts/output/qwen2.5-1.5b-v0-merged
- 테스트셋: scripts/data/test_benchmark_100.jsonl (학습에 일절 사용되지 않은 100개 격리 문서)
- B1 베이스라인과의 QuizScore, Groundedness, 생성 속도 정량 비교
"""

import json
import re
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "scripts" / "output" / "qwen2.5-1.5b-v0-merged"
TEST_FILE = BASE_DIR / "scripts" / "data" / "test_benchmark_100.jsonl"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_quiz_sample(quiz_item, source_article):
    """QuizScore = 0.30 G + 0.25 U + 0.20 D + 0.15 I + 0.10 L"""
    q_text = quiz_item.get("question", "")
    options = quiz_item.get("options", [])
    ans_idx = quiz_item.get("answer_index", -1)
    evidence = quiz_item.get("evidence", "")

    if not (isinstance(options, list) and len(options) == 4):
        return 0.0, "DROP: 선택지 수 오류"
    if ans_idx < 0 or ans_idx > 3:
        return 0.0, "DROP: 정답 인덱스 범위 초과"
    if len(set(options)) < 4:
        return 0.0, "DROP: 중복 선택지"

    correct_option = options[ans_idx]

    # 1. Groundedness (0.30)
    grounded_score = 0.0
    if evidence and (evidence in source_article or any(w in source_article for w in evidence.split() if len(w) > 2)):
        grounded_score = 1.0
    elif correct_option in source_article:
        grounded_score = 0.8
    else:
        grounded_score = 0.2

    # 2. Answer Uniqueness (0.25)
    uniqueness_score = 1.0
    for i, opt in enumerate(options):
        if i != ans_idx and opt.strip() == correct_option.strip():
            uniqueness_score = 0.0

    # 3. Distractor Quality (0.20)
    distractor_score = 1.0
    for opt in options:
        if re.search(r'option\s*[a-d]|보기\s*[1-4]', opt, re.IGNORECASE):
            distractor_score = 0.1

    # 4. Importance (0.15)
    importance_score = 0.9 if len(q_text) >= 15 else 0.5

    # 5. Language Quality (0.10)
    language_score = 1.0
    if re.search(r'[\u0400-\u04FF\u0590-\u05FF]', q_text):
        language_score = 0.0

    total_score = (
        0.30 * grounded_score +
        0.25 * uniqueness_score +
        0.20 * distractor_score +
        0.15 * importance_score +
        0.10 * language_score
    )
    return total_score, "VALID"

def run_lora_v0_benchmark(sample_limit=5):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n========================================================")
    print(f"🚀 [LoRA V0 Merged] 벤치마크 평가 시작 (대상: {sample_limit}개 문서)...")
    print(f"========================================================")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    model.eval()

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_docs = [json.loads(line) for line in f][:sample_limit]

    results = []
    total_scores = []
    latencies = []

    for i, doc in enumerate(test_docs, 1):
        article_text = doc["content"]

        # LoRA 모델은 시스템 프롬프트 없이 짧은 Conversational 입력으로 평가
        messages = [
            {"role": "system", "content": "주어진 글만을 근거로 객관식 학습 문제를 생성한다."},
            {"role": "user", "content": f"ARTICLE:\n{article_text}"}
        ]

        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

        start_t = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.01,
                repetition_penalty=1.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - start_t
        latencies.append(elapsed)

        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        quizzes = []
        try:
            # JSON 객체 또는 배열 파싱
            parsed_json = json.loads(raw_output)
            if isinstance(parsed_json, dict) and "questions" in parsed_json:
                quizzes = parsed_json["questions"]
            elif isinstance(parsed_json, list):
                quizzes = parsed_json
        except Exception:
            match = re.search(r'\[\s*\{.*\}\s*\]', raw_output, re.DOTALL)
            if match:
                try:
                    quizzes = json.loads(match.group(0))
                except Exception:
                    pass

        doc_scores = []
        for q in quizzes:
            score, status = evaluate_quiz_sample(q, article_text)
            doc_scores.append(score)

        avg_doc_score = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
        total_scores.append(avg_doc_score)

        print(f"[{i}/{sample_limit}] {doc['title'][:30]}... ➡️ 문항수: {len(quizzes)}, 평균 QuizScore: {avg_doc_score:.3f} (소요시간: {elapsed:.2f}s)")

        results.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "domain": doc["domain"],
            "quiz_count": len(quizzes),
            "quiz_score": avg_doc_score,
            "latency_seconds": elapsed,
            "raw_output": raw_output
        })

    final_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print(f"\n========================================================")
    print(f"🏆 [LoRA V0 Merged] 종합 평균 QuizScore: {final_score:.4f} (1.000 만점)")
    print(f"⚡ [LoRA V0 Merged] 평균 추론 속도: {avg_latency:.2f}초")
    print(f"========================================================\n")

    out_json = RESULTS_DIR / "evaluation_lora_v0_merged.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": "qwen2.5-1.5b-v0-merged",
            "final_quiz_score": final_score,
            "avg_latency_seconds": avg_latency,
            "details": results
        }, f, ensure_ascii=False, indent=2)

    print(f"📁 결과 저장 완료: {out_json}")

if __name__ == "__main__":
    run_lora_v0_benchmark(sample_limit=5)
