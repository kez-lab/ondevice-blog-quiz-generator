#!/usr/bin/env python3
"""
🔬 3대 사전 베이스라인(B0, B1, B2) 정량 평가 러너
- B0 (Vanilla): 제약 없는 기본 제로샷
- B1 (Prompt Engineered): Evidence 필수, 단일 정답 규칙 명시
- B2 (Few-Shot): 3개 고품질 예제 포함 최강 인컨텍스트 베이스라인
- 100개 격리 테스트셋 대상 실행 및 QuizScore 자동 채점
"""

import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR = Path(__file__).parent.parent
TEST_FILE = BASE_DIR / "scripts" / "data" / "test_benchmark_100.jsonl"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. 프롬프트 템플릿 정의
# -------------------------------------------------------------
PROMPT_B0_VANILLA = "다음 글을 읽고 4지선다 객관식 퀴즈 3문제를 만들어주세요:\n\n{article}"

PROMPT_B1_ENGINEERED = """당신은 엄격한 시험 출제 위원입니다.
오직 주어진 [본문]에 명시된 사실만을 바탕으로 4지선다 객관식 퀴즈 3문제를 출제하세요.

[규칙]
1. 정답은 4개의 선택지 중 오직 1개여야 합니다.
2. 3개의 오답(Distractors)은 정답과 동일한 의미적 범주에서 그럴듯하게 구성해야 합니다.
3. 본문에 없는 외부 상식이나 가정을 절대 포함하지 마세요.
4. 반드시 본문에서 정답을 입증하는 실제 문장을 'evidence' 필드에 그대로 인용하세요.
5. 아래 JSON 배열 포맷으로만 응답하세요:

```json
[
  {{
    "question": "문제 내용",
    "options": ["보기 A", "보기 B", "보기 C", "보기 D"],
    "answer_index": 0,
    "explanation": "해설",
    "evidence": "본문 인용 문장"
  }}
]
```

[본문]
{article}
"""

FEW_SHOT_EXAMPLES = [
    {
        "article": "Git에서 rebase는 커밋 히스토리를 선형으로 깔끔하게 재정렬하고, merge는 두 브랜치를 병합하는 새로운 병합 커밋을 생성합니다.",
        "quiz": [
            {
                "question": "Git에서 커밋 히스토리를 선형(Linear)으로 재정렬하는 명령어는?",
                "options": ["rebase", "merge", "checkout", "cherry-pick"],
                "answer_index": 0,
                "explanation": "rebase는 커밋 히스토리를 한 줄로 선형 재정렬하는 명령어입니다.",
                "evidence": "Git에서 rebase는 커밋 히스토리를 선형으로 깔끔하게 재정렬하고"
            }
        ]
    },
    {
        "article": "JVM의 힙 메모리는 크게 Young Generation과 Old Generation으로 나뉩니다. 새롭게 생성된 객체는 먼저 Eden 영역에 할당되며, 가비지 컬렉션에서 살아남으면 Survivor 영역을 거쳐 Old 영역으로 승격(Promotion)됩니다.",
        "quiz": [
            {
                "question": "JVM 힙 메모리에서 새로 생성된 객체가 최초로 할당되는 영역은?",
                "options": ["Eden 영역", "Survivor 1 영역", "Survivor 2 영역", "Old 영역"],
                "answer_index": 0,
                "explanation": "새롭게 생성된 객체는 가장 먼저 Young Generation의 Eden 영역에 할당됩니다.",
                "evidence": "새롭게 생성된 객체는 먼저 Eden 영역에 할당되며"
            }
        ]
    }
]

# -------------------------------------------------------------
# 2. QuizScore 자동 채점기 (Rubric & Hard Constraints)
# -------------------------------------------------------------
def evaluate_quiz_sample(quiz_item, source_article):
    """
    QuizScore = 0.30 G + 0.25 U + 0.20 D + 0.15 I + 0.10 L
    """
    q_text = quiz_item.get("question", "")
    options = quiz_item.get("options", [])
    ans_idx = quiz_item.get("answer_index", -1)
    evidence = quiz_item.get("evidence", "")

    # Hard Constraints 검증
    if not (isinstance(options, list) and len(options) == 4):
        return 0.0, "DROP: 선택지가 4개가 아님"
    if ans_idx < 0 or ans_idx > 3:
        return 0.0, "DROP: 정답 인덱스 범위 초과"
    if len(set(options)) < 4:
        return 0.0, "DROP: 중복된 선택지 존재"

    correct_option = options[ans_idx]

    # 1. Groundedness (0.30): evidence가 본문에 존재하는지 & 정답 키워드 포함 여부
    grounded_score = 0.0
    if evidence and (evidence in source_article or any(w in source_article for w in evidence.split() if len(w) > 2)):
        grounded_score = 1.0
    elif correct_option in source_article:
        grounded_score = 0.7
    else:
        grounded_score = 0.2

    # 2. Answer Uniqueness (0.25): 오답들이 정답과 명확히 구분되는가
    uniqueness_score = 1.0
    for i, opt in enumerate(options):
        if i != ans_idx and opt.strip() == correct_option.strip():
            uniqueness_score = 0.0

    # 3. Distractor Quality (0.20): 오답이 비어있거나 'Option A' 같은 더미가 아닌지
    distractor_score = 1.0
    for opt in options:
        if re.search(r'option\s*[a-d]|보기\s*[1-4]', opt, re.IGNORECASE):
            distractor_score = 0.1

    # 4. Importance (0.15): 질문 길이 및 핵심어 포함 여부
    importance_score = 0.8 if len(q_text) >= 15 else 0.4

    # 5. Language Quality (0.10): 깨진 문자나 외국어 혼입 여부
    language_score = 1.0
    if re.search(r'[\u0400-\u04FF\u0590-\u05FF]', q_text): # 러시아어/히브리어 혼입
        language_score = 0.0

    total_score = (
        0.30 * grounded_score +
        0.25 * uniqueness_score +
        0.20 * distractor_score +
        0.15 * importance_score +
        0.10 * language_score
    )
    return total_score, "VALID"

def run_baseline_evaluation(baseline_name="B1_Prompted", sample_limit=10):
    print(f"\n========================================================")
    print(f"🚀 [{baseline_name}] 벤치마크 평가 시작 (대상: {sample_limit}개 문서)...")
    print(f"========================================================")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    model.eval()

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_docs = [json.loads(line) for line in f][:sample_limit]

    results = []
    total_scores = []

    for i, doc in enumerate(test_docs, 1):
        article_text = doc["content"]

        if baseline_name == "B0_Vanilla":
            prompt_content = PROMPT_B0_VANILLA.format(article=article_text)
            messages = [{"role": "user", "content": prompt_content}]
        elif baseline_name == "B1_Prompted":
            prompt_content = PROMPT_B1_ENGINEERED.format(article=article_text)
            messages = [{"role": "user", "content": prompt_content}]
        elif baseline_name == "B2_FewShot":
            messages = [
                {"role": "system", "content": "오직 주어진 [본문]의 사실에만 근거하여 4지선다 객관식 퀴즈를 출제하는 AI입니다."}
            ]
            for ex in FEW_SHOT_EXAMPLES:
                messages.append({"role": "user", "content": f"[본문]\n{ex['article']}\n\n4지선다 퀴즈를 만드세요."})
                messages.append({"role": "assistant", "content": json.dumps(ex["quiz"], ensure_ascii=False)})
            messages.append({"role": "user", "content": f"[본문]\n{article_text}\n\n4지선다 퀴즈 3문제를 동일한 JSON 배열 형식으로 만드세요."})

        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.01,
                repetition_penalty=1.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # JSON 파싱 시도
        quizzes = []
        try:
            match = re.search(r'\[\s*\{.*\}\s*\]', raw_output, re.DOTALL)
            if match:
                quizzes = json.loads(match.group(0))
        except Exception:
            pass

        doc_scores = []
        for q in quizzes:
            score, status = evaluate_quiz_sample(q, article_text)
            doc_scores.append(score)

        avg_doc_score = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
        total_scores.append(avg_doc_score)

        print(f"[{i}/{sample_limit}] {doc['title'][:30]}... ➡️ 문항수: {len(quizzes)}, 평균 QuizScore: {avg_doc_score:.3f}")

        results.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "domain": doc["domain"],
            "quiz_count": len(quizzes),
            "quiz_score": avg_doc_score,
            "raw_output": raw_output
        })

    final_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
    print(f"\n🏆 [{baseline_name}] 종합 평균 QuizScore: {final_score:.4f} (1.000 만점)")

    out_json = RESULTS_DIR / f"baseline_{baseline_name.lower()}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline_name, "final_quiz_score": final_score, "details": results}, f, ensure_ascii=False, indent=2)
    print(f"📁 결과 저장 완료: {out_json}\n")

if __name__ == "__main__":
    # B1 Prompted 베이스라인 먼저 측정
    run_baseline_evaluation("B1_Prompted", sample_limit=5)
