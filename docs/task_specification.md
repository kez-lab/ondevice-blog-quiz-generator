# 📑 Source-Grounded Multiple Choice Question (MCQ) Generation Task Specification

## 1. 태스크 정의 (Task Definition)
본 태스크는 **"주어진 원본 문서(Source Document) 안의 정보에만 엄격하게 근거(Grounded)하여, 답이 하나로 결정되는 교육적 가치가 높은 4지선다 객관식 문제를 생성하는 작업"**으로 정의한다.

## 2. 입출력 스키마 (Input / Output Schema)

### Input (입력)
* `article`: 원본 텍스트 (블로그 글, 기술 문서, 학술 아티클)
* `count`: 생성할 문제 수 (기본값: 3)
* `difficulty`: 문제 난이도 (`Level 1` ~ `Level 3`)

### Output (출력 JSON)
```json
{
  "questions": [
    {
      "question": "본문 내용에 근거한 구체적인 질문",
      "options": [
        "선택지 A (동일 범주)",
        "선택지 B (동일 범주)",
        "선택지 C (동일 범주)",
        "선택지 D (동일 범주)"
      ],
      "answer_index": 1,
      "explanation": "정답의 근거 및 각 오답이 틀린 이유에 대한 명쾌한 해설",
      "evidence": "본문에서 정답을 직접적으로 증명하는 실제 문장 (Hallucination 원천 차단)"
    }
  ]
}
```

---

## 3. Bloom's Taxonomy 기반 3단계 난이도 체계

1. **Level 1 — Recall (사실 기억)**:
   * 본문에 명시적으로 서술된 핵심 용어, 수치, 정의를 정확히 기억하고 있는지 확인.
2. **Level 2 — Understanding (개념/관계 이해)**:
   * 두 개 이상의 개념 간 인과관계, 아키텍처의 작동 메커니즘, 장단점 비교 이해.
3. **Level 3 — Application (원리 상황 적용)**:
   * 본문에서 학습한 원리를 구체적인 실무 시나리오(코드 버그 상황, 비즈니스 의사결정 등)에 적용.

---

## 4. 품질 평가 루브릭 (Dataset & Model Quality Rubric)

### 종합 평가 지표: `QuizScore` (총점 1.0)
$$\text{QuizScore} = 0.30 \times G + 0.25 \times U + 0.20 \times D + 0.15 \times I + 0.10 \times L$$

| 지표 | 가중치 | 평가 기준 및 의미 |
| :--- | :---: | :--- |
| **G (Groundedness)** | **0.30** | 정답과 `evidence`가 본문 내용에 100% 직접적으로 근거하는가? (본문 밖 상식 개입 시 감점) |
| **U (Answer Uniqueness)** | **0.25** | 4개의 선택지 중 정답이 오직 1개로 명확히 특정되는가? (복수정답/무정답 시 탈락) |
| **D (Distractor Quality)** | **0.20** | 오답 보기가 정답과 **동일한 의미적 범주(Semantic Category)**에서 그럴듯하게 경쟁하는가? |
| **I (Importance)** | **0.15** | 본문의 지엽적인 오타나 곁가지가 아닌 핵심 논리와 중요 개념을 묻는가? |
| **L (Language Quality)** | **0.10** | 어색한 번역투나 문법적 오류가 없는 자연스러운 한국어 문장인가? |

### 🚫 즉시 탈락 조건 (Hard Constraints - Drop Rules)
* 본문에 근거가 없음 (Hallucination) $\rightarrow$ **DROP**
* 정답이 2개 이상 가능하거나 정답이 없음 $\rightarrow$ **DROP**
* `evidence` 문장과 `answer`의 의미가 불일치 $\rightarrow$ **DROP**
* 선택지가 4개가 아니거나 동일한 선택지가 중복됨 $\rightarrow$ **DROP**

---

## 5. 3대 사전 베이스라인 (Baseline Benchmarks)

* **B0 (Vanilla)**: 제약 없는 기본 제로샷 ("다음 글을 읽고 객관식 문제를 만들어라")
* **B1 (Prompt Engineered)**: `evidence` 필수, 단일 정답, 동일 범주 오답 등 규칙 명시
* **B2 (Few-Shot)**: 3~5개의 고품질 4지선다 예제를 주입한 최강 인컨텍스트 베이스라인
* **F1/F2 (LoRA Models)**: B2 베이스라인과 비교하여 실제 파인튜닝의 순수 부가가치를 검증
