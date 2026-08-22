---
language:
- ko
- en
license: gemma
library_name: transformers
tags:
- gemma
- gemma-2
- on-device
- android
- litert
- mediapipe
- quiz-generation
- source-grounded
- chain-of-thought
- multiple-choice
- text-generation
base_model: google/gemma-2-2b-it
pipeline_tag: text-generation
---

# Gemma-2-2B: Source-Grounded Korean MCQ Generator

[Base Model: Google Gemma-2-2B](https://ai.google.dev/edge) • [Benchmark Score: 0.985 / 1.000](https://github.com/kez-lab/ondevice-blog-quiz-generator) • [100% Grounded](https://github.com/kez-lab/ondevice-blog-quiz-generator) • [Android LiteRT Ready](https://github.com/kez-lab/ondevice-blog-quiz-generator)

`gemma-2-2b-quiz-korean` is a specialized lightweight language model fine-tuned from **Google Gemma-2-2B**. It is engineered specifically for **source-grounded Multiple Choice Question (MCQ) generation** directly from Korean blog posts, technical documentation, and long-form articles.

The model strictly restricts question synthesis to facts explicitly verified within the provided source text, eliminating hallucinations while generating pedagogically meaningful 4-choice questions with hard-negative distractors and exact evidence citations.

---

## Key Capabilities

1. **Internal Chain-of-Thought (`<thought>`) Reasoning**:
   Unlike standard single-pass generators, the model utilizes an internal autoregressive reasoning phase before producing the final JSON payload. It systematically plans the evidence sentence, target concept, unambiguous correct answer, and three same-category distractors.
2. **Zero-Hallucination Evidence Grounding**:
   Every generated question includes a verbatim `evidence` quote from the source text, ensuring strict factual alignment and auditability.
3. **Equi-Distributed Answer Permutation**:
   The model eliminates position bias (such as index-0 bias) through uniform answer position training (25% distribution across A, B, C, D).
4. **Clean 256k Multilingual Tokenization**:
   Leveraging Gemma-2's vocabulary, the model generates natural Korean syntax without CJK token leakage or Hanja bleeding.
5. **On-Device Edge Deployment**:
   Optimized for local inference via Google MediaPipe Tasks GenAI and LiteRT (TensorFlow Lite), running on modern Android devices with zero cloud dependency.

---

## Benchmark Evaluation

Evaluated against a held-out test suite of **100 document-level isolated articles** across 10 domains (Android, Distributed Systems, AI/ML, Economics, Medicine, Natural Sciences, History, Philosophy, UX Design, and Architecture).

$$\text{QuizScore} = 0.30 \times \text{Groundedness} + 0.25 \times \text{Uniqueness} + 0.20 \times \text{DistractorQuality} + 0.15 \times \text{Importance} + 0.10 \times \text{LanguageQuality}$$

| Evaluation Metric | Base Gemma-2-2B (Prompted) | **Gemma-2-2B Quiz Korean (Ours)** | Improvement |
| :--- | :---: | :---: | :---: |
| **Overall QuizScore** | 0.9380 | **`0.9850` / 1.000** | **+5.0%** |
| **Groundedness (Factuality)** | 0.940 | **`1.000` (100%)** | Zero Hallucination |
| **Answer Uniqueness** | 0.950 | **`1.000` (100%)** | Single Deterministic Answer |
| **Distractor Plausibility** | 0.920 | **`1.000`** | Semantic Category Match |
| **Language Quality** | 0.900 | **`1.000`** | Pure Korean Syntax |
| **System Prompt Overhead** | ~500 tokens | **`0 tokens`** | Fully Embedded Behavior |

---

## Output Schema

The model generates an internal `<thought>` trace followed by a structured JSON payload:

```json
{
  "questions": [
    {
      "question": "Specific question grounded in the source text",
      "options": [
        "Option A (same semantic category)",
        "Option B (same semantic category)",
        "Option C (same semantic category)",
        "Option D (same semantic category)"
      ],
      "answer_index": 2,
      "explanation": "Clear educational explanation of why the answer is correct",
      "evidence": "Verbatim excerpt from the article proving the answer"
    }
  ]
}
```

---

## Quickstart (Python / Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "kez-lab/gemma-2-2b-quiz-korean"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

article = """
Jetpack Compose에서 derivedStateOf를 사용하면 원본 상태의 잦은 변화 중에서
우리가 관심 있는 특정 조건(firstVisibleItemIndex > 0)이 변경되는 순간에만
다운스트림 Recomposition을 트리거하도록 캐싱 및 완충 역할을 합니다.
"""

messages = [
    {
        "role": "user",
        "content": f"주어진 글만을 근거로 핵심 개념을 분석하고 4지선다 객관식 문제를 생성하라.\n\n[ARTICLE]\n{article}"
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.01,
        repetition_penalty=1.1
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

---

## Android On-Device SDK Integration

This model can be compiled into `.task` / `.tflite` format for on-device execution using Google MediaPipe Tasks GenAI:

```kotlin
// Android SDK Usage
val generator = LocalQuizGenerator.builder(context)
    .fromHuggingFace("kez-lab/gemma-2-2b-quiz-korean")
    .setMaxTokens(1024)
    .setTemperature(0.01f)
    .build()

viewModelScope.launch {
    val result = generator.generateQuiz(articleText)
    result.onSuccess { quizzes ->
        // Render verified quiz cards with evidence quotes
    }
}
```

---

## Training Details

- **Base Architecture**: Google Gemma-2-2B (`unsloth/gemma-2-2b-it`, 2.6B parameters)
- **Methodology**: Parameter-Efficient Fine-Tuning (PEFT / LoRA) with Standalone Weight Merge
- **LoRA Configuration**: $r=16, \alpha=32$, targeting all linear projections (`q, k, v, o, gate, up, down`)
- **Precision**: `bfloat16`
- **Loss Masking**: Response-only loss computation on assistant reasoning and JSON tokens
- **Dataset**: Multi-domain curriculum of 1,000 verified CoT samples validated through strict Critic filtering

---

## License & Attribution

- **Base Model License**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **Developed by**: [KEZ Lab](https://github.com/kez-lab)
