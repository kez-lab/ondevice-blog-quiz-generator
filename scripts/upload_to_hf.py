#!/usr/bin/env python3
"""
Hugging Face Hub 모델 자동 업로드 및 Model Card 생성 스크립트
- 모델 가중치, 토크나이저, Model Card(README.md) 업로드
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

MODEL_CARD_TEMPLATE = """---
language:
- ko
- en
license: apache-2.0
tags:
- on-device
- android
- text-generation
- litert
- mediapipe
- quiz-generation
- qwen2.5
- education
base_model: Qwen/Qwen2.5-0.5B-Instruct
pipeline_tag: text-generation
---

# 🧠 Qwen2.5-0.5B Blog-to-Quiz On-Device AI for Android

한국어 블로그/기술 아티클 본문을 입력받아 **온디바이스(기기 내부)**에서 **4지선다 객관식 퀴즈(문제, 4개 보기, 정답 번호, 상세 해설)**를 JSON 형태로 실시간 생성하는 초경량 AI 모델입니다.

---

## 📱 Features
- **Zero Server Cost**: 스마트폰 내부(NPU/GPU/CPU)에서 100% 동작하여 서버 비용 0원
- **Privacy First**: 읽고 있는 글과 개인 메모가 외부 서버로 전송되지 않음
- **Ultra Lightweight**: INT4 양자화 기준 **~350MB**로 보급형 안드로이드 스마트폰에서도 쾌적하게 구동
- **Structured JSON Output**: 안드로이드 Kotlin 데이터 클래스로 즉시 파싱 가능

---

## 🛠️ Output Format (JSON)
```json
[
  {
    "question": "Kotlin Coroutine에서 I/O 작업에 최적화된 디스패처는?",
    "options": [
      "Dispatchers.Main",
      "Dispatchers.IO",
      "Dispatchers.Default",
      "Dispatchers.Unconfined"
    ],
    "answer_index": 1,
    "explanation": "Dispatchers.IO는 네트워크 통신 및 파일 입출력 작업에 최적화된 디스패처입니다."
  }
]
```

---

## 🚀 Android Integration (Kotlin SDK)

```kotlin
val quizGen = LocalQuizGenerator.builder(context)
    .fromHuggingFace("{repo_id}")
    .build()

val quizzes = quizGen.generateQuiz(blogText, count = 2)
```

---

## 📄 License
Apache License 2.0
"""

def upload_model(repo_id: str, local_dir: str, token: str = None):
    api = HfApi(token=token)
    
    print(f"🚀 Hugging Face 레포지토리 생성 확인: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    
    # Model Card 생성
    model_card_content = MODEL_CARD_TEMPLATE.replace("{repo_id}", repo_id)
    readme_path = Path(local_dir) / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)
    print("📝 Model Card (README.md) 생성 완료")

    print(f"⬆️ '{local_dir}' 디렉토리의 파일들을 Hugging Face Hub로 업로드 중...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    print(f"🎉 업로드 완료! 확인 링크: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face Hub 모델 업로더")
    parser.add_argument("--repo-id", required=True, help="예: your-username/qwen-blog-quiz-android")
    parser.add_argument("--local-dir", default=str(Path(__file__).parent / "output" / "qwen2.5-0.5b-blog-quiz-lora"), help="업로드할 로컬 폴더 경로")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="Hugging Face Access Token")
    
    args = parser.parse_args()
    upload_model(args.repo_id, args.local_dir, args.token)
