# 📖 On-Device AI & sLLM Fine-Tuning Handbook

AI 이해도 0단계 입문자부터 실무 모바일 AI 엔지니어까지 아우르는 **온디바이스 AI, LLM 작동 원리, sLLM 파인튜닝, 허깅페이스, 안드로이드 SDK 종합 핸드북**입니다.

---

## 📑 종합 목차 (12 Steps Learning Journey)

```
docs/
├── README.md                                  # 📑 전체 종합 목차
│
├── 🟢 Phase 1. AI 입문 & 핵심 멘탈 모델 (Foundations)
│   ├── [01] 완전 초보자를 위한 AI & 온디바이스 기초 개념
│   ├── [02] AI는 어떻게 글을 이해하고 말할까? (토큰, 임베딩, 어텐션)
│   ├── [03] Temperature, Top-P, Top-K 완벽 정복 (AI 뇌 조절 레버)
│   └── [04] AI 입문자를 위한 AI 엔지니어 성장 로드맵 & 멘탈 모델
│
├── 🟡 Phase 2. MLOps & 파인튜닝 실무 (Fine-Tuning & MLOps)
│   ├── [05] 파인튜닝(Fine-Tuning)과 LoRA 완벽 이해하기
│   ├── [06] 데이터 엔지니어링과 합성 데이터(Synthetic Data)의 원리
│   ├── [07] Apple Silicon Mac(M1~M4)에서 AI 학습/파인튜닝 실무 가이드
│   ├── [08] 실전 파인튜닝(Fine-Tuning) A to Z 단계별 플레이북
│   └── [09] 허깅페이스(Hugging Face) 완벽 가이드 & hf CLI
│
└── 🔵 Phase 3. 온디바이스 & 모바일 엔지니어링 (On-Device & Mobile)
    ├── [10] 온디바이스 AI 모델의 종류와 선정 실무 기준
    ├── [11] 안드로이드 온디바이스 AI SDK 및 오픈소스 배포 아키텍처
    └── [12] 자주 묻는 질문(FAQ) & 실전 문제 해결 가이드
```

---

## 📑 챕터별 세부 요약 & 바로가기

### 🟢 Phase 1. AI 입문 & 핵심 멘탈 모델 (Foundations)

1. [**[01] 완전 초보자를 위한 AI & 온디바이스 기초 개념**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/01_ai_fundamentals_for_beginners.md)
   * 파라미터(0.5B, 2B, 7B)와 가중치의 쉬운 비유
   * 온디바이스 AI vs 클라우드 AI 비교 (비용 0원, 100% 프라이버시)
   * 양자화(INT4)로 8배 압축하는 원리
2. [**[02] AI는 어떻게 글을 이해하고 말할까? (토큰, 임베딩, 어텐션)**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/02_how_llms_actually_work.md)
   * 글자를 숫자로 쪼개는 토크나이저(Tokenizer)
   * 고차원 의미 지도인 임베딩(Vector Embedding)
   * 문맥을 파악하는 어텐션(Self-Attention)과 Next Token Prediction
3. [**[03] Temperature, Top-P, Top-K 완벽 정복 (AI 뇌 조절 레버)**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/03_generation_parameters_mastery.md)
   * 온도(Temperature)의 창의성 vs 논리성 조절
   * Top-P와 Top-K 필터링 및 Repetition Penalty
   * 태스크별(퀴즈, 코딩, 챗봇) 최적 파라미터 치트시트
4. [**[04] AI 입문자를 위한 AI 엔지니어 성장 로드맵 & 멘탈 모델**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/04_ai_engineer_growth_roadmap.md)
   * 규칙 기반 코딩(If/Else) vs 머신러닝(확률 모델)의 차이점
   * AI 엔지니어 4대 기술 트리 (Prompting ➡️ RAG ➡️ Fine-Tuning ➡️ On-Device)
   * 세계 최고 무료 강의 추천 (Andrew Ng, Hugging Face)

---

### 🟡 Phase 2. MLOps & 파인튜닝 실무 (Fine-Tuning & MLOps)

5. [**[05] 파인튜닝(Fine-Tuning)과 LoRA 완벽 이해하기**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/05_fine_tuning_and_lora_explained.md)
   * 사전학습 vs 파인튜닝의 차이
   * LoRA 저순위 분해($A \times B$)와 Rank($r$), Alpha($\alpha$) 튜닝법
   * Loss(손실도), Epoch, Batch Size, Response-Only Loss 마스킹
6. [**[06] 데이터 엔지니어링과 합성 데이터의 원리**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/06_data_engineering_and_synthetic_data.md)
   * Instruction-Response (`.jsonl`) 데이터셋 구조
   * 3가지 합성 데이터(Synthetic Data) 생성 기법
   * 헷갈리는 오답(Hard Negative) 설계 원칙
7. [**[07] Apple Silicon Mac(M1~M4)에서 AI 학습 실무 가이드**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/07_mac_apple_silicon_ai_training_guide.md)
   * 통합 메모리(48GB)와 Metal MPS 가속 원리
   * PyTorch(MPS) vs Apple MLX vs llama.cpp 비교
   * Mac PyTorch 실무 최적화 팁
8. [**[08] 실전 파인튜닝 A to Z 8단계 플레이북**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/08_step_by_step_finetuning_playbook.md)
   * 기획 ➡️ 모델 리서치 ➡️ 데이터 포맷팅 ➡️ LoRA ➡️ Loss 마스킹 ➡️ 평가 ➡️ INT4 양자화 ➡️ 배포
9. [**[09] 허깅페이스(Hugging Face) 완벽 가이드 & hf CLI**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/09_huggingface_complete_guide.md)
   * Models, Datasets, Spaces 구조
   * 신뢰받는 Model Card(`README.md`) 작성법
   * 공식 `hf` CLI 명령어 치트시트

---

### 🔵 Phase 3. 온디바이스 & 모바일 엔지니어링 (On-Device & Mobile)

10. [**[10] 온디바이스 AI 모델 종류와 선정 실무 기준**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/10_ondevice_model_landscape_and_selection.md)
    * sLLM, Vision, VLM 라인업 총정리
    * 스마트폰 RAM 스펙별 모델 선정 공식 (4GB, 8GB, 12GB+)
    * 4대 배포 포맷 (`.tflite/.task`, `.gguf`, `.onnx`, `.pte`)
11. [**[11] 안드로이드 온디바이스 AI SDK 및 오픈소스 배포**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/11_android_ondevice_ai_sdk_architecture.md)
    * Google MediaPipe Tasks GenAI & LiteRT 런타임
    * Kotlin Flow 기반 실시간 토큰 스트리밍
    * JitPack을 통한 GitHub 오픈소스 라이브러리 배포
12. [**[12] 자주 묻는 질문(FAQ) & 실전 문제 해결 가이드**](file:///Users/kwak-euijin/Documents/antigravity/mysterious-noether/docs/12_faq_and_troubleshooting.md)
    * GPU 발열 및 팬 소음 대처법
    * Loss(손실도) 수렴 실패 시 해결책
    * AI가 뱉은 깨진 JSON을 살려내는 3중 방어 파서 설계법
    * M4 Pro 맥북(48GB RAM) 200% 활용 팁
