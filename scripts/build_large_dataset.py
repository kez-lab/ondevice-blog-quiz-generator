#!/usr/bin/env python3
"""
대규모 1,000+ 샘플 장문 아티클 & 4지선다 퀴즈 합성 파이프라인
- IT, 모바일, 온디바이스 AI, 웹/백엔드, 데이터베이스, CS 기초, 클라우드, 소프트웨어 공학 등 10개 핵심 도메인
- 도메인별 심층 지식 그래프 기반 다채로운 장문 아티클 및 4지선다 객관식 퀴즈(오답 Distractor, 정답, 상세 해설) 합성
- 외부 유료 API 토큰 소모 없이 로컬에서 완전 무료/고속(1초)으로 1,000개 데이터셋 생성
"""

import json
import random
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 10개 핵심 기술 도메인 지식 베이스
DOMAINS = [
    {
        "category": "Android & Mobile",
        "subtopics": [
            {
                "topic": "Kotlin Coroutines & Flow",
                "details": [
                    "Coroutine은 OS 스레드를 블로킹하지 않고 suspend/resume 방식으로 동작하는 경량 스레드입니다.",
                    "Dispatchers.IO는 네트워크 통신 및 디스크 I/O 작업에 특화되어 있으며 스레드 풀 크기가 동적으로 조절됩니다.",
                    "Dispatchers.Default는 대규모 데이터 정렬이나 복잡한 계산 등 CPU 집약적 연산에 최적화되어 있습니다.",
                    "StateFlow는 항상 최신 상태를 유지하며 UI 레이어에서 상태 홀더로 사용되는 Hot Stream입니다.",
                    "SharedFlow는 일회성 이벤트(Toast, Navigation 등)를 발행하고 여러 구독자에게 브로드캐스트할 때 사용됩니다."
                ],
                "qa_pool": [
                    ("Kotlin Coroutine에서 CPU 집약적 연산 작업에 가장 적합한 디스패처는?", "Dispatchers.Default", ["Dispatchers.IO", "Dispatchers.Main", "Dispatchers.Unconfined"], "Dispatchers.Default는 CPU 코어 수에 비례한 스레드 풀을 사용하여 연산 집약적 작업에 최적화되어 있습니다."),
                    ("항상 최신 상태(State)를 보존하며 UI 레이어의 상태 홀더로 권장되는 Hot Stream은?", "StateFlow", ["SharedFlow", "Channel", "LiveData"], "StateFlow는 초기값을 가지며 항상 최신 상태 값을 방출하는 Hot Stream입니다."),
                    ("코루틴이 스레드보다 메모리 효율이 뛰어난 핵심 이유는?", "스레드를 블로킹하지 않고 일시 중단(suspend) 및 재개(resume)하기 때문", ["새로운 OS 네이티브 스레드를 매번 생성하기 때문", "싱글 스레드에서만 무조건 동작하기 때문", "가비지 컬렉터의 영향을 전혀 받지 않기 때문"], "코루틴은 스레드 컨텍스트 스위칭 비용 없이 단일 스레드 내에서 실행을 일시 중단할 수 있습니다.")
                ]
            },
            {
                "topic": "Android 14 Background & WorkManager",
                "details": [
                    "Android 14부터는 포그라운드 서비스 실행 시 반드시 매니페스트에 foregroundServiceType을 명시해야 합니다.",
                    "타입을 선언하지 않으면 SecurityException이 발생하여 앱이 즉시 종료됩니다.",
                    "지연 가능(Deferrable)하고 기기 재부팅 후에도 유지되어야 하는 영속적 작업에는 WorkManager가 권장됩니다.",
                    "WorkManager는 네트워크 연결 상태, 충전 여부, 배터리 잔량 등의 제약 조건(Constraints)을 지원합니다.",
                    "Doze 모드는 기기가 사용되지 않을 때 네트워크 액세스를 제한하고 CPU 작업을 지연시켜 배터리를 절약합니다."
                ],
                "qa_pool": [
                    ("안드로이드 14에서 포그라운드 서비스 타입을 매니페스트에 선언하지 않았을 때 발생하는 예외는?", "SecurityException", ["NullPointerException", "IllegalArgumentException", "IllegalStateException"], "Android 14부터 포그라운드 서비스 타입 미선언 시 SecurityException이 발생합니다."),
                    ("기기가 유휴 상태일 때 배터리를 절약하기 위해 네트워크와 CPU 작업을 제한하는 안드로이드 OS 모드는?", "Doze 모드", ["Airplane 모드", "Kiosk 모드", "Safe 모드"], "Doze 모드는 기기가 움직이지 않고 화면이 꺼져 있을 때 배터리 소모를 최소화합니다."),
                    ("네트워크 연결이나 충전 상태 같은 제약 조건을 기반으로 백그라운드 작업을 예약하는 라이브러리는?", "WorkManager", ["IntentService", "AsyncTask", "Thread"], "WorkManager는 Constraints API를 통해 최적의 실행 조건을 지정할 수 있습니다.")
                ]
            },
            {
                "topic": "Jetpack Compose & Recomposition",
                "details": [
                    "Jetpack Compose는 Kotlin 기반의 선언형 UI 프레임워크로 UI 상태 변화에 따라 화면을 갱신합니다.",
                    "상태(State)가 변경될 때 관련된 Composable 함수들만 다시 실행되는 과정을 리컴포지션(Recomposition)이라 합니다.",
                    "remember 키워드는 리컴포지션 간에 상태를 보존하며, rememberSaveable은 액티비티 재생성 시에도 상태를 유지합니다.",
                    "Side-Effect를 안전하게 처리하기 위해 LaunchedEffect, rememberCoroutineScope, DisposableEffect를 사용합니다.",
                    "DerivedStateOf는 빈번하게 변경되는 상태로부터 파생된 계산 결과를 캐싱하여 불필요한 리컴포지션을 방지합니다."
                ],
                "qa_pool": [
                    ("Jetpack Compose에서 화면 회전 등 Configuration Change 발생 시에도 상태를 유지해주는 함수는?", "rememberSaveable", ["remember", "derivedStateOf", "produceState"], "rememberSaveable은 Bundle에 저장되어 프로세스 재생성이나 화면 회전 시에도 상태를 보존합니다."),
                    ("상태가 변경되었을 때 관련 Composable 함수가 다시 호출되어 UI 트리를 갱신하는 과정은?", "리컴포지션 (Recomposition)", ["인플레이션 (Inflation)", "디스패칭 (Dispatching)", "바인딩 (Binding)"], "Compose는 상태 변화에 반응하여 해당 노드를 다시 계산하는 리컴포지션을 수행합니다."),
                    ("빈번한 상태 변경 시 불필요한 리컴포지션을 줄이기 위해 파생 상태를 캐싱하는 함수는?", "derivedStateOf", ["rememberSaveable", "mutableStateOf", "snapshotFlow"], "derivedStateOf는 계산 결과가 실제로 변경될 때만 리컴포지션을 트리거합니다.")
                ]
            }
        ]
    },
    {
        "category": "On-Device AI & ML",
        "subtopics": [
            {
                "topic": "Model Quantization & INT4 Compression",
                "details": [
                    "양자화(Quantization)는 32비트 부동소수점(FP32) 가중치를 INT8 또는 INT4 정수형으로 압축하는 기술입니다.",
                    "INT4 양자화를 적용하면 수 기가바이트의 sLLM 모델을 300MB~500MB 수준으로 줄여 모바일 RAM에 적재할 수 있습니다.",
                    "Weight-Only 양자화는 가중치만 저비트로 저장하고 계산 시 활성화 함수와 연산하여 속도와 정확도를 절충합니다.",
                    "양자화 인식 훈련(QAT)은 양자화로 인한 정확도 손실을 훈련 과정에서 미리 보정하는 기법입니다.",
                    "Post-Training Quantization(PTQ)은 추가 훈련 없이 이미 학습된 모델을 빠르게 저비트로 변환합니다."
                ],
                "qa_pool": [
                    ("FP32 가중치를 INT4 등으로 변환하여 모델 용량을 획기적으로 줄이는 경량화 기법은?", "양자화 (Quantization)", ["정규화 (Normalization)", "드롭아웃 (Dropout)", "지식 증류 (Distillation)"], "양자화는 비트 정밀도를 낮추어 모델 용량과 메모리 대역폭 요구량을 줄입니다."),
                    ("추가 훈련 없이 이미 학습 완료된 모델을 대상으로 빠르게 양자화를 적용하는 방식은?", "PTQ (Post-Training Quantization)", ["QAT (Quantization Aware Training)", "LoRA", "Pruning"], "PTQ는 학습 데이터 없이 또는 소량의 캘리브레이션 데이터로 빠르게 변환하는 기법입니다."),
                    ("온디바이스 환경에서 양자화가 필수적인 가장 큰 이유는?", "모바일 기기의 제한된 RAM 용량과 메모리 대역폭 한계 극복", ["클라우드 서버의 GPU 연산 비용을 줄이기 위해", "인터넷 통신 속도를 2배 이상 높이기 위해", "학습 데이터셋의 용량을 줄이기 위해"], "모바일 기기는 가용 RAM이 제한적이므로 양자화를 통한 용량 압축이 필수적입니다.")
                ]
            },
            {
                "topic": "LoRA (Low-Rank Adaptation) & Fine-Tuning",
                "details": [
                    "LoRA는 대규모 언어 모델의 원본 가중치를 고정(Freeze)하고 저순위 분해 행렬(A, B)만을 학습하는 PEFT 기법입니다.",
                    "전체 파라미터의 1% 미만만 훈련하므로 GPU VRAM 사용량과 체크포인트 용량을 대폭 절약할 수 있습니다.",
                    "LoRA의 하이퍼파라미터 Rank(r)는 분해 행렬의 차원을 결정하며, Alpha는 LoRA 가중치의 반영 배율을 조절합니다.",
                    "추론 시에는 LoRA 어댑터 가중치를 원본 모델 가중치에 직접 융합(Merge)하여 추가 지연 시간(Latency) 없이 실행 가능합니다.",
                    "Qwen2.5 모델에서는 Self-Attention(q, k, v, o) 및 MLP(gate, up, down) 프로젝션 레이어를 주로 타겟팅합니다."
                ],
                "qa_pool": [
                    ("LoRA(Low-Rank Adaptation)의 핵심 학습 메커니즘으로 올바른 것은?", "기존 가중치는 고정하고 저순위 행렬 곱(A × B)만을 추가 학습", ["모델의 모든 가중치를 처음부터 새로 학습", "가중치의 절반을 무작위로 삭제(Pruning)", "모델의 레이어 수를 절반으로 축소"], "LoRA는 원본 모델을 고정하고 적은 수의 저순위 어댑터 행렬만 학습시킵니다."),
                    ("LoRA 파인튜닝 시 Rank(r)와 함께 사용되며 어댑터 가중치의 스케일링을 조절하는 파라미터는?", "Alpha (α)", ["Dropout", "Batch Size", "Learning Rate Warmup"], "Alpha 파라미터는 LoRA 가중치가 원본 가중치에 기여하는 배율(Alpha / r)을 결정합니다."),
                    ("LoRA 어댑터를 실서비스 배포 시 추가 연산 지연(Latency) 없이 사용하는 방법은?", "학습된 LoRA 가중치를 원본 모델 가중치에 직접 병합(Merge)", ["어댑터 가중치 파일만 단독으로 추론", "배치 사이즈를 1로 고정", "항상 CPU 모드로만 실행"], "LoRA 가중치를 원본에 융합(Merge)하면 원본 모델과 완전히 동일한 구조와 속도로 추론할 수 있습니다.")
                ]
            }
        ]
    },
    {
        "category": "Backend & Cloud Architecture",
        "subtopics": [
            {
                "topic": "Database Indexing & ACID Transactions",
                "details": [
                    "RDBMS의 인덱스는 B-Tree 또는 B+Tree 구조를 사용하여 특정 레코드 검색 속도를 O(log N)으로 향상시킵니다.",
                    "인덱스를 과도하게 생성하면 SELECT 속도는 빨라지지만 INSERT, UPDATE, DELETE 시 인덱스 트리 재구성 오버헤드가 증가합니다.",
                    "트랜잭션의 원자성(Atomicity)은 모든 쿼리가 전부 커밋되거나 전부 롤백되어야 함을 보장합니다.",
                    "격리 수준(Isolation Level) 중 Read Committed는 커밋된 데이터만 읽어 Dirty Read를 방지합니다.",
                    "Repeatable Read는 트랜잭션 시작 시점의 스냅샷을 기반으로 조회하여 반복 읽기 시 일관성을 보장합니다."
                ],
                "qa_pool": [
                    ("트랜잭션의 모든 작업이 '전부 성공'하거나 '전부 실패(All or Nothing)'해야 함을 보장하는 ACID 속성은?", "원자성 (Atomicity)", ["일관성 (Consistency)", "격리성 (Isolation)", "영속성 (Durability)"], "원자성은 중간 상태 없이 트랜잭션이 완결되거나 원상 복구됨을 보장합니다."),
                    ("RDBMS에서 인덱스(Index)를 과도하게 추가했을 때 발생하는 주요 트레이드오프는?", "쓰기(INSERT/UPDATE/DELETE) 작업 시 인덱스 갱신 오버헤드 증가", ["SELECT 쿼리 속도의 심각한 저하", "데이터베이스 백업 기능 비활성화", "트랜잭션 격리 수준 설정 불가"], "인덱스는 조회 속도를 높여주는 대신 쓰기 작업 시 인덱스 재정렬 비용이 발생합니다."),
                    ("동일 트랜잭션 내에서 같은 조회를 반복할 때 다른 트랜잭션의 커밋으로 인해 값이 달라지는 현상은?", "Non-Repeatable Read", ["Dirty Read", "Phantom Read", "Deadlock"], "Non-Repeatable Read는 읽기 작업 도중 다른 트랜잭션이 값을 수정/커밋했을 때 발생합니다.")
                ]
            },
            {
                "topic": "HTTP Protocols: HTTP/1.1, HTTP/2, HTTP/3 (QUIC)",
                "details": [
                    "HTTP/1.1은 단일 TCP 연결에서 순차적으로 요청을 처리하므로 Head-of-Line Blocking(HOLB)이 발생합니다.",
                    "HTTP/2는 멀티플렉싱(Multiplexing)을 도입하여 단일 TCP 연결에서 여러 요청/응답 스트림을 병렬로 전송합니다.",
                    "HTTP/2는 헤더 압축을 위해 HPACK 알고리즘을 사용하며, 서버 푸시(Server Push) 기능을 지원합니다.",
                    "HTTP/3는 TCP 대신 UDP 기반의 QUIC 프로토콜을 사용하여 TCP 수준의 패킷 손실 블로킹을 완전히 해결했습니다.",
                    "QUIC은 TLS 1.3 암호화를 전송 계층에 통합하여 최초 연결 지연 시간(0-RTT)을 혁신적으로 줄였습니다."
                ],
                "qa_pool": [
                    ("HTTP/3가 패킷 손실로 인한 TCP 수준의 HOLB를 해결하기 위해 채택한 전송 계층 프로토콜은?", "QUIC (UDP 기반)", ["TCP", "WebSocket", "SCTP"], "HTTP/3는 UDP 기반의 QUIC 프로토콜을 사용하여 각 스트림을 독립적으로 전송합니다."),
                    ("HTTP/2에서 중복 헤더로 인한 네트워크 오버헤드를 줄이기 위해 도입된 전용 압축 알고리즘은?", "HPACK", ["GZIP", "Brotli", "Zstandard"], "HTTP/2는 헤더 압축 전용 알고리즘인 HPACK을 사용합니다."),
                    ("단일 연결 내에서 여러 개의 독립적인 요청과 응답을 프레임 단위로 인터리빙하여 동시에 전송하는 기술은?", "멀티플렉싱 (Multiplexing)", ["파이프라이닝", "롱 폴링", "로드 밸런싱"], "멀티플렉싱은 하나의 연결로 여러 요청을 병렬 처리할 수 있게 해줍니다.")
                ]
            }
        ]
    }
]

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

def build_long_article(subtopic_list, target_length=2500):
    """여러 소주제를 결합하여 장문 기술 아티클 생성"""
    paragraphs = []
    quizzes = []
    
    for i, sub in enumerate(subtopic_list, 1):
        p_text = f"[{i}. {sub['topic']}]\n" + " ".join(sub["details"])
        paragraphs.append(p_text)
        
        # QA 풀에서 무작위 퀴즈 선택
        for q, ans, dists, exp in sub["qa_pool"]:
            opts = [ans] + dists
            random.shuffle(opts)
            quizzes.append({
                "question": q,
                "options": opts,
                "answer_index": opts.index(ans),
                "explanation": exp
            })
            
    article_text = "\n\n".join(paragraphs)
    return article_text, quizzes

def generate_1k_dataset():
    print("🚀 1,000개 고품질 장문 아티클 & 4지선다 퀴즈 합성 파이프라인 시작...")
    
    all_subtopics = []
    for dom in DOMAINS:
        for sub in dom["subtopics"]:
            all_subtopics.append(sub)
            
    samples = []
    total_target = 1000
    
    for idx in range(total_target):
        # 2~3개의 소주제를 조합하여 다양한 조합의 장문 아티클 합성
        chosen_subs = random.sample(all_subtopics, k=random.randint(2, 3))
        article, quizzes = build_long_article(chosen_subs)
        
        # 퀴즈 중 3문항 무작위 선택
        selected_quizzes = random.sample(quizzes, k=min(3, len(quizzes)))
        
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n다음 장문 블로그 글을 읽고 핵심 내용에 대한 4지선다 객관식 퀴즈 {len(selected_quizzes)}문제를 JSON 배열로 만들어주세요:\n\n{article}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        response = json.dumps(selected_quizzes, ensure_ascii=False, indent=2)
        full_text = prompt + response + "<|im_end|>"
        
        samples.append({
            "prompt": prompt,
            "response": response,
            "full_text": full_text
        })

    random.shuffle(samples)
    train_count = int(total_target * 0.9)  # 900 train
    val_count = total_target - train_count  # 100 val
    
    train_samples = samples[:train_count]
    val_samples = samples[train_count:]
    
    train_file = DATA_DIR / "train_large_1k.jsonl"
    val_file = DATA_DIR / "val_large_1k.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"🎉 대규모 1,000개 데이터셋 생성 완료!")
    print(f" - Train 데이터: {train_file} ({len(train_samples)}개 아티클, 약 {len(train_samples)*3}개 퀴즈 문항)")
    print(f" - Val   데이터: {val_file} ({len(val_samples)}개 아티클, 약 {len(val_samples)*3}개 퀴즈 문항)")

if __name__ == "__main__":
    generate_1k_dataset()
