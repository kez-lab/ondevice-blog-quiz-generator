#!/usr/bin/env python3
"""
장문 블로그 글 (최대 1만자) & 다중 퀴즈 (3~5문항) 생성 특화 데이터셋 빌더
"""

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1만 자에 준하는 심층 기술 블로그 및 대용량 아티클 데이터셋
LONG_ARTICLE_SAMPLES = [
    {
        "title": "안드로이드 최신 아키텍처 및 백그라운드 처리 완전 정복 가이드",
        "article": (
            "[1. 안드로이드 클린 아키텍처와 계층 분리]\n"
            "안드로이드 권장 앱 아키텍처는 크게 UI 레이어, 도메인 레이어(선택), 데이터 레이어로 나뉩니다. "
            "UI 레이어는 화면에 데이터를 렌더링하는 UI 요소(Composable 또는 View)와 상태를 보존하고 가공하는 ViewModel로 구성됩니다. "
            "데이터 레이어는 비즈니스 로직의 원천 데이터를 제공하는 Repository와 로컬 DB(Room), 원격 API(Retrofit)를 다루는 DataSource로 구성됩니다. "
            "단방향 데이터 흐름(UDF, Unidirectional Data Flow)을 통해 상태는 아래로 흐르고(State flows down), 이벤트는 위로 전달(Events flow up)됩니다.\n\n"
            "[2. 백그라운드 처리: WorkManager vs Foreground Service]\n"
            "안드로이드에서 백그라운드 작업을 처리할 때 작업의 성격에 따라 올바른 도구를 선택해야 합니다. "
            "즉각적이고 사용자에게 반드시 알림(Notification)으로 진행 상황을 보여주어야 하는 작업(예: 음악 재생, 내비게이션 길 안내, 활성 통화)은 Foreground Service를 사용합니다. "
            "특히 Android 14(API 34)부터는 매니페스트에 반드시 foregroundServiceType(예: location, mediaPlayback 등)을 명시해야 하며, 누락 시 SecurityException이 발생합니다. "
            "반면 앱이 종료되거나 기기가 재부팅되어도 반드시 완료되어야 하는 지연 가능(Deferrable)하고 영속적인(Persistent) 작업(예: 이미지 서버 백업, 데이터 주기적 동기화)은 WorkManager를 사용해야 합니다. "
            "WorkManager는 배터리 절약 모드(Doze Mode)와 네트워크 제약 조건을 고려하여 가장 최적의 타이밍에 작업을 실행합니다.\n\n"
            "[3. 메모리 관리와 GC (Garbage Collection)]\n"
            "안드로이드 ART(Android Runtime)는 메모리 누수를 방지하기 위해 가비지 컬렉션을 수행합니다. "
            "하지만 정적 변수(Static field)에 Context나 Activity 참조를 저장하거나, 수명 주기가 끝난 리스너를 해제하지 않으면 가비지 컬렉터가 메모리를 회수하지 못해 메모리 누수(Memory Leak)가 발생합니다. "
            "앱의 메모리 사용량이 시스템 가용 한도를 초과하면 OutOfMemoryError(OOM)가 발생하고, 안드로이드의 Low Memory Killer(LMK) 데몬이 앱 프로세스를 강제 종료합니다."
        ),
        "quizzes": [
            {
                "question": "안드로이드 권장 단방향 데이터 흐름(UDF) 아키텍처에 대한 설명으로 가장 올바른 것은?",
                "options": [
                    "상태(State)는 위로 흐르고 이벤트(Event)는 아래로 전달된다.",
                    "상태(State)는 아래로 흐르고 이벤트(Event)는 위로 전달된다.",
                    "UI와 데이터 소스가 양방향으로 직접 상태를 수정한다.",
                    "ViewModel은 데이터를 직접 데이터베이스에 영구 저장한다."
                ],
                "answer_index": 1,
                "explanation": "단방향 데이터 흐름(UDF)에서는 상태(State)가 하위 UI로 흐르고, 사용자 인터랙션 이벤트(Event)는 상위 ViewModel로 전달됩니다."
            },
            {
                "question": "앱이 종료되거나 재부팅되어도 반드시 완료되어야 하는 영속적이고 지연 가능한 백그라운드 작업에 가장 권장되는 라이브러리는?",
                "options": [
                    "AsyncTask",
                    "Thread sleep",
                    "WorkManager",
                    "IntentService"
                ],
                "answer_index": 2,
                "explanation": "영속적이고 제약 조건(네트워크, 충전 상태 등)을 고려한 백그라운드 작업에는 WorkManager가 표준 권장 도구입니다."
            },
            {
                "question": "안드로이드 14(API 34)에서 포그라운드 서비스를 시작할 때 매니페스트에 타입을 명시하지 않으면 발생하는 예외는?",
                "options": [
                    "NullPointerException",
                    "SecurityException",
                    "IllegalArgumentException",
                    "ActivityNotFoundException"
                ],
                "answer_index": 1,
                "explanation": "안드로이드 14부터 포그라운드 서비스의 foregroundServiceType을 명시하지 않으면 SecurityException이 발생합니다."
            },
            {
                "question": "Activity가 파괴된 후에도 메모리에서 해제되지 않아 OOM을 유발하는 가장 대표적인 원인은?",
                "options": [
                    "Activity 참조를 정적 변수(Static)에 유지하는 경우",
                    "ViewModel에서 StateFlow를 사용하는 경우",
                    "rememberSaveable로 상태를 보존하는 경우",
                    "WorkManager로 백그라운드 작업을 예약한 경우"
                ],
                "answer_index": 0,
                "explanation": "Activity나 Context 참조를 정적(Static) 변수나 장기 실행 싱글톤에 저장하면 GC가 회수하지 못해 메모리 누수가 발생합니다."
            }
        ]
    },
    {
        "title": "온디바이스 AI sLLM과 경량화/양자화 기술 총정리",
        "article": (
            "[1. 온디바이스 AI의 부상과 필요성]\n"
            "인공지능의 활용이 일상화되면서 클라우드 기반 AI의 한계점들이 부각되고 있습니다. "
            "첫째, 서버 호스팅 비용과 API 토큰 비용이 사용자 수에 비례하여 기하급수적으로 증가합니다. "
            "둘째, 개인의 사적인 메시지, 사진, 금융 정보 등이 외부 서버로 전송되어야 하므로 프라이버시 침해 우려가 큽니다. "
            "셋째, 네트워크 상태가 불안정하거나 비행기/지하철 등 오프라인 환경에서는 사용할 수 없습니다. "
            "온디바이스 AI는 이러한 문제를 해결하기 위해 스마트폰 내부의 NPU, GPU, CPU를 직접 활용하여 로컬에서 모델을 실행합니다.\n\n"
            "[2. 모델 경량화: 양자화(Quantization)와 LoRA]\n"
            "일반적인 거대 언어 모델(LLM)은 수십 기가바이트의 메모리를 요구하여 스마트폰에 탑재할 수 없습니다. "
            "따라서 양자화(Quantization) 기술이 적용됩니다. 양자화는 모델의 가중치를 32비트 부동소수점(FP32)에서 4비트(INT4) 또는 8비트(INT8) 정수로 압축합니다. "
            "이를 통해 모델 용량을 1/4~1/8 수준으로 줄여 수억 파라미터(0.5B~2B) 모델을 300MB~1GB 내외로 스마트폰 RAM에 적재할 수 있습니다. "
            "또한 LoRA(Low-Rank Adaptation)는 기본 가중치를 고정(Freeze)하고 소수의 저순위 분해 행렬만을 튜닝하여 적은 연산량으로 특화 태스크를 학습시킵니다.\n\n"
            "[3. 모바일 구동 프레임워크: Google LiteRT와 MediaPipe]\n"
            "안드로이드에서 sLLM을 안정적으로 구동하기 위해 Google은 LiteRT(구 TensorFlow Lite) 및 MediaPipe GenAI 태스크 라이브러리를 제공합니다. "
            "MediaPipe LLM Inference API는 토크나이저와 KV 캐시 메모리 관리를 네이티브 레벨에서 자동으로 처리하며, "
            "GPU 및 NPU 하드웨어 가속기를 통해 초당 수십 토큰 이상의 고속 추론을 지원합니다."
        ),
        "quizzes": [
            {
                "question": "온디바이스 AI가 클라우드 AI 대비 가지는 핵심 장점이 아닌 것은?",
                "options": [
                    "클라우드 API 서버 비용이 발생하지 않는다.",
                    "네트워크 연결이 없는 오프라인에서도 즉시 작동한다.",
                    "개인 데이터가 기기 외부로 전송되지 않아 프라이버시가 보호된다.",
                    "모델 크기가 무제한으로 커져도 기기 성능에 영향이 없다."
                ],
                "answer_index": 3,
                "explanation": "온디바이스 AI는 스마트폰의 RAM과 배터리 제한을 받으므로 모델 크기 경량화가 필수적입니다."
            },
            {
                "question": "FP32 부동소수점 가중치를 INT4 또는 INT8 정수형으로 변환하여 모델 크기를 줄이는 기술은?",
                "options": [
                    "양자화 (Quantization)",
                    "지식 증류 (Distillation)",
                    "배치 정규화 (Batch Normalization)",
                    "데이터 증강 (Data Augmentation)"
                ],
                "answer_index": 0,
                "explanation": "양자화(Quantization)는 가중치의 정밀도를 낮추어 모델 용량을 압축하고 연산 속도를 높이는 기법입니다."
            },
            {
                "question": "구글이 제공하는 안드로이드 온디바이스 LLM 추론 전용 공식 프레임워크는?",
                "options": [
                    "Google Play Console",
                    "MediaPipe GenAI (LiteRT)",
                    "Firebase Realtime DB",
                    "Android Jetpack Compose"
                ],
                "answer_index": 1,
                "explanation": "Google은 MediaPipe GenAI 및 LiteRT 런타임을 통해 모바일 온디바이스 LLM 추론을 지원합니다."
            }
        ]
    },
    {
        "title": "웹 프로토콜 발전사: HTTP/1.1에서 HTTP/3까지",
        "article": (
            "[1. HTTP/1.1의 한계: HOLB 현상]\n"
            "HTTP/1.1은 웹의 대중화를 이끌었지만 단일 TCP 연결에서 요청이 순차적으로 처리되는 구조였습니다. "
            "이로 인해 앞선 요청의 처리가 지연되면 뒤따르는 모든 요청이 대기해야 하는 'Head-of-Line Blocking(HOLB)' 문제가 발생했습니다. "
            "브라우저는 이를 완화하기 위해 도메인당 최대 6개의 TCP 연결을 병렬로 생성하는 꼼수를 사용했습니다.\n\n"
            "[2. HTTP/2: 멀티플렉싱과 HPACK]\n"
            "HTTP/2는 단일 TCP 연결 내에서 여러 개의 독립적인 양방향 스트림을 전송하는 멀티플렉싱(Multiplexing)을 도입했습니다. "
            "또한 바이너리 프레이밍 계층을 적용하고, 반복되는 헤더 필드를 압축하는 HPACK 알고리즘을 도입하여 네트워크 오버헤드를 대폭 줄였습니다. "
            "그러나 여전히 전송 계층으로 TCP를 사용하므로, TCP 패킷 유실 시 모든 스트림이 멈추는 TCP 수준의 HOLB 문제는 남아 있었습니다.\n\n"
            "[3. HTTP/3: UDP 기반 QUIC 프로토콜]\n"
            "HTTP/3는 TCP 대신 UDP 기반의 QUIC 프로토콜을 사용합니다. "
            "QUIC은 패킷 손실이 발생해도 손실된 스트림만 재전송을 기다리고 나머지 독립적인 스트림은 계속 전달되므로 완벽하게 HOLB를 해결했습니다. "
            "또한 TLS 1.3 암호화 핸드셰이크를 연결 수립 과정에 통합하여 연결 지연 시간(0-RTT)을 혁신적으로 단축했습니다."
        ),
        "quizzes": [
            {
                "question": "HTTP/1.1에서 앞선 요청이 지연될 때 뒤따르는 요청들이 모두 지연되는 현상을 무엇이라 하는가?",
                "options": [
                    "Deadlock",
                    "Head-of-Line Blocking (HOLB)",
                    "Race Condition",
                    "Buffer Overflow"
                ],
                "answer_index": 1,
                "explanation": "순차적 요청 처리로 인해 앞선 요청이 멈추면 뒤의 요청이 블로킹되는 현상을 Head-of-Line Blocking(HOLB)이라 합니다."
            },
            {
                "question": "HTTP/2에서 중복되는 HTTP 헤더를 효율적으로 압축하기 위해 도입된 알고리즘은?",
                "options": [
                    "GZIP",
                    "HPACK",
                    "Brotli",
                    "Snappy"
                ],
                "answer_index": 1,
                "explanation": "HTTP/2는 헤더 압축을 위해 특별히 설계된 HPACK 알고리즘을 사용합니다."
            },
            {
                "question": "HTTP/3가 TCP 대신 채택한 UDP 기반의 차세대 전송 계층 프로토콜은?",
                "options": [
                    "QUIC",
                    "SCTP",
                    "RTSP",
                    "WebSocket"
                ],
                "answer_index": 0,
                "explanation": "HTTP/3는 UDP 기반의 QUIC 프로토콜을 사용하여 TCP 수준의 패킷 손실 블로킹을 근본적으로 해결했습니다."
            }
        ]
    },
    {
        "title": "데이터베이스 트랜잭션과 ACID, 그리고 격리 수준(Isolation Level)",
        "article": (
            "[1. 트랜잭션과 ACID 속성]\n"
            "데이터베이스 트랜잭션은 하나의 논리적 작업 단위를 의미하며, 다음 네 가지 ACID 속성을 만족해야 합니다. "
            "원자성(Atomicity)은 모든 작업이 전부 성공(Commit)하거나 전부 실패(Rollback)해야 함을 보장합니다. "
            "일관성(Consistency)은 트랜잭션 전후에 데이터베이스의 제약 조건과 무결성이 유지되어야 함을 뜻합니다. "
            "격리성(Isolation)은 동시에 실행되는 여러 트랜잭션이 서로 간섭하지 못하게 보장하는 속성입니다. "
            "영속성(Durability)은 성공적으로 완료된 트랜잭션의 결과가 시스템 장애가 발생해도 영구히 보존됨을 뜻합니다.\n\n"
            "[2. 트랜잭션 격리 수준 (Isolation Levels)]\n"
            "격리성이 높을수록 데이터 일관성은 완벽해지지만 동시 처리 성능(Concurrency)은 떨어집니다. ANSI/ISO 표준은 4단계 격리 수준을 정의합니다. "
            "1단계 Read Uncommitted는 커밋되지 않은 데이터도 읽을 수 있어 Dirty Read가 발생합니다. "
            "2단계 Read Committed는 커밋된 데이터만 읽을 수 있지만, 동일 트랜잭션 내에서 같은 조회를 반복할 때 값이 달라지는 Non-Repeatable Read가 발생할 수 있습니다. "
            "3단계 Repeatable Read는 트랜잭션 시작 시점의 스냅샷을 읽어 반복 조회 시 동일한 값을 보장하지만, 새로운 행이 삽입되어 나타나는 Phantom Read가 발생할 수 있습니다 (MySQL InnoDB는 MVCC로 Phantom Read를 대부분 방지). "
            "4단계 Serializable은 가장 엄격한 수준으로 모든 트랜잭션을 순차적으로 실행하여 모든 이상 현상을 방지하지만 동시성 성능이 가장 낮습니다."
        ),
        "quizzes": [
            {
                "question": "트랜잭션의 ACID 속성 중 모든 작업이 '전부 성공'하거나 '전부 롤백'되어야 한다는 원칙은?",
                "options": [
                    "원자성 (Atomicity)",
                    "일관성 (Consistency)",
                    "격리성 (Isolation)",
                    "영속성 (Durability)"
                ],
                "answer_index": 0,
                "explanation": "원자성(Atomicity)은 트랜잭션이 All or Nothing으로 처리되어야 함을 보장합니다."
            },
            {
                "question": "커밋되지 않은 다른 트랜잭션의 변경 데이터를 읽게 되는 이상 현상(Dirty Read)이 허용되는 가장 낮은 격리 수준은?",
                "options": [
                    "Read Uncommitted",
                    "Read Committed",
                    "Repeatable Read",
                    "Serializable"
                ],
                "answer_index": 0,
                "explanation": "Read Uncommitted 격리 수준에서는 커밋되지 않은 데이터의 조회가 허용되어 Dirty Read가 발생합니다."
            },
            {
                "question": "가장 높은 격리성을 제공하여 데이터 부정합을 완벽히 방지하지만 동시 처리 성능이 가장 떨어지는 격리 수준은?",
                "options": [
                    "Serializable",
                    "Read Committed",
                    "Repeatable Read",
                    "Snapshot Isolation"
                ],
                "answer_index": 0,
                "explanation": "Serializable은 모든 트랜잭션을 직렬화하여 실행하므로 데이터 정합성은 완벽하지만 성능 저하가 가장 큽니다."
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

def format_prompt(article: str, count: int) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n다음 장문 블로그 글을 읽고 핵심 내용에 대한 4지선다 객관식 퀴즈 {count}문제를 JSON 배열로 만들어주세요:\n\n{article}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def build_dataset():
    train_file = DATA_DIR / "train_quiz_dataset.jsonl"
    val_file = DATA_DIR / "val_quiz_dataset.jsonl"
    
    rows = []
    for item in LONG_ARTICLE_SAMPLES:
        quizzes = item["quizzes"]
        # 전체 퀴즈 세트
        prompt = format_prompt(item["article"], len(quizzes))
        response = json.dumps(quizzes, ensure_ascii=False, indent=2)
        rows.append({
            "prompt": prompt,
            "response": response,
            "full_text": prompt + response + "<|im_end|>"
        })

        # 2문제 버전도 서브 샘플로 추가 (다양한 count 요청 학습)
        if len(quizzes) >= 2:
            sub_quizzes = quizzes[:2]
            sub_prompt = format_prompt(item["article"], 2)
            sub_response = json.dumps(sub_quizzes, ensure_ascii=False, indent=2)
            rows.append({
                "prompt": sub_prompt,
                "response": sub_response,
                "full_text": sub_prompt + sub_response + "<|im_end|>"
            })

    # Train / Val 분할
    train_rows = rows[:6]
    val_rows = rows[6:]

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"✅ 장문 & 다중 퀴즈 데이터셋 생성 완료: Train {len(train_rows)}개, Val {len(val_rows)}개")

if __name__ == "__main__":
    build_dataset()
