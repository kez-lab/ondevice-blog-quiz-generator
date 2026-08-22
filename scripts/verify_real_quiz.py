#!/usr/bin/env python3
"""
고품질 프롬프트 엔지니어링 & 1.5B 기반 실제 블로그 퀴즈 생성 검증
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

BLOG_TEXT = """
[Android/Compose] Compose에서 SingleLiveEvent를 사용하며 겪은 두 가지 문제

1. 완료된 LaunchedEffect의 scope를 다시 참조하는 문제:
API 호출이 성공하면 특정 위치로 스크롤해야 할 때 LaunchedEffect 안에서 LiveData.observe()를 호출하면, observe()는 등록 즉시 반환되므로 LaunchedEffect 블록이 종료되어 Job이 완료됩니다. 나중에 API 응답이 도착해 observer 내부에서 launch로 suspend 함수(animateScrollToItem)를 실행하려 하면 이미 완료된 부모 Job 때문에 코루틴이 시작하지 못하고 취소됩니다.

2. 조건부 Composition에서 Observer가 중복 등록되는 문제:
if (visible) { LaunchedEffect(Unit) { event.observe(lifecycleOwner) { ... } } } 형태의 코드에서, visible이 false가 되었다가 다시 true가 되어 재진입할 때마다 LaunchedEffect(Unit)이 새로 생성되면서 같은 LifecycleOwner에 observer가 계속 누적 등록됩니다. SingleLiveEvent는 AtomicBoolean 플래그 때문에 한 번만 실행되는 것처럼 보이지만 내부에 미사용 observer 래퍼가 계속 쌓이게 됩니다.

3. 해결 방법:
LiveData를 유지해야 한다면 DisposableEffect의 onDispose에서 removeObserver를 호출하여 등록과 해제를 한 쌍으로 관리해야 합니다. 또는 UI State(scrollTarget)로 단일 이벤트를 처리하거나, ViewModel에서 Channel을 receiveAsFlow()로 변환하여 LaunchedEffect에서 collect하는 구조가 권장됩니다.
"""

PROMPT_TEMPLATE = """다음은 안드로이드 개발 블로그 글입니다.
본문의 내용을 정확히 이해하고, 본문에 등장하는 핵심 기술적 사실(LaunchedEffect 생명주기, Observer 중복 등록, DisposableEffect, Channel 등)을 묻는 4지선다 객관식 퀴즈 3문제를 만들어주세요.

[규칙]
1. 각 문제는 본문의 구체적인 내용을 바탕으로 작성하세요.
2. 4개의 보기는 본문에 나오는 실제 기술 용어로 구성하고, 'Option A'나 '보기1' 같은 임시 텍스트를 절대 사용하지 마세요.
3. 반드시 아래 JSON 배열 형식으로만 출력하세요.

```json
[
  {
    "question": "구체적인 질문 내용",
    "options": ["실제 보기1", "실제 보기2", "실제 보기3", "실제 보기4"],
    "answer_index": 0,
    "explanation": "본문에 근거한 정확한 해설"
  }
]
```

[블로그 본문]
""" + BLOG_TEXT

def test():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"📦 1.5B 모델 로드 중: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    model.eval()
    
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"🔍 고도화된 프롬프트로 퀴즈 생성 중...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    print("\n================ [생성된 실제 퀴즈 결과] ================")
    print(raw_output)
    print("=========================================================\n")

if __name__ == "__main__":
    test()
