#!/usr/bin/env python3
"""
🌐 온디바이스 블로그 퀴즈 생성 AI - 인터랙티브 웹 데모 (Gradio)
- 허깅페이스 모델(kez-lab/qwen2.5-0.5b-blog-quiz-android) 또는 로컬 LoRA 가중치 로드
- 웹 브라우저에서 실시간으로 블로그 글을 넣고 4지선다 퀴즈 생성 및 즉시 풀기/채점 인터랙션 제공
"""

import json
import re
import torch
import gradio as gr
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_LORA_DIR = Path(__file__).parent / "output" / "qwen2.5-0.5b-blog-quiz-lora"
HF_REPO_ID = "kez-lab/qwen2.5-0.5b-blog-quiz-android"

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

# 전역 모델 & 토크나이저 캐시
model = None
tokenizer = None
device = "mps" if torch.backends.mps.is_available() else "cpu"

def load_ai_model():
    global model, tokenizer
    if model is not None:
        return
    
    print(f"📦 [Web Demo] 베이스 모델 로드 중: {MODEL_ID} (Device: {device})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    
    # 1. 로컬 가중치가 있으면 로컬 로드, 없으면 허깅페이스 Hub에서 직접 로드
    if LOCAL_LORA_DIR.exists() and (LOCAL_LORA_DIR / "adapter_model.safetensors").exists():
        print(f"🎯 로컬 LoRA 어댑터 로드 중: {LOCAL_LORA_DIR}")
        model = PeftModel.from_pretrained(base_model, str(LOCAL_LORA_DIR))
    else:
        print(f"🌐 허깅페이스 Hub에서 LoRA 어댑터 다운로드 및 결합 중: {HF_REPO_ID}")
        model = PeftModel.from_pretrained(base_model, HF_REPO_ID)
        
    model.eval()
    print("✅ 모델 준비 완료!")

def robust_parse_quizzes(raw_text: str):
    match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
    if not match:
        return []
    
    clean_str = match.group(0)
    clean_str = re.sub(r'\"\)', '\"', clean_str)
    clean_str = re.sub(r',\s*\]', ']', clean_str)
    clean_str = re.sub(r',\s*\}', '}', clean_str)

    try:
        return json.loads(clean_str)
    except Exception:
        pass

    # 정규식 폴백
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

def generate_quiz_web(article_text: str, question_count: int):
    if not article_text or len(article_text.strip()) < 20:
        return "⚠️ 글의 길이가 너무 짧습니다. 최소 20자 이상의 블로그 글이나 문서를 입력해주세요.", ""
    
    load_ai_model()
    
    user_prompt = f"다음 블로그 글을 읽고 핵심 내용에 대한 4지선다 객관식 퀴즈 {question_count}문제를 JSON 배열로 만들어주세요:\n\n{article_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
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
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    quizzes = robust_parse_quizzes(raw_output)
    
    if not quizzes:
        return f"⚠️ 퀴즈 파싱 실패\n\nAI 원본 출력:\n{raw_output}", raw_output
    
    # 예쁜 HTML 퀴즈 카드 렌더링
    html_cards = "<div style='font-family: system-ui, -apple-system, sans-serif; display: flex; flex-direction: column; gap: 20px;'>"
    for i, q in enumerate(quizzes, 1):
        q_text = q.get('question', '')
        opts = q.get('options', [])
        ans_idx = q.get('answer_index', 0)
        exp = q.get('explanation', '')
        
        html_cards += f"""
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);'>
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 12px;'>
                <span style='background: #3b82f6; color: white; border-radius: 6px; padding: 4px 10px; font-weight: bold; font-size: 14px;'>Q{i}</span>
                <h3 style='margin: 0; font-size: 17px; color: #1e293b;'>{q_text}</h3>
            </div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px;'>
        """
        
        for idx, opt in enumerate(opts):
            is_correct = (idx == ans_idx)
            badge = "✅ (정답)" if is_correct else ""
            border_color = "#22c55e" if is_correct else "#e2e8f0"
            bg_color = "#f0fdf4" if is_correct else "#f8fafc"
            
            html_cards += f"""
                <div style='border: 2px solid {border_color}; background: {bg_color}; border-radius: 8px; padding: 12px 16px; font-size: 14px; color: #334155;'>
                    <strong>{idx + 1}.</strong> {opt} <span style='font-weight: bold; color: #16a34a;'>{badge}</span>
                </div>
            """
            
        html_cards += f"""
            </div>
            <div style='margin-top: 15px; background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 4px;'>
                <p style='margin: 0; font-size: 13px; color: #1e40af;'><strong>💡 해설:</strong> {exp}</p>
            </div>
        </div>
        """
    html_cards += "</div>"
    
    return html_cards, raw_output

# 샘플 프리셋 아티클들
SAMPLE_ARTICLES = {
    "📱 안드로이드 14 백그라운드 & WorkManager": (
        "안드로이드 14(API 34)부터는 포그라운드 서비스(Foreground Service) 실행 시 매니페스트에 반드시 foregroundServiceType(예: location, mediaPlayback 등)을 명시해야 합니다. "
        "타입을 선언하지 않으면 SecurityException이 발생하여 앱이 즉시 강제 종료됩니다. "
        "한편 지연 가능하고 기기 재부팅 후에도 유지되어야 하는 영속적 작업에는 WorkManager가 표준 권장 사항입니다. "
        "WorkManager는 네트워크 연결 상태나 충전 여부 같은 제약 조건(Constraints)을 지원합니다."
    ),
    "⚡ Kotlin Coroutine & StateFlow": (
        "Kotlin 코루틴은 OS 네이티브 스레드를 매번 생성하는 대신, 기존 스레드를 차단(blocking)하지 않고 실행을 중단(suspend)했다가 재개(resume)하는 경량 스레드입니다. "
        "Dispatchers.IO는 네트워크 통신 및 디스크 I/O 작업에 특화되어 있으며, Dispatchers.Default는 대규모 데이터 정렬 등 CPU 집약적 연산에 최적화되어 있습니다. "
        "StateFlow는 항상 최신 상태를 유지하며 UI 레이어에서 상태 홀더로 사용되는 Hot Stream입니다."
    ),
    "🤖 온디바이스 AI & INT4 양자화": (
        "온디바이스 AI는 클라우드 서버 없이 기기 내부의 NPU/GPU에서 모델을 직접 구동하여 100% 개인정보 보호와 오프라인 실행을 보장합니다. "
        "32비트 부동소수점(FP32) 가중치를 4비트(INT4) 정수로 압축하는 양자화(Quantization)를 적용하여 모델 용량을 수백 메가바이트로 축소할 수 있습니다. "
        "LoRA(Low-Rank Adaptation)는 원본 모델을 고정하고 저순위 어댑터 행렬만 학습시켜 파인튜닝 효율을 극대화합니다."
    )
}

def create_ui():
    custom_css = """
    .gradio-container { max-width: 1000px !important; margin: auto; }
    """
    with gr.Blocks(title="🧠 온디바이스 퀴즈 생성 AI 웹 데모", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧠 On-Device Blog Quiz Generator (Web Demo)
            > **Hugging Face Hub**: [`kez-lab/qwen2.5-0.5b-blog-quiz-android`](https://huggingface.co/kez-lab/qwen2.5-0.5b-blog-quiz-android)  
            > **GitHub Repository**: [`kez-lab/ondevice-blog-quiz-generator`](https://github.com/kez-lab/ondevice-blog-quiz-generator)
            
            블로그 글이나 기술 문서를 입력하면, 파인튜닝된 **Qwen2.5-0.5B AI 모델**이 핵심 내용을 분석하여 4지선다 객관식 퀴즈를 실시간으로 생성합니다!
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 1. 블로그 글 입력")
                preset_dropdown = gr.Dropdown(
                    choices=list(SAMPLE_ARTICLES.keys()),
                    label="📌 샘플 프리셋 선택",
                    value=list(SAMPLE_ARTICLES.keys())[0]
                )
                article_input = gr.Textbox(
                    label="블로그 본문 텍스트 (최대 10,000자 지원)",
                    lines=10,
                    value=SAMPLE_ARTICLES[list(SAMPLE_ARTICLES.keys())[0]],
                    placeholder="여기에 블로그 글을 붙여넣으세요..."
                )
                count_slider = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="🎯 생성할 퀴즈 문항 수"
                )
                generate_btn = gr.Button("🎲 4지선다 퀴즈 생성하기", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 2. 생성된 퀴즈 카드 & 해설")
                quiz_output_html = gr.HTML(label="생성된 퀴즈 카드")
                with gr.Accordion("🔍 AI 원본 JSON 출력 보기", open=False):
                    raw_output_text = gr.Code(label="JSON Response", language="json")
                    
        def on_preset_change(choice):
            return SAMPLE_ARTICLES.get(choice, "")
            
        preset_dropdown.change(fn=on_preset_change, inputs=[preset_dropdown], outputs=[article_input])
        generate_btn.click(
            fn=generate_quiz_web,
            inputs=[article_input, count_slider],
            outputs=[quiz_output_html, raw_output_text]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_ui()
    # 로컬 웹 서버 시작 (브라우저 자동 열림)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
