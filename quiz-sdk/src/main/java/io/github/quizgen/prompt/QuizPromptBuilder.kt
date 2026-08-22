package io.github.quizgen.prompt

/**
 * 온디바이스 sLLM(Qwen2.5)에 최적화된 프롬프트 생성 빌더
 */
object QuizPromptBuilder {

    private const val SYSTEM_PROMPT =
        "당신은 주어진 글(블로그, 아티클, 문서)을 분석하여 핵심 내용을 묻는 4지선다 객관식 퀴즈를 생성하는 AI입니다.\n" +
        "반드시 아래 JSON 형식으로만 응답하세요:\n" +
        "[\n" +
        "  {\n" +
        "    \"question\": \"문제 내용\",\n" +
        "    \"options\": [\"보기1\", \"보기2\", \"보기3\", \"보기4\"],\n" +
        "    \"answer_index\": 0,\n" +
        "    \"explanation\": \"정답에 대한 명확한 해설\"\n" +
        "  }\n" +
        "]"

    fun build(articleText: String, count: Int = 1): String {
        val userInstruction = if (count > 1) {
            "다음 글을 꼼꼼히 읽고 중요한 핵심 개념에 대해 객관식 퀴즈 $count 문제를 JSON 배열로 만들어주세요:\n\n$articleText"
        } else {
            "다음 글을 읽고 객관식 퀴즈 1문제를 JSON 형식으로 만들어주세요:\n\n$articleText"
        }

        // Qwen2.5 ChatML 포맷 적용
        return "<|im_start|>system\n$SYSTEM_PROMPT<|im_end|>\n" +
               "<|im_start|>user\n$userInstruction<|im_end|>\n" +
               "<|im_start|>assistant\n"
    }
}
