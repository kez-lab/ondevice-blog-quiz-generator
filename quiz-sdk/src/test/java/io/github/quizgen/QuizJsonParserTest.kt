package io.github.quizgen

import io.github.quizgen.parser.QuizJsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class QuizJsonParserTest {

    @Test
    fun parse_cleanJsonArray_returnsQuizList() {
        val json = """
        [
          {
            "question": "Kotlin의 코루틴 디스패처 중 I/O에 최적화된 것은?",
            "options": ["Main", "IO", "Default", "Unconfined"],
            "answer_index": 1,
            "explanation": "Dispatchers.IO는 네트워크 및 디스크 I/O에 최적화되어 있습니다."
          }
        ]
        """.trimIndent()

        val quizzes = QuizJsonParser.parse(json)
        assertEquals(1, quizzes.size)
        assertEquals("Kotlin의 코루틴 디스패처 중 I/O에 최적화된 것은?", quizzes[0].question)
        assertEquals(4, quizzes[0].options.size)
        assertEquals(1, quizzes[0].answerIndex)
        assertEquals("IO", quizzes[0].correctAnswer)
        assertTrue(quizzes[0].isCorrect(1))
    }

    @Test
    fun parse_markdownCodeBlock_extractsAndParses() {
        val markdown = """
        다음은 생성된 퀴즈입니다:
        ```json
        [
          {
            "question": "온디바이스 AI의 장점은?",
            "options": ["오프라인 작동", "서버 비용 증가", "높은 지연 시간", "배터리 무제한"],
            "answer_index": 0,
            "explanation": "온디바이스 AI는 오프라인에서도 작동 가능합니다."
          }
        ]
        ```
        확인 부탁드립니다.
        """.trimIndent()

        val quizzes = QuizJsonParser.parse(markdown)
        assertEquals(1, quizzes.size)
        assertEquals("온디바이스 AI의 장점은?", quizzes[0].question)
        assertEquals(0, quizzes[0].answerIndex)
    }

    @Test
    fun parse_singleJsonObject_wrapsInList() {
        val json = """
        {
          "question": "Hugging Face의 핵심 기능은?",
          "options": ["모델 호스팅", "커피 주문", "주식 거래", "날씨 예보"],
          "answer_index": 0,
          "explanation": "Hugging Face는 AI 모델과 데이터셋을 호스팅합니다."
        }
        """.trimIndent()

        val quizzes = QuizJsonParser.parse(json)
        assertEquals(1, quizzes.size)
        assertEquals("Hugging Face의 핵심 기능은?", quizzes[0].question)
        assertEquals(0, quizzes[0].answerIndex)
    }
}
