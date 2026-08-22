package io.github.quizgen.engine

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * 에뮬레이터 또는 네이티브 GPU 미지원 환경 및 단위 테스트를 위한 시뮬레이션 엔진
 */
class MockLlmEngine(
    private val simulatedDelayMs: Long = 100L
) : LlmEngine {

    private val sampleResponses = listOf(
        """[
  {
    "question": "본문에서 다루는 기술의 가장 핵심적인 목적은 무엇인가요?",
    "options": [
      "네트워크 트래픽과 통신 지연 시간을 줄이기 위해",
      "복잡한 코드를 단순화하고 가독성을 높이기 위해",
      "서버 비용 없이 온디바이스에서 안전하게 연산하기 위해",
      "데이터베이스 쿼리 속도를 2배 이상 향상시키기 위해"
    ],
    "answer_index": 2,
    "explanation": "본문의 기술은 외부 서버 전송 없이 기기 로컬에서 개인정보를 보호하며 빠른 처리를 지원하는 것이 핵심 목적입니다."
  },
  {
    "question": "본문 내용에 비추어 볼 때 가장 권장되는 최적화 기법은?",
    "options": [
      "양자화(Quantization)를 통한 저비트 정수 변환",
      "모든 연산을 CPU 싱글스레드로만 실행",
      "원시 부동소수점(FP32)을 그대로 유지",
      "실행 시마다 모델 전체를 매번 재다운로드"
    ],
    "answer_index": 0,
    "explanation": "모바일 환경의 메모리 제약을 극복하기 위해 양자화(Quantization)를 통한 경량화가 필수적으로 권장됩니다."
  }
]"""
    )

    override suspend fun generate(prompt: String): String {
        delay(simulatedDelayMs * 5)
        return sampleResponses.first()
    }

    override fun generateStream(prompt: String): Flow<String> = flow {
        val target = sampleResponses.first()
        val chunkSize = 6
        var i = 0
        while (i < target.length) {
            val end = (i + chunkSize).coerceAtMost(target.length)
            emit(target.substring(i, end))
            i = end
            delay(simulatedDelayMs)
        }
    }

    override fun close() {
        // No-op
    }
}
