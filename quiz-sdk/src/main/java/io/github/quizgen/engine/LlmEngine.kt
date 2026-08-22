package io.github.quizgen.engine

import kotlinx.coroutines.flow.Flow

/**
 * 온디바이스 LLM 추론 엔진 공통 인터페이스
 */
interface LlmEngine {
    /**
     * 프롬프트를 전달받아 생성된 텍스트 전체를 반환
     */
    suspend fun generate(prompt: String): String

    /**
     * 프롬프트를 전달받아 실시간 토큰 스트림으로 반환
     */
    fun generateStream(prompt: String): Flow<String>

    /**
     * 엔진 메모리 해제
     */
    fun close()
}
