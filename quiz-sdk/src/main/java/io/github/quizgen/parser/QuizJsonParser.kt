package io.github.quizgen.parser

import io.github.quizgen.model.Quiz
import kotlinx.serialization.json.Json

/**
 * LLM 출력 텍스트로부터 JSON 퀴즈 리스트를 안전하게 추출/파싱하는 파서
 */
object QuizJsonParser {

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        coerceInputValues = true
    }

    fun parse(rawText: String): List<Quiz> {
        val cleaned = extractJsonSubstring(rawText) ?: return emptyList()

        return try {
            if (cleaned.startsWith("[")) {
                json.decodeFromString<List<Quiz>>(cleaned)
            } else if (cleaned.startsWith("{")) {
                val singleQuiz = json.decodeFromString<Quiz>(cleaned)
                listOf(singleQuiz)
            } else {
                emptyList()
            }
        } catch (e: Exception) {
            // 정규식 기반 폴백 파서 시도
            fallbackRegexParse(rawText)
        }
    }

    private fun extractJsonSubstring(rawText: String): String? {
        val trimmed = rawText.trim()

        // 1. 마크다운 코드 블록 제거 (```json ... ``` 또는 ``` ... ```)
        val codeBlockRegex = Regex("```(?:json)?\\s*([\\s\\S]*?)\\s*```", RegexOption.IGNORE_CASE)
        val matchInBlock = codeBlockRegex.find(trimmed)
        val candidate = matchInBlock?.groupValues?.get(1)?.trim() ?: trimmed

        // 2. 가장 바깥쪽 대괄호 `[...]` 탐색
        val arrayStart = candidate.indexOf('[')
        val arrayEnd = candidate.lastIndexOf(']')
        if (arrayStart != -1 && arrayEnd != -1 && arrayEnd > arrayStart) {
            return candidate.substring(arrayStart, arrayEnd + 1)
        }

        // 3. 가장 바깥쪽 중괄호 `{...}` 탐색 (단일 객체)
        val objStart = candidate.indexOf('{')
        val objEnd = candidate.lastIndexOf('}')
        if (objStart != -1 && objEnd != -1 && objEnd > objStart) {
            return candidate.substring(objStart, objEnd + 1)
        }

        return null
    }

    private fun fallbackRegexParse(rawText: String): List<Quiz> {
        val quizzes = mutableListOf<Quiz>()
        // 정규식으로 question, options, answer_index, explanation 추출 시도
        val questionRegex = Regex("\"question\"\\s*:\\s*\"([^\"]+)\"")
        val explanationRegex = Regex("\"explanation\"\\s*:\\s*\"([^\"]+)\"")
        val answerIndexRegex = Regex("\"answer_index\"\\s*:\\s*(\\d+)")

        val questionMatch = questionRegex.find(rawText)
        val explanationMatch = explanationRegex.find(rawText)
        val answerIndexMatch = answerIndexRegex.find(rawText)

        if (questionMatch != null) {
            val question = questionMatch.groupValues[1]
            val explanation = explanationMatch?.groupValues?.get(1) ?: "본문 내용을 기반으로 생성된 퀴즈입니다."
            val answerIndex = answerIndexMatch?.groupValues?.get(1)?.toIntOrNull() ?: 0

            // Options 추출
            val optionsRegex = Regex("\"options\"\\s*:\\s*\\[([^\\]]+)\\]")
            val optionsMatch = optionsRegex.find(rawText)
            val options = if (optionsMatch != null) {
                optionsMatch.groupValues[1]
                    .split(",")
                    .map { it.replace("\"", "").trim() }
                    .filter { it.isNotEmpty() }
            } else {
                listOf("보기 1", "보기 2", "보기 3", "보기 4")
            }

            quizzes.add(
                Quiz(
                    question = question,
                    options = options,
                    answerIndex = answerIndex.coerceIn(0, (options.size - 1).coerceAtLeast(0)),
                    explanation = explanation
                )
            )
        }

        return quizzes
    }
}
