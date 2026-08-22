package io.github.quizgen

import io.github.quizgen.prompt.QuizPromptBuilder
import org.junit.Assert.assertTrue
import org.junit.Test

class QuizPromptBuilderTest {

    @Test
    fun build_containsSystemPromptAndArticle() {
        val blogText = "안드로이드 14 백그라운드 서비스 정책 요약"
        val prompt = QuizPromptBuilder.build(blogText, count = 2)

        assertTrue(prompt.contains("<|im_start|>system"))
        assertTrue(prompt.contains("<|im_start|>user"))
        assertTrue(prompt.contains("<|im_start|>assistant"))
        assertTrue(prompt.contains("2 문제"))
        assertTrue(prompt.contains("안드로이드 14 백그라운드 서비스 정책 요약"))
    }
}
