package io.github.quizgen.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * 온디바이스 AI에 의해 생성된 퀴즈 객체
 */
@Serializable
data class Quiz(
    @SerialName("question")
    val question: String,

    @SerialName("options")
    val options: List<String>,

    @SerialName("answer_index")
    val answerIndex: Int,

    @SerialName("explanation")
    val explanation: String
) {
    val correctAnswer: String
        get() = options.getOrElse(answerIndex) { "" }

    fun isCorrect(selectedIndex: Int): Boolean = selectedIndex == answerIndex
}

/**
 * 퀴즈 생성 진행 상태 (UI 스트리밍 연동용)
 */
sealed interface QuizGenerationState {
    data object Idle : QuizGenerationState
    data class DownloadingModel(val progressPercent: Int) : QuizGenerationState
    data object LoadingModel : QuizGenerationState
    data class Generating(val rawTokens: String) : QuizGenerationState
    data class Success(val quizzes: List<Quiz>) : QuizGenerationState
    data class Error(val message: String, val cause: Throwable? = null) : QuizGenerationState
}

/**
 * AI 모델 소스 지정
 */
sealed interface ModelSource {
    /**
     * Hugging Face Hub에서 다운로드 (예: "your-username/qwen2.5-0.5b-blog-quiz-android")
     */
    data class HuggingFace(
        val repoId: String,
        val filename: String = "model.task"
    ) : ModelSource {
        val downloadUrl: String
            get() = "https://huggingface.co/$repoId/resolve/main/$filename"
    }

    /**
     * 앱 내부 assets/ 폴더에 포함된 모델
     */
    data class Asset(val assetPath: String) : ModelSource

    /**
     * 기기 로컬 파일 경로
     */
    data class LocalFile(val absolutePath: String) : ModelSource

    /**
     * 임의의 원격 URL
     */
    data class RemoteUrl(val url: String) : ModelSource
}
