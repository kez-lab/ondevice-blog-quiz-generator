package io.github.quizgen

import android.content.Context
import io.github.quizgen.download.ModelManager
import io.github.quizgen.engine.LlmEngine
import io.github.quizgen.engine.MediaPipeLlmEngine
import io.github.quizgen.engine.MockLlmEngine
import io.github.quizgen.model.ModelSource
import io.github.quizgen.model.Quiz
import io.github.quizgen.model.QuizGenerationState
import io.github.quizgen.parser.QuizJsonParser
import io.github.quizgen.prompt.QuizPromptBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.Closeable

/**
 * 온디바이스 블로그 퀴즈 생성기 SDK 메인 엔트리포인트
 */
class LocalQuizGenerator private constructor(
    private val context: Context,
    private val modelSource: ModelSource,
    private val maxTokens: Int,
    private val temperature: Float,
    private val topK: Int,
    private val forceSimulation: Boolean
) : Closeable {

    private val modelManager = ModelManager(context)
    private var activeEngine: LlmEngine? = null

    /**
     * 모델이 기기에 준비되어 있는지 확인
     */
    fun isModelReady(): Boolean = modelManager.isModelReady(modelSource)

    /**
     * 모델을 사전 다운로드/준비
     */
    suspend fun preloadModel(onProgress: (Int) -> Unit = {}): Unit = withContext(Dispatchers.IO) {
        modelManager.resolveModelPath(modelSource, onProgress)
    }

    private suspend fun getOrCreateEngine(onProgress: (Int) -> Unit = {}): LlmEngine {
        if (activeEngine == null) {
            if (forceSimulation) {
                activeEngine = MockLlmEngine()
            } else {
                try {
                    val path = modelManager.resolveModelPath(modelSource, onProgress)
                    activeEngine = MediaPipeLlmEngine(
                        context = context,
                        modelPath = path,
                        maxTokens = maxTokens,
                        temperature = temperature,
                        topK = topK
                    )
                } catch (e: Throwable) {
                    // 에뮬레이터나 GPU 미지원 시 안전하게 Mock 엔진으로 폴백
                    activeEngine = MockLlmEngine()
                }
            }
        }
        return activeEngine!!
    }

    /**
     * 단일 동기/비동기 호출로 퀴즈 리스트 생성
     */
    suspend fun generateQuiz(
        articleText: String,
        count: Int = 1
    ): Result<List<Quiz>> = withContext(Dispatchers.Default) {
        try {
            require(articleText.isNotBlank()) { "본문 텍스트가 비어 있습니다." }
            val engine = getOrCreateEngine()
            val prompt = QuizPromptBuilder.build(articleText, count)
            val rawOutput = engine.generate(prompt)
            val quizzes = QuizJsonParser.parse(rawOutput)

            if (quizzes.isEmpty()) {
                Result.failure(IllegalStateException("퀴즈 생성 결과를 파싱할 수 없습니다:\n$rawOutput"))
            } else {
                Result.success(quizzes)
            }
        } catch (e: Throwable) {
            Result.failure(e)
        }
    }

    /**
     * 실시간 토큰 생성 진행 상태를 Flow로 수신
     */
    fun generateQuizStream(
        articleText: String,
        count: Int = 1
    ): Flow<QuizGenerationState> = flow {
        emit(QuizGenerationState.Idle)

        if (articleText.isBlank()) {
            emit(QuizGenerationState.Error("본문 텍스트가 비어 있습니다."))
            return@flow
        }

        try {
            // 1. 모델 준비 및 다운로드
            emit(QuizGenerationState.DownloadingModel(0))
            val engine = getOrCreateEngine { progress ->
                // Progress callback handled via engine setup
            }

            emit(QuizGenerationState.LoadingModel)

            // 2. 프롬프트 구성 및 스트리밍 추론
            val prompt = QuizPromptBuilder.build(articleText, count)
            val stringBuffer = StringBuilder()

            engine.generateStream(prompt).collect { token ->
                stringBuffer.append(token)
                emit(QuizGenerationState.Generating(stringBuffer.toString()))
            }

            // 3. 최종 JSON 파싱
            val rawText = stringBuffer.toString()
            val quizzes = QuizJsonParser.parse(rawText)

            if (quizzes.isNotEmpty()) {
                emit(QuizGenerationState.Success(quizzes))
            } else {
                emit(QuizGenerationState.Error("퀴즈 파싱 실패", IllegalStateException(rawText)))
            }
        } catch (e: Throwable) {
            emit(QuizGenerationState.Error(e.message ?: "알 수 없는 오류", e))
        }
    }.flowOn(Dispatchers.Default)

    override fun close() {
        activeEngine?.close()
        activeEngine = null
    }

    companion object {
        fun builder(context: Context): Builder = Builder(context.applicationContext)
    }

    class Builder(private val context: Context) {
        private var modelSource: ModelSource = ModelSource.HuggingFace("kez-lab/qwen2.5-0.5b-blog-quiz-android")
        private var maxTokens: Int = 512
        private var temperature: Float = 0.3f
        private var topK: Int = 40
        private var forceSimulation: Boolean = false

        fun fromHuggingFace(repoId: String, filename: String = "model.task") = apply {
            this.modelSource = ModelSource.HuggingFace(repoId, filename)
        }

        fun fromAsset(assetPath: String) = apply {
            this.modelSource = ModelSource.Asset(assetPath)
        }

        fun fromLocalFile(absolutePath: String) = apply {
            this.modelSource = ModelSource.LocalFile(absolutePath)
        }

        fun fromUrl(url: String) = apply {
            this.modelSource = ModelSource.RemoteUrl(url)
        }

        fun setMaxTokens(tokens: Int) = apply {
            this.maxTokens = tokens
        }

        fun setTemperature(temp: Float) = apply {
            this.temperature = temp
        }

        fun setTopK(k: Int) = apply {
            this.topK = k
        }

        fun enableSimulationMode(enable: Boolean = true) = apply {
            this.forceSimulation = enable
        }

        fun build(): LocalQuizGenerator = LocalQuizGenerator(
            context = context,
            modelSource = modelSource,
            maxTokens = maxTokens,
            temperature = temperature,
            topK = topK,
            forceSimulation = forceSimulation
        )
    }
}
