package io.github.quizgen.engine

import android.content.Context
import com.google.mediapipe.tasks.core.ErrorListener
import com.google.mediapipe.tasks.core.OutputHandler
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext

class MediaPipeLlmEngine(
    private val context: Context,
    private val modelPath: String,
    private val maxTokens: Int = 512,
    private val topK: Int = 40,
    private val temperature: Float = 0.3f
) : LlmEngine {

    private var llmInference: LlmInference? = null

    private fun getOrInitEngine(): LlmInference {
        if (llmInference == null) {
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelPath)
                .setMaxTokens(maxTokens)
                .setMaxTopK(topK)
                .build()
            llmInference = LlmInference.createFromOptions(context, options)
        }
        return llmInference!!
    }

    override suspend fun generate(prompt: String): String = withContext(Dispatchers.Default) {
        val engine = getOrInitEngine()
        engine.generateResponse(prompt)
    }

    override fun generateStream(prompt: String): Flow<String> = callbackFlow {
        val sessionOptions = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelPath)
            .setMaxTokens(maxTokens)
            .setMaxTopK(topK)
            .setResultListener(OutputHandler.ProgressListener<String> { partialResult: String, done: Boolean ->
                trySend(partialResult)
                if (done) {
                    channel.close()
                }
            })
            .setErrorListener(ErrorListener { error: RuntimeException ->
                channel.close(IllegalStateException("MediaPipe 추론 에러: ${error.message}", error))
            })
            .build()

        val streamingEngine = LlmInference.createFromOptions(context, sessionOptions)
        streamingEngine.generateResponseAsync(prompt)

        awaitClose {
            streamingEngine.close()
        }
    }.flowOn(Dispatchers.Default)

    override fun close() {
        llmInference?.close()
        llmInference = null
    }
}
