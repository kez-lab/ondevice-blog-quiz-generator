package io.github.quizgen.demo

import android.app.Application
import androidx.compose.runtime.mutableStateMapOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.quizgen.LocalQuizGenerator
import io.github.quizgen.model.Quiz
import io.github.quizgen.model.QuizGenerationState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class PresetArticle(val title: String, val content: String)

class QuizViewModel(application: Application) : AndroidViewModel(application) {

    val presetArticles = listOf(
        PresetArticle(
            title = "코틀린 코루틴",
            content = "Kotlin의 Coroutine은 스레드를 차단(blocking)하지 않고 일시 중단(suspend)할 수 있는 경량 스레드입니다. Dispatchers.IO는 네트워크나 파일 입출력(I/O)에 최적화되어 있고, Dispatchers.Default는 CPU 집약적 연산에 사용됩니다."
        ),
        PresetArticle(
            title = "온디바이스 AI",
            content = "온디바이스 AI는 외부 클라우드 서버 없이 스마트폰 내부의 NPU/GPU/CPU에서 머신러닝 모델을 직접 구동합니다. 데이터가 기기 밖으로 나가지 않아 프라이버시가 완벽히 보호되며 오프라인에서도 지연 없이 즉시 실행됩니다."
        ),
        PresetArticle(
            title = "Jetpack Compose",
            content = "Jetpack Compose는 안드로이드의 최신 선언형 UI 프레임워크입니다. 데이터의 변경에 따라 Composable 함수가 자동으로 다시 실행되는 리컴포지션(Recomposition) 과정을 통해 화면을 갱신합니다."
        )
    )

    private val generator: LocalQuizGenerator = LocalQuizGenerator.builder(application)
        .fromHuggingFace("kez-lab/gemma-2-2b-quiz-korean")
        .build()

    private val _inputText = MutableStateFlow(presetArticles.first().content)
    val inputText: StateFlow<String> = _inputText.asStateFlow()

    private val _uiState = MutableStateFlow<QuizGenerationState>(QuizGenerationState.Idle)
    val uiState: StateFlow<QuizGenerationState> = _uiState.asStateFlow()

    // 퀴즈 번호 -> 사용자가 선택한 보기 인덱스 저장
    val selectedAnswers = mutableStateMapOf<Int, Int>()

    fun onInputTextChanged(newText: String) {
        _inputText.value = newText
    }

    fun applyPreset(preset: PresetArticle) {
        _inputText.value = preset.content
        selectedAnswers.clear()
    }

    fun generateQuizzes(count: Int = 2) {
        val text = _inputText.value.trim()
        if (text.isBlank()) return

        selectedAnswers.clear()
        viewModelScope.launch {
            generator.generateQuizStream(text, count).collect { state ->
                _uiState.value = state
            }
        }
    }

    fun selectOption(quizIndex: Int, optionIndex: Int) {
        if (!selectedAnswers.containsKey(quizIndex)) {
            selectedAnswers[quizIndex] = optionIndex
        }
    }

    override fun onCleared() {
        super.onCleared()
        generator.close()
    }
}
