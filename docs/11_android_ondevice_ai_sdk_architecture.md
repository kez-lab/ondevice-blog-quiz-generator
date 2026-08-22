# 📱 [11] 안드로이드 온디바이스 AI SDK 및 오픈소스 배포 아키텍처

AI 모델을 안드로이드 앱에서 부드럽게 구동하고, 다른 개발자들이 쓸 수 있는 오픈소스 SDK로 만드는 엔지니어링 가이드입니다.

---

## 1. 안드로이드 온디바이스 엔진: Google MediaPipe GenAI & LiteRT

구글은 안드로이드에서 sLLM을 돌리기 위해 **MediaPipe Tasks GenAI (`com.google.mediapipe:tasks-genai`)**와 **LiteRT(구 TensorFlow Lite)**를 제공합니다.

* **네이티브 C++ & GPU 가속**: 안드로이드의 NPU/GPU 드라이버(Vulkan, OpenCL)와 직접 연결되어 초고속 연산 수행
* **KV 캐시 관리**: 대화나 글이 길어져도 이전 문맥의 토큰 캐시를 메모리에 효율적으로 유지
* **비동기 스트리밍 (`LlmInferenceSession`)**: 문장이 전부 완성될 때까지 기다리지 않고, 1글자씩 실시간으로 안드로이드 UI에 출력

---

## 2. Kotlin Coroutines `Flow` 기반 클린 API 설계

우리가 개발한 `:quiz-sdk` 라이브러리의 핵심 아키텍처입니다:

```kotlin
// SDK 내부 동작 흐름
fun generateQuizStream(article: String, count: Int): Flow<QuizGenerationState> = flow {
    emit(QuizGenerationState.DownloadingModel) // 1. 허깅페이스에서 모델 다운로드
    emit(QuizGenerationState.LoadingModel)      // 2. 모바일 GPU에 적재
    
    // 3. 실시간 토큰 방출
    llmEngine.generateStream(prompt).collect { token ->
        emit(QuizGenerationState.Generating(token))
    }
    
    // 4. 최종 JSON 파싱 및 데이터 클래스 변환
    emit(QuizGenerationState.Success(quizzes))
}
```

---

## 3. JitPack을 통한 GitHub 오픈소스 라이브러리 배포

안드로이드 개발자가 내 라이브러리를 `build.gradle.kts`에 `implementation(...)` 한 줄로 가져다 쓸 수 있게 만드는 과정입니다:

1. 안드로이드 모듈에 `maven-publish` 플러그인을 설정합니다.
2. 깃허브 레포지토리에 코드를 푸시하고 **Release Tag (예: `v1.0.0`)**를 발행합니다.
3. [JitPack.io](https://jitpack.io) 웹사이트에 내 깃허브 URL(`github.com/내아이디/레포지토리명`)을 입력하고 **Get It** 버튼을 누르면 전 세계에 배포됩니다.
