# MediaPipe GenAI Proguard rules
-keep class com.google.mediapipe.tasks.genai.** { *; }
-keep interface com.google.mediapipe.tasks.genai.** { *; }
-dontwarn com.google.mediapipe.tasks.genai.**

# Keep Quiz Data Models for serialization
-keep class io.github.quizgen.model.** { *; }
-keepclassmembers class io.github.quizgen.model.** { *; }
