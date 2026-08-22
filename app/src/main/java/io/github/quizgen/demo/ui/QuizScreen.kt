package io.github.quizgen.demo.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Lightbulb
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.quizgen.demo.PresetArticle
import io.github.quizgen.demo.QuizViewModel
import io.github.quizgen.demo.ui.theme.ErrorRed
import io.github.quizgen.demo.ui.theme.SuccessGreen
import io.github.quizgen.model.Quiz
import io.github.quizgen.model.QuizGenerationState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun QuizScreen(
    viewModel: QuizViewModel,
    modifier: Modifier = Modifier
) {
    val inputText by viewModel.inputText.collectAsState()
    val uiState by viewModel.uiState.collectAsState()
    val selectedAnswers = viewModel.selectedAnswers

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Memory,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(28.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Column {
                            Text(
                                text = "Blog Quiz AI",
                                fontWeight = FontWeight.Bold,
                                fontSize = 20.sp
                            )
                            Text(
                                text = "On-Device Qwen2.5-0.5B",
                                fontSize = 12.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        },
        modifier = modifier
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            item {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "💡 샘플 블로그 아티클 선택",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(8.dp))
                PresetChips(
                    presets = viewModel.presetArticles,
                    onSelectPreset = { viewModel.applyPreset(it) }
                )
            }

            item {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = { viewModel.onInputTextChanged(it) },
                    label = { Text("블로그 본문 텍스트") },
                    placeholder = { Text("퀴즈를 만들고 싶은 기술 글이나 메모를 입력하세요...") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(140.dp),
                    shape = RoundedCornerShape(12.dp)
                )
            }

            item {
                val isLoading = uiState is QuizGenerationState.LoadingModel ||
                                uiState is QuizGenerationState.DownloadingModel ||
                                uiState is QuizGenerationState.Generating

                Button(
                    onClick = { viewModel.generateQuizzes(count = 2) },
                    enabled = !isLoading && inputText.isNotBlank(),
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(52.dp),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    if (isLoading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(24.dp),
                            color = MaterialTheme.colorScheme.onPrimary,
                            strokeWidth = 2.5.dp
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(
                            text = when (uiState) {
                                is QuizGenerationState.DownloadingModel -> "모델 다운로드 중..."
                                is QuizGenerationState.LoadingModel -> "온디바이스 엔진 적재 중..."
                                is QuizGenerationState.Generating -> "퀴즈 추론 중..."
                                else -> "처리 중..."
                            }
                        )
                    } else {
                        Icon(imageVector = Icons.Default.AutoAwesome, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(text = "온디바이스 AI 퀴즈 생성", fontSize = 16.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }

            // 실시간 스트리밍 생성 토큰 프리뷰
            if (uiState is QuizGenerationState.Generating) {
                item {
                    val rawTokens = (uiState as QuizGenerationState.Generating).rawTokens
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                        ),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            Text(
                                text = "⚡ 실시간 로컬 추론 스트리밍:",
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = rawTokens,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp,
                                lineHeight = 16.sp
                            )
                        }
                    }
                }
            }

            // 에러 표시
            if (uiState is QuizGenerationState.Error) {
                item {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = "❌ ${(uiState as QuizGenerationState.Error).message}",
                            color = MaterialTheme.colorScheme.onErrorContainer,
                            modifier = Modifier.padding(16.dp)
                        )
                    }
                }
            }

            // 생성 완료된 퀴즈 카드 목록
            if (uiState is QuizGenerationState.Success) {
                val quizzes = (uiState as QuizGenerationState.Success).quizzes
                item {
                    Text(
                        text = "📝 생성된 온디바이스 퀴즈 (${quizzes.size}문항)",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                itemsIndexed(quizzes) { index, quiz ->
                    AnimatedVisibility(
                        visible = true,
                        enter = fadeIn() + slideInVertically()
                    ) {
                        QuizCard(
                            quizIndex = index,
                            quiz = quiz,
                            selectedOption = selectedAnswers[index],
                            onSelectOption = { optIdx -> viewModel.selectOption(index, optIdx) }
                        )
                    }
                }
            }

            item {
                Spacer(modifier = Modifier.height(24.dp))
            }
        }
    }
}

@Composable
fun PresetChips(
    presets: List<PresetArticle>,
    onSelectPreset: (PresetArticle) -> Unit
) {
    LazyRow(
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(presets) { preset ->
            SuggestionChip(
                onClick = { onSelectPreset(preset) },
                label = { Text(preset.title) },
                shape = RoundedCornerShape(8.dp)
            )
        }
    }
}

@Composable
fun QuizCard(
    quizIndex: Int,
    quiz: Quiz,
    selectedOption: Int?,
    onSelectOption: (Int) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Q${quizIndex + 1}. ${quiz.question}",
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp,
                lineHeight = 22.sp
            )

            Spacer(modifier = Modifier.height(12.dp))

            quiz.options.forEachIndexed { optIndex, optionText ->
                val isSelected = selectedOption == optIndex
                val isAnswer = optIndex == quiz.answerIndex
                val isAnswered = selectedOption != null

                val backgroundColor = when {
                    !isAnswered -> MaterialTheme.colorScheme.surface
                    isSelected && isAnswer -> SuccessGreen.copy(alpha = 0.2f)
                    isSelected && !isAnswer -> ErrorRed.copy(alpha = 0.2f)
                    !isSelected && isAnswer -> SuccessGreen.copy(alpha = 0.1f)
                    else -> MaterialTheme.colorScheme.surface.copy(alpha = 0.5f)
                }

                val borderColor = when {
                    isSelected && isAnswer -> SuccessGreen
                    isSelected && !isAnswer -> ErrorRed
                    !isSelected && isAnswer && isAnswered -> SuccessGreen
                    else -> MaterialTheme.colorScheme.outlineVariant
                }

                OutlinedCard(
                    onClick = { if (!isAnswered) onSelectOption(optIndex) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    shape = RoundedCornerShape(10.dp),
                    border = BorderStroke(1.5.dp, borderColor),
                    colors = CardDefaults.outlinedCardColors(containerColor = backgroundColor)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "${optIndex + 1})",
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.width(24.dp)
                        )
                        Text(
                            text = optionText,
                            fontSize = 14.sp,
                            modifier = Modifier.weight(1f)
                        )
                        if (isAnswered) {
                            if (isSelected && isAnswer) {
                                Icon(Icons.Default.CheckCircle, contentDescription = "정답", tint = SuccessGreen)
                            } else if (isSelected && !isAnswer) {
                                Icon(Icons.Default.Close, contentDescription = "오답", tint = ErrorRed)
                            }
                        }
                    }
                }
            }

            // 해설 표시 (보기를 클릭한 후 펼쳐짐)
            AnimatedVisibility(visible = selectedOption != null) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 12.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.4f))
                        .padding(12.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Lightbulb,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "정답 해설",
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = quiz.explanation,
                        fontSize = 13.sp,
                        color = MaterialTheme.colorScheme.onSurface,
                        lineHeight = 18.sp
                    )
                }
            }
        }
    }
}
