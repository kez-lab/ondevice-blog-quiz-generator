package io.github.quizgen.demo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import io.github.quizgen.demo.ui.QuizScreen
import io.github.quizgen.demo.ui.theme.BlogQuizTheme

class MainActivity : ComponentActivity() {

    private val viewModel: QuizViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            BlogQuizTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    QuizScreen(viewModel = viewModel)
                }
            }
        }
    }
}
