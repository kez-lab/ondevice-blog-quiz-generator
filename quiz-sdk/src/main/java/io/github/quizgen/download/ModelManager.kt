package io.github.quizgen.download

import android.content.Context
import io.github.quizgen.model.ModelSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

class ModelManager(private val context: Context) {

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    private val modelDir: File
        get() = File(context.filesDir, "quizgen_models").apply { if (!exists()) mkdirs() }

    fun isModelReady(source: ModelSource): Boolean {
        return when (source) {
            is ModelSource.LocalFile -> File(source.absolutePath).exists()
            is ModelSource.Asset -> true
            is ModelSource.HuggingFace -> getTargetFile(source.repoId, source.filename).exists()
            is ModelSource.RemoteUrl -> getTargetFile("remote", source.url.hashCode().toString()).exists()
        }
    }

    suspend fun resolveModelPath(
        source: ModelSource,
        onProgress: (Int) -> Unit = {}
    ): String = withContext(Dispatchers.IO) {
        when (source) {
            is ModelSource.LocalFile -> source.absolutePath
            is ModelSource.Asset -> copyAssetToInternal(source.assetPath)
            is ModelSource.HuggingFace -> {
                val targetFile = getTargetFile(source.repoId, source.filename)
                if (!targetFile.exists() || targetFile.length() == 0L) {
                    downloadFile(source.downloadUrl, targetFile, onProgress)
                }
                targetFile.absolutePath
            }
            is ModelSource.RemoteUrl -> {
                val targetFile = getTargetFile("remote", "${source.url.hashCode()}.task")
                if (!targetFile.exists() || targetFile.length() == 0L) {
                    downloadFile(source.url, targetFile, onProgress)
                }
                targetFile.absolutePath
            }
        }
    }

    fun downloadProgressFlow(source: ModelSource): Flow<Int> = flow {
        when (source) {
            is ModelSource.HuggingFace -> {
                val targetFile = getTargetFile(source.repoId, source.filename)
                if (targetFile.exists() && targetFile.length() > 0L) {
                    emit(100)
                } else {
                    downloadFile(source.downloadUrl, targetFile) { percent ->
                        // Emits via internal loop or callback
                    }
                }
            }
            else -> emit(100)
        }
    }.flowOn(Dispatchers.IO)

    private fun getTargetFile(prefix: String, name: String): File {
        val sanitizedPrefix = prefix.replace("/", "_").replace(":", "_")
        val subDir = File(modelDir, sanitizedPrefix).apply { if (!exists()) mkdirs() }
        return File(subDir, name)
    }

    private fun copyAssetToInternal(assetPath: String): String {
        val fileName = File(assetPath).name
        val targetFile = File(modelDir, fileName)
        if (targetFile.exists() && targetFile.length() > 0) {
            return targetFile.absolutePath
        }

        context.assets.open(assetPath).use { input ->
            FileOutputStream(targetFile).use { output ->
                input.copyTo(output)
            }
        }
        return targetFile.absolutePath
    }

    private fun downloadFile(url: String, targetFile: File, onProgress: (Int) -> Unit) {
        val request = Request.Builder().url(url).build()
        val response = httpClient.newCall(request).execute()

        if (!response.isSuccessful) {
            throw IllegalStateException("모델 다운로드 실패 (HTTP ${response.code}): $url")
        }

        val body = response.body ?: throw IllegalStateException("응답 본문이 비어있습니다: $url")
        val totalBytes = body.contentLength()
        val tempFile = File(targetFile.parentFile, "${targetFile.name}.tmp")

        body.byteStream().use { input ->
            FileOutputStream(tempFile).use { output ->
                val buffer = ByteArray(8192)
                var bytesRead: Int
                var accumulatedBytes = 0L

                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    accumulatedBytes += bytesRead
                    if (totalBytes > 0) {
                        val percent = ((accumulatedBytes * 100) / totalBytes).toInt()
                        onProgress(percent)
                    }
                }
                output.flush()
            }
        }

        if (tempFile.exists()) {
            if (targetFile.exists()) targetFile.delete()
            tempFile.renameTo(targetFile)
        }
        onProgress(100)
    }
}
