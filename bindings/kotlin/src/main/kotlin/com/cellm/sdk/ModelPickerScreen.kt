package com.cellm.sdk

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch

/**
 * Model selection screen with HuggingFace download support.
 *
 * Shows available models, their download status, and a progress bar
 * for active downloads. Selecting a downloaded model navigates to
 * the chat screen.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelPickerScreen(
    onModelSelected: (modelPath: String, tokenizerPath: String) -> Unit
) {
    val context = LocalContext.current
    val models = ModelDownloader.availableModels
    var downloadingId by remember { mutableStateOf<String?>(null) }
    var downloadFraction by remember { mutableFloatStateOf(0f) }
    var selectedModel by remember { mutableStateOf<ModelDownloader.ModelSpec?>(null) }
    var error by remember { mutableStateOf<String?>(null) }

    val scope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("cellm Models") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(models) { model ->
                val downloaded = remember { ModelDownloader.isDownloaded(context, model) }
                val isDownloading = downloadingId == model.id

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable(enabled = downloaded) {
                            val files = ModelDownloader.getModelFiles(context, model)
                            if (files != null) {
                                onModelSelected(files.modelPath, files.tokenizerPath)
                            }
                        }
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = model.displayName,
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold
                            )
                            Text(
                                text = formatSize(model.sizeMb),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        Spacer(modifier = Modifier.height(4.dp))

                        Text(
                            text = model.description,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        when {
                            isDownloading -> {
                                LinearProgressIndicator(
                                    progress = downloadFraction,
                                    modifier = Modifier.fillMaxWidth()
                                )
                                Text(
                                    text = "Downloading... ${(downloadFraction * 100).toInt()}%",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                            downloaded -> {
                                Text(
                                    text = "Ready",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                            else -> {
                                Button(
                                    onClick = {
                                        downloadingId = model.id
                                        selectedModel = model
                                        downloadFraction = 0f
                                        error = null

                                        scope.launch {
                                            try {
                                                ModelDownloader.download(
                                                    context, model
                                                ) { progress ->
                                                    downloadFraction = progress.fraction
                                                }
                                                downloadingId = null
                                            } catch (e: Exception) {
                                                error = e.message
                                                downloadingId = null
                                            }
                                        }
                                    }
                                ) {
                                    Text("Download ${formatSize(model.sizeMb)}")
                                }
                            }
                        }
                    }
                }
            }

            if (error != null) {
                item {
                    Text(
                        text = "Error: $error",
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                }
            }
        }
    }
}

private fun formatSize(mb: Int): String {
    return if (mb >= 1000) {
        "%.1f GB".format(mb / 1000.0)
    } else {
        "$mb MB"
    }
}
