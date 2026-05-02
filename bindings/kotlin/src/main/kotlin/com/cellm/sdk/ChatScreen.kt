package com.cellm.sdk

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch

/**
 * Chat interface for cellm inference.
 *
 * Displays message history with streaming assistant responses.
 * Requires [ChatViewModel] to be provided by the composition.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    modelPath: String,
    tokenizerPath: String,
    onBack: () -> Unit,
    chatViewModel: ChatViewModel = viewModel()
) {
    val state by chatViewModel.state.collectAsState()
    val scope = rememberCoroutineScope()
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    // Load model on first composition.
    LaunchedEffect(modelPath) {
        chatViewModel.loadModel(modelPath, tokenizerPath)
    }

    // Auto-scroll to bottom when new messages arrive.
    LaunchedEffect(state.messages.size) {
        if (state.messages.isNotEmpty()) {
            listState.animateScrollToItem(state.messages.size - 1)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = state.modelName.ifEmpty { "Loading..." },
                            style = MaterialTheme.typography.titleSmall
                        )
                        if (state.isModelLoaded) {
                            Text(
                                text = "${state.backendName} | ${"%.1f".format(state.tokPerSec)} tok/s",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                },
                navigationIcon = {
                    TextButton(onClick = onBack) {
                        Text("Back")
                    }
                },
                actions = {
                    if (state.isModelLoaded) {
                        TextButton(onClick = { chatViewModel.resetSession() }) {
                            Text("Reset")
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        },
        bottomBar = {
            if (state.isModelLoaded) {
                Surface(
                    tonalElevation = 2.dp,
                    shadowElevation = 4.dp
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        verticalAlignment = Alignment.Bottom
                    ) {
                        OutlinedTextField(
                            value = inputText,
                            onValueChange = { inputText = it },
                            modifier = Modifier.weight(1f),
                            placeholder = { Text("Type a message...") },
                            enabled = !state.isGenerating,
                            maxLines = 4
                        )

                        Spacer(modifier = Modifier.width(8.dp))

                        IconButton(
                            onClick = {
                                if (inputText.isNotBlank()) {
                                    val text = inputText
                                    inputText = ""
                                    scope.launch {
                                        chatViewModel.sendMessage(text)
                                    }
                                }
                            },
                            enabled = !state.isGenerating && inputText.isNotBlank()
                        ) {
                            Icon(
                                imageVector = Icons.Filled.Send,
                                contentDescription = "Send"
                            )
                        }
                    }
                }
            }
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            when {
                state.isLoading -> {
                    Column(
                        modifier = Modifier.align(Alignment.Center),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("Loading model...")
                    }
                }

                state.error != null -> {
                    Column(
                        modifier = Modifier.align(Alignment.Center),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = state.error!!,
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }

                else -> {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        items(state.messages) { message ->
                            ChatBubble(message = message)
                        }

                        if (state.isGenerating) {
                            item {
                                Row(
                                    modifier = Modifier.padding(8.dp),
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(16.dp),
                                        strokeWidth = 2.dp
                                    )
                                    Text(
                                        text = "Generating...",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ChatBubble(message: ChatViewModel.ChatMessage) {
    val isUser = message.role == ChatViewModel.Role.USER
    val isSystem = message.role == ChatViewModel.Role.SYSTEM

    val backgroundColor = when {
        isSystem -> MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
        isUser -> MaterialTheme.colorScheme.primaryContainer
        else -> MaterialTheme.colorScheme.secondaryContainer
    }

    val alignment = if (isUser) Alignment.End else Alignment.Start

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = alignment
    ) {
        Surface(
            modifier = Modifier.widthIn(max = 300.dp),
            shape = RoundedCornerShape(12.dp),
            color = backgroundColor
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                if (!isSystem) {
                    Text(
                        text = if (isUser) "You" else "Assistant",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                }
                Text(
                    text = message.text,
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}
