package com.cellm.sdk

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * ViewModel managing chat inference state for a single model.
 *
 * Holds the loaded engine, tokenizer, and chat message history.
 * The decode loop runs on a background dispatcher and updates
 * the UI via StateFlow emissions.
 */
class ChatViewModel(application: Application) : AndroidViewModel(application) {

    data class ChatMessage(
        val id: Long,
        val role: Role,
        val text: String
    )

    enum class Role { USER, ASSISTANT, SYSTEM }

    data class UiState(
        val messages: List<ChatMessage> = emptyList(),
        val isLoading: Boolean = false,
        val isModelLoaded: Boolean = false,
        val isGenerating: Boolean = false,
        val modelName: String = "",
        val backendName: String = "",
        val tokPerSec: Double = 0.0,
        val error: String? = null
    )

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state

    private var engine: CellmEngine? = null
    private var tokenizer: CellmTokenizer? = null
    private var session: CellmSession? = null
    private var messageIdCounter = 0L

    /**
     * Load a model from local files and initialize the engine.
     */
    fun loadModel(modelPath: String, tokenizerPath: String) {
        viewModelScope.launch {
            _state.value = _state.value.copy(isLoading = true, error = null)

            try {
                withContext(Dispatchers.IO) {
                    val eng = CellmEngine.create(
                        modelPath = modelPath,
                        tokensPerBlock = 16,
                        totalBlocks = 128,
                        topK = 40,
                        temperature = 0.8f
                    )
                    val tok = CellmTokenizer.load(tokenizerPath)

                    engine = eng
                    tokenizer = tok
                    session = eng.createSession()
                }

                _state.value = _state.value.copy(
                    isLoading = false,
                    isModelLoaded = true,
                    modelName = modelPath.substringAfterLast("/"),
                    backendName = engine?.getBackendName() ?: "unknown",
                    messages = listOf(
                        ChatMessage(messageIdCounter++, Role.SYSTEM, "Model loaded. Start a conversation.")
                    )
                )
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    isLoading = false,
                    error = "Failed to load model: ${e.message}"
                )
            }
        }
    }

    /**
     * Send a user message and generate an assistant response.
     */
    fun sendMessage(text: String) {
        val tok = tokenizer ?: return
        val eng = engine ?: return
        val ses = session ?: return

        if (text.isBlank()) return

        val userMessage = ChatMessage(messageIdCounter++, Role.USER, text)
        val messages = _state.value.messages + userMessage
        _state.value = _state.value.copy(
            messages = messages,
            isGenerating = true
        )

        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val tokens = tok.encode(text).toList().toIntArray()
                    ses.submitTokens(tokens)

                    eng.resetStatsWindow()

                    val assistantText = StringBuilder()
                    while (true) {
                        val result = eng.stepDecode() ?: break
                        val (_, tokenId) = result
                        val piece = tok.decodeOne(tokenId)

                        if (piece.isEmpty()) continue

                        // Check for stop tokens (EOS for common models).
                        if (tokenId == 151645 ||     // <|im_end|> for Qwen
                            tokenId == 1 ||           // </s> for Gemma/Llama
                            tokenId == 151643         // <|endoftext|>
                        ) break

                        assistantText.append(piece)

                        // Emit partial response.
                        val partialMessage = ChatMessage(
                            id = messageIdCounter,
                            role = Role.ASSISTANT,
                            text = assistantText.toString()
                        )
                        val currentMessages = _state.value.messages
                        val updated = currentMessages.filter { it.id != messageIdCounter } + partialMessage
                        _state.value = _state.value.copy(messages = updated)
                    }

                    val tokPerSec = eng.getTokPerSec()
                    _state.value = _state.value.copy(
                        tokPerSec = tokPerSec,
                        isGenerating = false
                    )
                    messageIdCounter++
                }
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    isGenerating = false,
                    error = "Generation error: ${e.message}"
                )
            }
        }
    }

    /**
     * Reset the current session (clears context).
     */
    fun resetSession() {
        session?.reset()
        _state.value = _state.value.copy(
            messages = emptyList(),
            tokPerSec = 0.0
        )
        messageIdCounter = 0
    }

    fun clearError() {
        _state.value = _state.value.copy(error = null)
    }

    override fun onCleared() {
        super.onCleared()
        engine?.close()
        tokenizer?.close()
    }
}
