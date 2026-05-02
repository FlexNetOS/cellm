package com.cellm.sdk

import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Vision Language Model (VLM) screen for image-to-text inference.
 *
 * Lets the user pick an image from the gallery, enter a prompt, and
 * generate a description using a VLM model loaded via CellmEngine.
 *
 * Requires a VLM-compatible model (SmolVLM-256M or similar) and
 * the cellm engine with multimodal support.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VlmScreen(
    modelPath: String,
    tokenizerPath: String,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var engine by remember { mutableStateOf<CellmEngine?>(null) }
    var session by remember { mutableStateOf<CellmSession?>(null) }
    var isModelLoaded by remember { mutableStateOf(false) }

    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var imageBytes by remember { mutableStateOf<ByteArray?>(null) }
    var prompt by remember { mutableStateOf("Describe this image.") }
    var output by remember { mutableStateOf("") }
    var isRunning by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var timingMs by remember { mutableStateOf(0L) }

    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        imageUri = uri
        uri?.let {
            val input = context.contentResolver.openInputStream(it)
            val rawBytes = input?.readBytes()
            input?.close()
            if (rawBytes != null) {
                val bitmap = BitmapFactory.decodeByteArray(rawBytes, 0, rawBytes.size)
                if (bitmap != null) {
                    imageBytes = CellmVLM.bitmapToJpegBytes(bitmap)
                }
            }
        }
    }

    // Load model on first composition.
    LaunchedEffect(modelPath) {
        withContext(Dispatchers.IO) {
            try {
                val eng = CellmEngine.create(
                    modelPath = modelPath,
                    tokensPerBlock = 16,
                    totalBlocks = 128,
                    topK = 40,
                    temperature = 0.8f
                )
                engine = eng
                session = eng.createSession()
                isModelLoaded = true
            } catch (e: Exception) {
                error = "Failed to load model: ${e.message}"
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("cellm VLM") },
                navigationIcon = {
                    TextButton(onClick = onBack) { Text("Back") }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Image picker area.
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    if (imageUri != null && imageBytes != null) {
                        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes!!.size)
                        if (bitmap != null) {
                            Image(
                                bitmap = bitmap.asImageBitmap(),
                                contentDescription = "Selected image",
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = 240.dp)
                                    .clip(RoundedCornerShape(8.dp)),
                                contentScale = ContentScale.Fit
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                        }
                    }

                    Button(onClick = { imagePicker.launch("image/*") }) {
                        Text(if (imageBytes == null) "Pick Image" else "Change Image")
                    }
                }
            }

            // Prompt input.
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Prompt", fontWeight = FontWeight.SemiBold)
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedTextField(
                        value = prompt,
                        onValueChange = { prompt = it },
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 100.dp),
                        enabled = !isRunning
                    )
                }
            }

            // Run button.
            Button(
                onClick = {
                    scope.launch {
                        isRunning = true
                        error = null
                        output = ""

                        val start = System.currentTimeMillis()
                        try {
                            withContext(Dispatchers.IO) {
                                val eng = engine ?: throw Exception("Engine not loaded")
                                val ses = session ?: throw Exception("Session not created")
                                val result = eng.describeImage(
                                    ses,
                                    imageBytes!!,
                                    prompt
                                )
                                output = result
                            }
                        } catch (e: Exception) {
                            error = e.message
                        }
                        timingMs = System.currentTimeMillis() - start
                        isRunning = false
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = isModelLoaded && imageBytes != null && !isRunning
            ) {
                Text(if (isRunning) "Running..." else "Run VLM")
            }

            // Timing.
            if (timingMs > 0) {
                Text(
                    text = "Completed in ${timingMs}ms",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // Error.
            if (error != null) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Text(
                        text = error!!,
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.onErrorContainer
                    )
                }
            }

            // Output.
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Output", fontWeight = FontWeight.SemiBold)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = output.ifEmpty { "No output yet." },
                        style = MaterialTheme.typography.bodyMedium,
                        color = if (output.isEmpty())
                            MaterialTheme.colorScheme.onSurfaceVariant
                        else
                            MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }
    }
}
