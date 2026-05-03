package com.cellm.demo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.*
import androidx.compose.runtime.*

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                var modelPath by remember { mutableStateOf<String?>(null) }
                var tokenizerPath by remember { mutableStateOf<String?>(null) }

                if (modelPath == null) {
                    com.cellm.sdk.ModelPickerScreen { mp, tp ->
                        modelPath = mp
                        tokenizerPath = tp
                    }
                } else {
                    MainTabs(
                        modelPath = modelPath!!,
                        tokenizerPath = tokenizerPath!!,
                        onBackToModels = {
                            modelPath = null
                            tokenizerPath = null
                        }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainTabs(
    modelPath: String,
    tokenizerPath: String,
    onBackToModels: () -> Unit
) {
    var selectedTab by remember { mutableIntStateOf(0) }

    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    icon = { Icon(Icons.Filled.Chat, contentDescription = null) },
                    label = { Text("Chat") },
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 }
                )
                NavigationBarItem(
                    icon = { Icon(Icons.Filled.Image, contentDescription = null) },
                    label = { Text("VLM") },
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 }
                )
                NavigationBarItem(
                    icon = { Icon(Icons.Filled.Storage, contentDescription = null) },
                    label = { Text("Models") },
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 }
                )
            }
        }
    ) { padding ->
        when (selectedTab) {
            0 -> com.cellm.sdk.ChatScreen(
                modelPath = modelPath,
                tokenizerPath = tokenizerPath,
                onBack = onBackToModels
            )
            1 -> com.cellm.sdk.VlmScreen(
                modelPath = modelPath,
                tokenizerPath = tokenizerPath,
                onBack = onBackToModels
            )
            2 -> com.cellm.sdk.ModelPickerScreen(
                onModelSelected = { _, _ -> onBackToModels() }
            )
        }
    }
}
