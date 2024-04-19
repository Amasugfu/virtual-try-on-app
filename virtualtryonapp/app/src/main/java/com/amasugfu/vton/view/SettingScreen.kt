package com.amasugfu.vton.view

import android.content.Context
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.selection.LocalTextSelectionColors
import androidx.compose.foundation.text.selection.TextSelectionColors
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.TextUnitType
import androidx.compose.ui.unit.dp
import com.amasugfu.vton.data.domain.GetSharedPreferenceUseCase
import com.amasugfu.vton.view.theme.ConnectionSettingsGroupName
import com.amasugfu.vton.view.theme.ConnectionSettingsHostKey
import com.amasugfu.vton.view.theme.ConnectionSettingsHostName
import com.amasugfu.vton.view.theme.ConnectionSettingsPortKey

@Composable
fun SettingScreen() {
    val context = LocalContext.current
    val focusManager = LocalFocusManager.current

    val preferences = context.getSharedPreferences("vton3d_settings", Context.MODE_PRIVATE)
    val editor = preferences.edit()
    val host = GetSharedPreferenceUseCase.getHost(preferences)
    val port = GetSharedPreferenceUseCase.getPort(preferences)

    Column(
        modifier = Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                detectTapGestures {
                    focusManager.clearFocus()
                }
            }
            .verticalScroll(rememberScrollState())
            .padding(5.dp)
            .padding(top = 10.dp)
    ) {
        PreferenceGroupBox(
            ConnectionSettingsGroupName,
        ) {
            SettingBlock(ConnectionSettingsHostName, host) {
                editor.putString(ConnectionSettingsHostKey, it)
                editor.apply()
            }
            SettingBlock("Port", port.toString()) {
                editor.putInt(ConnectionSettingsPortKey, it.toInt())
                editor.apply()
            }
        }
    }
}

@Composable
fun PreferenceGroupBox(
    title: String,
    content: @Composable () -> Unit
) {
    Column(
        modifier = Modifier.padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            title,
            fontSize = TextUnit(4.5f, TextUnitType.Em),
            fontWeight = FontWeight.ExtraBold,
            color = MaterialTheme.colorScheme.onSurface,
        )

        Surface(
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(10),
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                content()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingBlock(
    title: String,
    value: String,
    onChanged: (String) -> Unit = {}
) {
    val state = remember { mutableStateOf(value) }

    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier
            .padding(bottom = 5.dp)
    ) {

        Row(
            verticalAlignment = Alignment.Bottom
        ) {
            Text(
                title,
                fontWeight = FontWeight.Bold,
                fontSize = TextUnit(4.5f, TextUnitType.Em),
                color = MaterialTheme.colorScheme.onPrimary
            )

            Spacer(modifier = Modifier.weight(1f))

            CompositionLocalProvider(
                LocalTextSelectionColors provides TextSelectionColors(
                    handleColor = MaterialTheme.colorScheme.secondary,
                    backgroundColor = Color.White
                )
            ) {
                BasicTextField (
                    state.value,
                    onValueChange = {
                        state.value = it
                        onChanged(it)
                    },
                    maxLines = 1,
                    textStyle = LocalTextStyle.current.copy(
                        textAlign = TextAlign.End,
                        color = MaterialTheme.colorScheme.onSurface
                    ),
                    cursorBrush = SolidColor(MaterialTheme.colorScheme.secondary)
                )
            }
        }

        Divider(
            color = MaterialTheme.colorScheme.onPrimary
        )
    }
}