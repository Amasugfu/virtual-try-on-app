package com.amasugfu.vton.view.activity

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.amasugfu.vton.view.AppBar
import com.amasugfu.vton.view.NavigationController
import com.amasugfu.vton.view.SettingScreen
import com.amasugfu.vton.view.theme.SettingsNamespace
import com.amasugfu.vton.view.theme.VirtualtryonappTheme
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class SettingsActivity : ComponentActivity() {

    @Inject lateinit var navigationController: NavigationController

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            VirtualtryonappTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    AppBar(
                        navigationController,
                        supplementaryTitle = SettingsNamespace,
                        actionsVisible = false
                    ) {
                        SettingScreen()
                    }
                }
            }
        }
    }
}