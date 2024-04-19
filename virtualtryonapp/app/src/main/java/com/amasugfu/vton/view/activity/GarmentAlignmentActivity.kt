package com.amasugfu.vton.view.activity

import android.os.Bundle
import android.view.MotionEvent
import androidx.activity.ComponentActivity
import androidx.activity.addCallback
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.amasugfu.vton.view.GarmentAlignmentScreen
import com.amasugfu.vton.view.theme.VirtualtryonappTheme
import com.amasugfu.vton.viewmodel.GarmentAlignmentViewModel
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class GarmentAlignmentActivity : ComponentActivity() {

    val viewModel: GarmentAlignmentViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        onBackPressedDispatcher.addCallback {
            var poseSettingsOpened by viewModel.poseSettingsOpened
            val loading by viewModel.loading

            if (loading) {
                viewModel.cancelLoading()
            }
            else if (poseSettingsOpened) {
                poseSettingsOpened = false
            }
            else {
                finish()
            }
        }

        setContent {
            VirtualtryonappTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    GarmentAlignmentScreen()
                }
            }
        }
    }

    override fun dispatchTouchEvent(ev: MotionEvent?): Boolean = if (viewModel.loading.value) true else super.dispatchTouchEvent(ev)
}