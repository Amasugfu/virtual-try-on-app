package com.amasugfu.vton.view

import android.view.SurfaceView
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import com.amasugfu.vton.viewmodel.GarmentIOverlay3DViewModel

@Composable
fun GarmentView3DOverlay() {
    val context = LocalContext.current
    val viewModel: GarmentIOverlay3DViewModel = viewModel()

    AndroidView(
        factory = {
            SurfaceView(context).also { viewModel.bindSurface(it) }.also { viewModel.displayScene(it) }
        }
    )
}