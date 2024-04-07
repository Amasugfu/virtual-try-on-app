package com.amasugfu.vton.view

import android.content.Context
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.material3.IconButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.amasugfu.vton.view.google.GraphicOverlay
import com.amasugfu.vton.viewmodel.PoseDetectionViewModel

@Composable
fun PoseDetectionScreen() {
    val viewModel: PoseDetectionViewModel = viewModel()

    val context: Context = LocalContext.current
    val lifecycleOwner: LifecycleOwner = LocalLifecycleOwner.current

    val graphicOverlay = GraphicOverlay(context, null)

    Box {
//        Text("Hi")
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = {
                PreviewView(context).apply {
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }.also {
                    viewModel.bindCamera(
                        context,
                        lifecycleOwner,
                        it,
                        graphicOverlay
                    )
                }
            }
        )

        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = {
                graphicOverlay
            }
        )

        GarmentView3DOverlay()

        Column {
            IconButton(
                onClick = { viewModel.flipCamera() },
                modifier = Modifier.size(80.dp),
                content = {

                }
            )
        }
    }
}