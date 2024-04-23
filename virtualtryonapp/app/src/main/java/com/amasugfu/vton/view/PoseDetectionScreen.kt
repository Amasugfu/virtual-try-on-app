package com.amasugfu.vton.view

import android.content.Context
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.amasugfu.vton.R
import com.amasugfu.vton.view.google.GraphicOverlay
import com.amasugfu.vton.viewmodel.PoseDetectionViewModel

@Composable
fun PoseDetectionScreen() {
    val viewModel: PoseDetectionViewModel = viewModel()

    val context: Context = LocalContext.current
    val lifecycleOwner: LifecycleOwner = LocalLifecycleOwner.current

    val graphicOverlay = GraphicOverlay(context, null)

    Box {
        // camera preview
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

//        // graphic overlay
//        AndroidView(
//            modifier = Modifier.fillMaxSize(),
//            factory = {
//                graphicOverlay
//            }
//        )

        Model3DView(viewModel, transparent = true)

        // control buttons
        Column(
            modifier = Modifier.align(Alignment.BottomEnd).padding(25.dp)
        ) {
            RoundIconButton(viewModel::flipCamera, R.drawable.ic_flip_camera)

            var rigid by viewModel.rigidMovement
            RoundIconButton({ rigid = !rigid }, R.drawable.ic_flip_camera, isToggle = { rigid })
        }
    }
}
