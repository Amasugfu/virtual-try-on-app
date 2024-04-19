package com.amasugfu.vton.viewmodel

import Requests
import android.content.Context
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.amasugfu.vton.data.PoseDetectionSession
import com.amasugfu.vton.data.repo.RemotePoseReconstruction
import com.amasugfu.vton.view.google.GraphicOverlay
import com.amasugfu.vton.view.google.PoseGraphic
import com.google.mlkit.vision.pose.Pose
import com.google.protobuf.ByteString
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class PoseDetectionViewModel @Inject constructor(
    val poseDetectionSession: PoseDetectionSession,
    val remotePoseReconstruction: RemotePoseReconstruction
) : ViewModel() {

    lateinit var cameraController: LifecycleCameraController
    var cameraSelector: CameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

    val pose: MutableState<Pose?> = mutableStateOf(null)
    var drawPose: MutableState<Boolean> = mutableStateOf(true)

    fun bindCamera(
        context: Context,
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        graphicOverlay: GraphicOverlay
    ) {
        cameraController = LifecycleCameraController(context)
        cameraController.bindToLifecycle(lifecycleOwner)
        previewView.controller = cameraController

        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
            .build()
        cameraController.previewResolutionSelector = resolutionSelector
        cameraController.imageAnalysisResolutionSelector = resolutionSelector
        // ensure the image is processed one by one / synchronized
        cameraController.imageAnalysisBackpressureStrategy = ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
        cameraController.setImageAnalysisAnalyzer(
            ContextCompat.getMainExecutor(context),
            poseDetectionSession
        )
        cameraController.imageAnalysisOutputImageFormat = ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888


        cameraController.cameraSelector = cameraSelector

//        poseDetectionSession.setPoseDetectedListener { pose, image ->
//            this.pose.value = pose
//
//            // display detected landmarks
//            if (drawPose.value) {
//                graphicOverlay.setImageSourceInfo(
//                    image.height,
//                    image.width,
//                    cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA)
//                drawPoseOverlay(pose, graphicOverlay)
//            }
//        }


        poseDetectionSession.setMediaImageListener { imageProxy ->
            @OptIn(ExperimentalGetImage::class)
            val image = imageProxy.image
            if (image != null) {
                viewModelScope.launch {
                    val bytes = image.planes[0].buffer

                    val request = Requests.ByteBuffer.newBuilder()
                        .addBuffer(ByteString.copyFrom(bytes))
                        .build()

                    val transformations = remotePoseReconstruction.postRetrievalRequest(request)

                    transformations

                }.invokeOnCompletion {
                    imageProxy.close()
                }
            }
        }


        poseDetectionSession.startSession()
    }

    fun captureHumanShape() {

    }

    private fun drawPoseOverlay(pose: Pose, graphicOverlay: GraphicOverlay) {
        graphicOverlay.clear()
        graphicOverlay.add(
            PoseGraphic(
                graphicOverlay,
                pose,
                showInFrameLikelihood = true,
                visualizeZ = true,
                rescaleZForVisualization = true
            )
        )
        graphicOverlay.postInvalidate()
    }

    fun flipCamera() {
        cameraSelector =
            if (cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA)
                CameraSelector.DEFAULT_BACK_CAMERA
            else
                CameraSelector.DEFAULT_FRONT_CAMERA
        cameraController.cameraSelector = cameraSelector
    }
}