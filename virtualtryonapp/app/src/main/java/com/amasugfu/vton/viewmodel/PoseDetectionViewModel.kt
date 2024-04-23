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
import androidx.lifecycle.viewModelScope
import com.amasugfu.vton.data.PoseDetectionSession
import com.amasugfu.vton.data.domain.GetAssetsUseCase
import com.amasugfu.vton.data.repo.IGarmentRepository
import com.amasugfu.vton.data.repo.IResourceRetriever
import com.amasugfu.vton.data.repo.RemotePoseReconstruction
import com.amasugfu.vton.view.google.GraphicOverlay
import com.amasugfu.vton.view.google.PoseGraphic
import com.google.android.filament.utils.Float3
import com.google.android.filament.utils.RotationsOrder
import com.google.android.filament.utils.rotation
import com.google.mlkit.vision.pose.Pose
import com.google.protobuf.ByteString
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import javax.inject.Inject

@HiltViewModel
class PoseDetectionViewModel @Inject constructor(
    val poseDetectionSession: PoseDetectionSession,
    val remotePoseReconstruction: RemotePoseReconstruction,
    val garmentRepository: IGarmentRepository,
    getAssetsUseCase: GetAssetsUseCase
) : Model3DViewModel(
    object : IResourceRetriever<ByteBuffer> {
        override suspend fun postRetrievalRequest(data: Any?): ByteBuffer {
            return garmentRepository["final"]!!
        }
    },
    getAssetsUseCase,
) {

    lateinit var cameraController: LifecycleCameraController
    var cameraSelector: CameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

//    val pose: MutableState<Pose?> = mutableStateOf(null)
//    var drawPose: MutableState<Boolean> = mutableStateOf(true)
    val rigidMovement: MutableState<Boolean> = mutableStateOf(true)

    companion object {
        val JOINTS_MAP = hashMapOf(
            "Pelvis" to 0,
            "L_Hip" to 1,
            "L_Knee" to 4,
            "L_Ankle" to 7,
            "L_Foot" to 10,
            "R_Hip" to 2,
            "R_Knee" to 5,
            "R_Ankle" to 8,
            "R_Foot" to 11,
            "Spine1" to 3,
            "Spine2" to 6,
            "Spine3" to 9,
            "Neck" to 12,
            "Head" to 15,
            "L_Collar" to 13,
            "L_Shoulder" to 16,
            "L_Elbow" to 18,
            "L_Wrist" to 20,
            "L_Hand" to 22,
            "R_Collar" to 14,
            "R_Shoulder" to 17,
            "R_Elbow" to 19,
            "R_Wrist" to 21,
            "R_Hand" to 23,
        )
    }

    init {
        allowMove = false
    }

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

                    try {
                        val pose = remotePoseReconstruction.postRetrievalRequest(request)
                        transformByPose(pose)
                    }
                    catch (e: Exception) {
                        alertSessionEnded()
                    }
                }.invokeOnCompletion {
                    imageProxy.close()
                }
            }
        }


        poseDetectionSession.startSession()
    }

    fun transformByPose(pose: FloatArray) {
        modelViewer.asset?.also { asset ->
            val transformManager = modelViewer.engine.transformManager
            for (i in 0 until 75 step 3) {
                rotation(pose[i], pose[i+1], pose[i+2])
            }

            transformManager.getInstance(asset.getFirstEntityByName("Pelvis")).also { entity ->
                // get transformation matrices

                val transform = FloatArray(16).also { arr -> transformManager.getTransform(entity, arr) }
                // blender coordinates system swapped Y,Z
                val R = rotation(Float3(-pose[3], pose[5], -pose[4]), RotationsOrder.ZYX).times(1.2f).toFloatArray()

                // flip X
                R[0] = -R[0]
                R[1] = -R[1]
                R[2] = -R[2]
                // flip Y
                R[4] = -R[4]
                R[5] = -R[5]
                R[6] = -R[6]
                // flip Z
                R[8] = -R[8]
                R[9] = -R[9]
                R[10] = -R[10]

                // add camera translation
                R[12] = pose[0]
                R[13] = -pose[1]
                R[14] = -pose[2] + 1f

                transformManager.setTransform(entity, R)
            }


            modelViewer.animator?.updateBoneMatrices()
        }
    }

    fun alertSessionEnded() {

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