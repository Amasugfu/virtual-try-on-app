package com.amasugfu.vton.data

import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis.Analyzer
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import javax.inject.Inject


interface IPoseDetectionSession {
    fun startSession()
    fun endSession()
    fun setPoseDetectedListener(listener: (Pose, InputImage) -> Unit)
}

class PoseDetectionSession @Inject constructor() : IPoseDetectionSession, Analyzer {

    var isStarted: Boolean = false

    // pose detector config
    private val options: PoseDetectorOptions =
        PoseDetectorOptions.Builder()
            .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
            .setPreferredHardwareConfigs(PoseDetectorOptionsBase.CPU_GPU)
            .build()

    private val poseDetector: PoseDetector = PoseDetection.getClient(options)

    // listener to call when a pose is detected
    private var listener: (Pose, InputImage) -> Unit = { _, _ ->  }

    override fun setPoseDetectedListener(listener: (Pose, InputImage) -> Unit) {
        this.listener = listener
    }

    override fun startSession() {
        isStarted = true
    }

    override fun endSession() {
        isStarted = false
    }

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        if (!isStarted) {
            imageProxy.close()
            return
        }

        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

            poseDetector.process(image)
                .addOnSuccessListener {
                    listener(it, image)
                }
                .addOnFailureListener {
                    Log.d("pose detection", "failed")
                }
                .addOnCompleteListener {
                    imageProxy.close()
                }
        }
    }
}