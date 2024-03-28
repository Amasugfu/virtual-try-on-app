package com.amasugfu.vton.core.pose_detection

import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions


interface IPoseDetectionSession {
    fun getDetectedPoses()
    fun startSession()
    fun endSession()
}

class PoseDetectionSession : ImageAnalysis.Analyzer, IPoseDetectionSession {
    var isStarted: Boolean = false
    val options: AccuratePoseDetectorOptions = AccuratePoseDetectorOptions.Builder().setDetectorMode(AccuratePoseDetectorOptions.STREAM_MODE).build()
    val poseDetector: PoseDetector = PoseDetection.getClient(options)

    private lateinit var image: InputImage
    lateinit var pose: Pose
        private set

    override fun getDetectedPoses() {
        var result = poseDetector.process(image)
    }

    override fun startSession() {
        isStarted = true
    }

    override fun endSession() {
        isStarted = false
    }

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
            poseDetector.process(image).onSuccessTask { result -> pose = result }
        }
    }
}