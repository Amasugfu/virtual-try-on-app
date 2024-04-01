package com.amasugfu.vton.core.activity

import android.os.Bundle
import androidx.activity.ComponentActivity
import com.amasugfu.vton.core.pose_detection.IPoseDetectionSession
import com.amasugfu.vton.core.pose_detection.PoseDetectionSession

class PoseDetectionActivity : ComponentActivity() {
    private var poseDetectionSession: PoseDetectionSession = PoseDetectionSession()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        poseDetectionSession.startSession()
    }
}