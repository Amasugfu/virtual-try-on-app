package com.amasugfu.vton.core.activity

import android.os.Bundle
import androidx.activity.ComponentActivity
import com.amasugfu.vton.core.pose_detection.IPoseDetectionSession

class PoseDetectionActivity(val poseDetectionSession: IPoseDetectionSession) : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        poseDetectionSession.startSession()
    }
}