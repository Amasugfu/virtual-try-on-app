package com.amasugfu.vton.data.domain.navigation

import android.content.Intent
import androidx.activity.ComponentActivity
import com.amasugfu.vton.view.activity.PoseDetectionActivity

class NavigateToPoseDetectionActivity(
    context: ComponentActivity,
) : NavigateUsingContextUseCase(context) {
    override fun navigate() {
        Intent(context.applicationContext, PoseDetectionActivity::class.java).also { context.startActivity(it) }
    }
}