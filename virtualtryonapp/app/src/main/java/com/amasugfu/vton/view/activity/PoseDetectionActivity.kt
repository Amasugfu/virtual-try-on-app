package com.amasugfu.vton.view.activity

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.amasugfu.vton.view.PoseDetectionScreen
import com.amasugfu.vton.view.theme.VirtualtryonappTheme
import com.amasugfu.vton.viewmodel.PoseDetectionViewModel
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class PoseDetectionActivity : ComponentActivity() {

    private val REQUIRED_PERMISSIONS: Array<String> = arrayOf(
        "android.permission.CAMERA"
    )

    private val permissionRequestLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        // Handle Permission granted/rejected
        var permissionGranted = true

        permissions.entries.forEach {
            if (it.key in REQUIRED_PERMISSIONS && !it.value)
                permissionGranted = false
        }

        if (!permissionGranted) {
            Toast.makeText(baseContext,
                "Permission request denied",
                Toast.LENGTH_SHORT).show()

            finish()
        }
    }

    val poseDetectionViewModel: PoseDetectionViewModel by viewModels()

//    val garmentIOverlay3DViewModel: Model3DViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        requestPermissions()

        setContent {
            VirtualtryonappTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    PoseDetectionScreen()
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        poseDetectionViewModel.postFrameCallback()
    }

    override fun onPause() {
        super.onPause()
        poseDetectionViewModel.removeFrameCallback()
    }

    override fun onDestroy() {
        super.onDestroy()
        poseDetectionViewModel.removeFrameCallback()
    }

    private fun requestPermissions() {
        permissionRequestLauncher.launch(REQUIRED_PERMISSIONS)
    }
}