package com.amasugfu.vton.view.activity

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.amasugfu.vton.view.MainScreen
import com.amasugfu.vton.view.theme.VirtualtryonappTheme
import com.amasugfu.vton.viewmodel.MainViewModel
import com.amasugfu.vton.view.NavigationController
import com.amasugfu.vton.data.domain.navigation.NavigateToCameraActivity
import com.amasugfu.vton.data.domain.navigation.NavigateToImageSelectionActivity
import com.amasugfu.vton.data.domain.navigation.NavigateToPoseDetectionActivity
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    @Inject lateinit var navigationController: NavigationController
    val mainViewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // register navigation delegate
        registerPhotoPickerNavigation()
        registerCameraNavigation()
        registerPoseDetectionNavigation()

        // register observer for uploads
        mainViewModel.inputValue.observe(this) { mainViewModel.onInputValueChanged() }

        setContent {
            VirtualtryonappTheme {
                // A surface container using the 'background' color from the theme
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    MainScreen()
                }
            }
        }
    }

    private fun setMainViewModelInput(input: Any?) {
        if (input != null) {
            mainViewModel.inputValue.value = input
        }
    }

    private fun registerPhotoPickerNavigation() {
        navigationController.registerNavigation("PhotoPicker", NavigateToImageSelectionActivity(this, ::setMainViewModelInput))
    }

    private fun registerCameraNavigation() {
        navigationController.registerNavigation("Camera", NavigateToCameraActivity(this, ::setMainViewModelInput))
    }

    private fun registerPoseDetectionNavigation() {
        navigationController.registerNavigation("PoseDetection", NavigateToPoseDetectionActivity(this))
    }
}
