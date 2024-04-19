package com.amasugfu.vton.view.activity

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.amasugfu.vton.data.domain.navigation.NavigateToActivity
import com.amasugfu.vton.data.domain.navigation.NavigateToCameraActivity
import com.amasugfu.vton.data.domain.navigation.NavigateToImageSelectionActivity
import com.amasugfu.vton.view.AppBar
import com.amasugfu.vton.view.MainScreen
import com.amasugfu.vton.view.NavigationController
import com.amasugfu.vton.view.theme.*
import com.amasugfu.vton.viewmodel.MainViewModel
import com.google.android.filament.utils.Utils
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    // init for filament 3D rendering
    companion object {
        init {
            Utils.init()
        }
    }

    @Inject lateinit var navigationController: NavigationController
    val mainViewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // register navigation delegate
        registerSettingsNavigation()
        registerPhotoPickerNavigation()
        registerCameraNavigation()
        registerPoseDetectionNavigation()
        registerGarmentAlignmentNavigation()

        // register observer for uploads
        mainViewModel.resourceUri.observe(this) { mainViewModel.onInputValueChanged() }

        setContent {
            VirtualtryonappTheme {
                // A surface container using the 'background' color from the theme
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    AppBar(navigationController) {
                        MainScreen()
                    }
                }
            }
        }
    }

    private fun registerSettingsNavigation() {
        navigationController.registerNavigation(SettingsNamespace, NavigateToActivity(SettingsActivity::class.java, this))
    }

    private fun setMainViewModelInput(uri: Any?) {
        if (uri is Uri) {
            mainViewModel.setResourceUri(uri)
        }
    }

    private fun registerPhotoPickerNavigation() {
        navigationController.registerNavigation(PhotoPickerNamespace, NavigateToImageSelectionActivity(this, ::setMainViewModelInput))
    }

    private fun registerCameraNavigation() {
        navigationController.registerNavigation(LiveCameraNamespace, NavigateToCameraActivity(this, ::setMainViewModelInput))
    }

    private fun registerPoseDetectionNavigation() {
        navigationController.registerNavigation(PoseDetectionNamespace, NavigateToActivity(PoseDetectionActivity::class.java, this))
    }

    private fun registerGarmentAlignmentNavigation() {
        navigationController.registerNavigation(GarmentAlignmentNamespace, NavigateToActivity(GarmentAlignmentActivity::class.java, this))
    }
}
