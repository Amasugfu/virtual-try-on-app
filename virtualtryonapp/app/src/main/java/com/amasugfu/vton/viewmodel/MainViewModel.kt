package com.amasugfu.vton.viewmodel

import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.amasugfu.vton.view.NavigationController
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class MainViewModel @Inject constructor(
    private val navigationController: NavigationController
) : ViewModel() {

    enum class InputMode {
        UPLOAD_IMAGE,
        UPLOAD_MODEL,
        CAMERA,
    }

    val inputMode: MutableState<InputMode> = mutableStateOf(InputMode.UPLOAD_IMAGE)
    val inputValue: MutableLiveData<Any> by lazy { MutableLiveData<Any>() }

    fun openImageSelection() {
        navigationController.navigateTo("PhotoPicker")
    }

    fun openModelSelection() {

    }

    fun openCamera() {
        navigationController.navigateTo("Camera")
    }

    fun onInputValueChanged() {
        when (inputMode.value) {
            InputMode.UPLOAD_IMAGE, InputMode.CAMERA -> startImageAlignment()
            InputMode.UPLOAD_MODEL -> startVTON()
        }
    }

    fun startImageAlignment() {
        // TODO: add alignment window

        navigationController.navigateTo("PoseDetection")
    }

    fun startVTON() {

    }
}