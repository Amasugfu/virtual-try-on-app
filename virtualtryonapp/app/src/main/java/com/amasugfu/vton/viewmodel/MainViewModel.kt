package com.amasugfu.vton.viewmodel

import androidx.camera.core.impl.LiveDataObservable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import kotlin.properties.ObservableProperty

class MainViewModel : ViewModel() {
    enum class InputMode {
        UPLOAD_IMAGE,
        UPLOAD_MODEL,
        CAMERA,
    }

    val inputMode: MutableState<InputMode> = mutableStateOf(InputMode.UPLOAD_IMAGE)

    fun openImageSelection() {

    }

    fun openModelSelection() {

    }

    fun openCamera() {

    }

    fun changeInputMode(inputMode: InputMode) {
        this.inputMode.value = inputMode
    }
}