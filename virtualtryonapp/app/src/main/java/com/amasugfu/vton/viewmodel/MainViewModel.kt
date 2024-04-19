package com.amasugfu.vton.viewmodel

import android.net.Uri
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.amasugfu.vton.data.repo.GarmentData
import com.amasugfu.vton.data.repo.IGarmentRepository
import com.amasugfu.vton.data.repo.ResourceType
import com.amasugfu.vton.view.NavigationController
import com.amasugfu.vton.view.theme.GarmentAlignmentNamespace
import com.amasugfu.vton.view.theme.LiveCameraNamespace
import com.amasugfu.vton.view.theme.PhotoPickerNamespace
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class MainViewModel @Inject constructor(
    private val navigationController: NavigationController,
    private val garmentRepository: IGarmentRepository
) : ViewModel() {

    enum class InputMode {
        UPLOAD_IMAGE,
        UPLOAD_MODEL,
        CAMERA,
    }

    val inputMode: MutableState<InputMode> = mutableStateOf(InputMode.UPLOAD_IMAGE)
    val resourceUri: MutableLiveData<Uri> by lazy { MutableLiveData() }

    fun openImageSelection() {
        navigationController.navigateTo(PhotoPickerNamespace)
    }

    fun openModelSelection() {

    }

    fun openCamera() {
        navigationController.navigateTo(LiveCameraNamespace)
    }

    fun onInputValueChanged() {
        when (inputMode.value) {
            InputMode.UPLOAD_IMAGE, InputMode.CAMERA -> startImageAlignment()
            InputMode.UPLOAD_MODEL -> startVTON()
        }
    }

    fun startImageAlignment() {
        garmentRepository.pushDataTo(
            GarmentData(resourceUri.value!!, ResourceType.IMAGE), GarmentAlignmentViewModel::class.java
        )
        navigationController.navigateTo(GarmentAlignmentNamespace)
    }

    fun startVTON() {

    }

    fun setResourceUri(uri: Uri) {
        resourceUri.value = uri
    }
}