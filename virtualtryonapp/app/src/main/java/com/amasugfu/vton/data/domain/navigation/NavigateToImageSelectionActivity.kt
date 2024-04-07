package com.amasugfu.vton.data.domain.navigation

import androidx.activity.ComponentActivity
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts

class NavigateToImageSelectionActivity(
    context: ComponentActivity, onCompleted: (Any?) -> Unit,
) : NavigateUsingContextUseCase(context, onCompleted) {

    val photoPicker = context.registerForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
        onCompleted
    )

    override fun navigate() {
        photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }
}