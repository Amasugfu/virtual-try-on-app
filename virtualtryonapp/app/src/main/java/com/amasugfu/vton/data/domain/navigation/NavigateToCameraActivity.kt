package com.amasugfu.vton.data.domain.navigation

import android.content.ContentValues
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

class NavigateToCameraActivity(
    context: ComponentActivity, onCompleted: (Any?) -> Unit,
): NavigateUsingContextUseCase(context, onCompleted, ) {

    val camera = context.registerForActivityResult(
        ActivityResultContracts.TakePicture(),
        onCompleted
    )

    override fun navigate() {
        val resolver = context.contentResolver
        val collections = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)

        val formatter = DateTimeFormatter.ofPattern("ddMMyyyyHHmmss")
        val date = LocalDateTime.now().format(formatter)

        val newImage = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "vton-$date.png")
        }

        val uri = resolver.insert(collections, newImage)
        camera.launch(uri)
    }
}