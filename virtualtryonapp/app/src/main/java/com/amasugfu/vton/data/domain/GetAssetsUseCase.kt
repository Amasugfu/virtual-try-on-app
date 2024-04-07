package com.amasugfu.vton.data.domain

import android.content.res.AssetManager
import java.nio.ByteBuffer
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class GetAssetsUseCase @Inject constructor(
    val assetManager: AssetManager
) {
    fun execute(path: String): ByteBuffer {
        val input = assetManager.open(path)
        val bytes = ByteArray(input.available())
        input.read(bytes)
        return ByteBuffer.wrap(bytes)
    }
}