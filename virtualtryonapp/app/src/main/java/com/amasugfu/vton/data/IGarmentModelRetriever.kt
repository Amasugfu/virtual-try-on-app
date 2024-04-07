package com.amasugfu.vton.data

import android.content.res.AssetManager
import java.nio.ByteBuffer
import javax.inject.Inject

interface IGarmentModelRetriever {
    fun postModelRetrieval(): ByteBuffer
}

class RemoteGarmentModelRetriever : IGarmentModelRetriever {
    override fun postModelRetrieval(): ByteBuffer {
        TODO("Not yet implemented")
    }
}

class DebugLocalGarmentModelRetriever @Inject constructor(
    val assetManager: AssetManager
) : IGarmentModelRetriever {
    override fun postModelRetrieval(): ByteBuffer {
        val input = assetManager.open("debug/sample.glb")
        val bytes = ByteArray(input.available())
        input.read(bytes)
        return ByteBuffer.wrap(bytes)
    }
}
