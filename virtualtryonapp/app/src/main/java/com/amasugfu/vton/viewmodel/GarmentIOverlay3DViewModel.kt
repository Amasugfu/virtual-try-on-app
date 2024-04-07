package com.amasugfu.vton.viewmodel

import android.graphics.Color
import android.graphics.PixelFormat
import android.view.Choreographer
import android.view.SurfaceView
import androidx.lifecycle.ViewModel
import com.amasugfu.vton.data.IGarmentModelRetriever
import com.amasugfu.vton.data.domain.GetAssetsUseCase
import com.google.android.filament.Skybox
import com.google.android.filament.View
import com.google.android.filament.utils.KTX1Loader
import com.google.android.filament.utils.ModelViewer
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class GarmentIOverlay3DViewModel @Inject constructor(
    val garmentModelRetriever: IGarmentModelRetriever,
    val getAssetsUseCase: GetAssetsUseCase
) : ViewModel() {

    private lateinit var choreographer: Choreographer
    private lateinit var modelViewer: ModelViewer

    private val frameCallback = object : Choreographer.FrameCallback {
        override fun doFrame(currentTime: Long) {
            choreographer.postFrameCallback(this)
            modelViewer.render(currentTime)
        }
    }

    fun bindSurface(surfaceView: SurfaceView) {
        choreographer = Choreographer.getInstance()
        modelViewer = ModelViewer(surfaceView)
        surfaceView.setOnTouchListener(modelViewer)
    }

    fun postFrameCallback() {
        if (this::choreographer.isInitialized)
            choreographer.postFrameCallback(frameCallback)
    }

    fun removeFrameCallback() {
        if (this::choreographer.isInitialized)
            choreographer.removeFrameCallback(frameCallback)
    }

    fun loadGlb() {
        val buffer = garmentModelRetriever.postModelRetrieval()
        modelViewer.loadModelGlb(buffer)
        modelViewer.transformToUnitCube()
    }

    fun displayScene(
        surfaceView: SurfaceView,
        transparent: Boolean = true
    ) {
        loadGlb()

        var buffer = getAssetsUseCase.execute("ibl/lightroom_14b_ibl.ktx")
        KTX1Loader.createIndirectLight(modelViewer.engine, buffer).apply {
            intensity = 50_000f
            modelViewer.scene.indirectLight = this
        }

        if (transparent) {
            surfaceView.setZOrderOnTop(true)
            surfaceView.setBackgroundColor(Color.TRANSPARENT)
            surfaceView.getHolder().setFormat(PixelFormat.TRANSLUCENT)

            modelViewer.view.blendMode = View.BlendMode.TRANSLUCENT
            modelViewer.scene.skybox = null
            val options = modelViewer.renderer.clearOptions
            options.clear = true
            modelViewer.renderer.clearOptions = options
        }
        else {
            modelViewer.scene.skybox = Skybox.Builder()
                .build(modelViewer.engine)
        }

        postFrameCallback()
    }
}