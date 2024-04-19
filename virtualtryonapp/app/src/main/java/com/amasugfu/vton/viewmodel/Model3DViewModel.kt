package com.amasugfu.vton.viewmodel

import android.annotation.SuppressLint
import android.graphics.Color
import android.graphics.PixelFormat
import android.view.Choreographer
import android.view.SurfaceView
import androidx.lifecycle.ViewModel
import com.amasugfu.vton.data.domain.GetAssetsUseCase
import com.amasugfu.vton.data.repo.IResourceRetriever
import com.amasugfu.vton.view.google.ModelViewer
import com.google.android.filament.Skybox
import com.google.android.filament.View
import com.google.android.filament.utils.KTX1Loader
import kotlinx.coroutines.runBlocking
import java.nio.ByteBuffer

open class Model3DViewModel(
    protected val modelRetriever: IResourceRetriever<ByteBuffer>,
    protected val getAssetsUseCase: GetAssetsUseCase
) : ViewModel() {

    protected lateinit var choreographer: Choreographer
    protected lateinit var modelViewer: ModelViewer
    var allowOrbit = true

    protected open val preRenderTask: () -> Unit = {}

    protected val frameCallback = object : Choreographer.FrameCallback {
        override fun doFrame(currentTime: Long) {
            choreographer.postFrameCallback(this)
            preRenderTask()
            modelViewer.render(currentTime)
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    open fun bindSurface(surfaceView: SurfaceView) {
        choreographer = Choreographer.getInstance()
        modelViewer = ModelViewer(surfaceView)
        surfaceView.setOnTouchListener { v, event ->
            if (allowOrbit) modelViewer.onTouchEvent(event)
            true
        }
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
        runBlocking {
            val buffer = modelRetriever.postRetrievalRequest(null)
            replaceAsset(buffer)
        }
    }

    fun replaceAsset(buffer: ByteBuffer) {
        modelViewer.loadModelGlb(buffer)
        modelViewer.transformToUnitCube()
    }

    open fun displayScene(
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
                .color(0.5f, 0.5f, 0.5f, 1f)
                .build(modelViewer.engine)
        }

        postFrameCallback()
    }
}