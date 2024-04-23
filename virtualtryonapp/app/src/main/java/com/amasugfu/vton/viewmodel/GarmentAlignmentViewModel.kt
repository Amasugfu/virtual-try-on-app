package com.amasugfu.vton.viewmodel

import android.content.ContentResolver
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.IntSize
import androidx.lifecycle.viewModelScope
import com.amasugfu.vton.data.domain.GetAssetsUseCase
import com.amasugfu.vton.data.repo.IGarmentRepository
import com.amasugfu.vton.data.repo.IResourceRetriever
import com.amasugfu.vton.data.repo.RemoteGarmentReconstruction
import com.amasugfu.vton.data.repo.SMPL
import com.amasugfu.vton.view.NavigationController
import com.amasugfu.vton.view.google.ModelViewer
import com.google.android.filament.utils.*
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import javax.inject.Inject

private fun ModelViewer.reset() {
    cameraManipulator.jumpToBookmark(cameraManipulator.homeBookmark)
    switchGesture(orbit = false)
}

private fun ModelViewer.switchGesture(orbit: Boolean) {
    gestureDetector.swapOrbitPan(orbit)
}

@HiltViewModel
class GarmentAlignmentViewModel @Inject constructor(
    @SMPL smplModelRetriever: IResourceRetriever<ByteBuffer>,
    getAssetsUseCase: GetAssetsUseCase,
    private val navigationController: NavigationController,
    private val garmentRepo: IGarmentRepository
) : Model3DViewModel(smplModelRetriever, getAssetsUseCase) {

    val showGarmentOpaque: MutableState<Boolean> = mutableStateOf(true)
    val showAlert: MutableState<Boolean> = mutableStateOf(false)

    val poseSettingsOpened: MutableState<Boolean> = mutableStateOf(false)
    val poseLeftArm: MutableState<Float> = mutableStateOf(0f)
    val poseRightArm: MutableState<Float> = mutableStateOf(0f)
    val poseLeftLeg: MutableState<Float> = mutableStateOf(0f)
    val poseRightLeg: MutableState<Float> = mutableStateOf(0f)

    val executor: ExecutorService by lazy { Executors.newSingleThreadScheduledExecutor() }

    companion object {
        val rangeLeftArm = -90f..90f
        val rangeRightArm = -90f..90f
        val rangeLeftLeg = -90f..15f
        val rangeRightLeg = -15f..90f
    }

    val loading: MutableState<Boolean> = mutableStateOf(false)
    private lateinit var loadingJob: Job

    // LShoulder, RShoulder, LHip, RHip
    val jointIDs = hashMapOf(
        "L_Shoulder" to poseLeftArm,
        "R_Shoulder" to poseRightArm,
        "L_Hip" to poseLeftLeg,
        "R_Hip" to poseRightLeg
    )

    lateinit var rootTransform: FloatArray

    val imageUri: Uri
    var imageSize: IntSize? = null
    var imagePosition: Offset? = null
    val imageCenter: Offset?
        get() = imageSize?.let { size -> imagePosition?.let { pos -> Offset(pos.x + size.width / 2, pos.y + size.height / 2) } }

    val resultShowCase: MutableState<Boolean> = mutableStateOf(false)

    init {
        // the activity will only be opened after the image is set
        imageUri = garmentRepo.getData(this::class.java)!!.resourceUri
    }

    override val preRenderTask = {
        poseSMPL()
    }

    private fun getImageDataHelper(x: Int, y: Int, d: Float, array: FloatArray, start: Int, onCompleted: () -> Unit = {}) {
        modelViewer.view.pick(x, y, executor) { result ->
            val coords = result.fragCoords
            coords[2] = d

            val imageCorner2Position = computeWorldPosition(coords)
            array[start] = imageCorner2Position.x / imageCorner2Position.w
            array[start + 1] = imageCorner2Position.y / imageCorner2Position.w
            array[start + 2] = imageCorner2Position.z / imageCorner2Position.w

            onCompleted()
        }
    }

    private fun submissionHelper(
        array: FloatArray,
        coords: FloatArray,
        contentResolver: ContentResolver,
        onCompleted: (ByteBuffer) -> Unit
    ) {
        loadingJob = viewModelScope.launch {
            val smplWorldPosition = computeWorldPosition(coords)

            // center position of the image in world
            array[0] = smplWorldPosition.x / smplWorldPosition.w
            array[1] = smplWorldPosition.y / smplWorldPosition.w
            array[2] = smplWorldPosition.z / smplWorldPosition.w

            // translation
            array[3] = rootTransform[12]
            array[4] = rootTransform[13]
            array[5] = rootTransform[14]
            array[6] = rootTransform[0]

            // pose
            array[13] = poseLeftArm.value
            array[14] = poseRightArm.value
            array[15] = poseLeftLeg.value
            array[16] = poseRightLeg.value

            val bitmap = ImageDecoder.decodeBitmap(
                ImageDecoder.createSource(contentResolver, imageUri)
            ) { decoder, _, _ ->
                decoder.isMutableRequired = true
            }

            val request = RemoteGarmentReconstruction.RequestBuilder()
                .supplyPose(array)
                .supplyImage(bitmap)
                .build()

            try {
                if (loading.value) {
                    val buffer = garmentRepo.loadResource(request)
                    onCompleted(buffer)
                }
                loading.value = false
            }
            catch (e: Exception) {
                showAlert.value = true
                e.printStackTrace()
            }
        }
    }

    fun submitForReconstruction(contentResolver: ContentResolver, onCompleted: (ByteBuffer) -> Unit) {
        loading.value = true

        val center = imageCenter ?: return

        val dataArray = FloatArray(17)
        var sendRequest = false

        modelViewer.view.pick(center.x.toInt(), center.y.toInt(), executor) { result ->
            val callback = {
                // this is threadsafe because the executor is single threaded
                if (sendRequest) {
                    submissionHelper(dataArray, result.fragCoords, contentResolver, onCompleted)
                }
                sendRequest = true
            }

            getImageDataHelper(
                (imagePosition!!.x).toInt(),
                (imagePosition!!.y).toInt(),
                result.fragCoords[2],
                dataArray,
                7,
                callback
            )

            getImageDataHelper(
                (imagePosition!!.x + imageSize!!.width).toInt(),
                (imagePosition!!.y + imageSize!!.height).toInt(),
                result.fragCoords[2],
                dataArray,
                10,
                callback
            )
        }
    }

    private fun computeWorldPosition(fragCoords: FloatArray): Float4 {
        val viewport = modelViewer.view.viewport
        val clipSpaceX = (fragCoords[0] / viewport.width) * 2f - 1f
        val clipSpaceY = (fragCoords[1] / viewport.height) * 2f - 1f
        val clipSpaceZ = fragCoords[2] * 2f - 1f
        val clipSpacePosition = Float4(clipSpaceX, clipSpaceY, clipSpaceZ, 1f)

        val projectionD = DoubleArray(16)
        val projectionF = FloatArray(16)
        val model = FloatArray(16)
        modelViewer.camera.getProjectionMatrix(projectionD)
        modelViewer.camera.getModelMatrix(model)
        for (i in projectionD.indices) {
            projectionF[i] = projectionD[i].toFloat()
        }

        val cameraPosition = FloatArray(3)
        modelViewer.camera.getPosition(cameraPosition)

        val viewSpacePosition = transpose(inverse(Mat4.of(*projectionF))) * clipSpacePosition

        return transpose(Mat4.of(*model)) * viewSpacePosition
    }

    fun confirmResult(onCompleted: () -> Unit) {
        loading.value = true
        loadingJob = viewModelScope.launch {
            try {
                val res = garmentRepo.loadResource(
                    RemoteGarmentReconstruction.RequestBuilder().setWeightTransfer(true).build()
                )
                garmentRepo["final"] = res
                navigationController.navigateTo("PoseDetection")
            }
            catch (e: Exception) {
                e.printStackTrace()
            }
            loading.value = false

            onCompleted()
        }
    }

    fun resetPose() {
        poseLeftArm.value = 0f
        poseRightArm.value = 0f
        poseLeftLeg.value = 0f
        poseRightLeg.value = 0f
    }

    private fun poseSMPL() {
        modelViewer.asset?.apply {
            val transformManager = modelViewer.engine.transformManager
            jointIDs.forEach { (id, state) ->
                transformManager.getInstance(getFirstEntityByName(id)).also { entity ->
                    // get transformation matrices
                    val transform = FloatArray(16).also { arr -> transformManager.getTransform(entity, arr) }
                    val R = rotation(Float3(0f, 0f, state.value)).toFloatArray()

                    // add original translation
                    R[12] = transform[12]
                    R[13] = transform[13]
                    R[14] = transform[14]

                    transformManager.setTransform(entity, R)
                }
            }

            if (!::rootTransform.isInitialized) {
                rootTransform = FloatArray(16).also { arr -> transformManager.getTransform(transformManager.getInstance(this.root), arr) }
            }
        }

        modelViewer.animator?.updateBoneMatrices()
    }

    fun cancelLoading() {
        if (::loadingJob.isInitialized) loadingJob.cancel()
        loading.value = false
    }

    fun retry() {
        resultShowCase.value = false
        viewModelScope.launch {
            modelViewer.reset()
            replaceAsset(modelRetriever.postRetrievalRequest(null))
        }
    }

    fun showResult() {
        resultShowCase.value = true
        modelViewer.reset()
        modelViewer.switchGesture(orbit = true)
    }
}