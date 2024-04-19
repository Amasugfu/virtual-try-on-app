package com.amasugfu.vton.data.repo

import GarmentReconstructionGrpcKt
import PoseDetectionGrpcKt
import Requests
import Requests.GarmentReconstructionRequest
import android.content.res.AssetManager
import android.graphics.Bitmap
import com.amasugfu.vton.data.domain.GetSharedPreferenceUseCase
import com.google.android.filament.utils.Float4
import com.google.android.filament.utils.Mat4
import floatMat
import io.grpc.Channel
import io.grpc.ManagedChannelBuilder
import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Qualifier

interface IResourceRetriever<T> {
    suspend fun postRetrievalRequest(data: Any?): T
}

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class SMPL

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class RuntimeReconstruction

fun readAsset(assetManager: AssetManager, name: String): ByteBuffer {
    val input = assetManager.open(name)
    val bytes = ByteArray(input.available())
    input.read(bytes)
    return ByteBuffer.wrap(bytes)
}

class SMPLModelRetriever @Inject constructor(
    val assetManager: AssetManager
) : IResourceRetriever<ByteBuffer> {
    override suspend fun postRetrievalRequest(data: Any?): ByteBuffer = readAsset(assetManager, "models/smpl_male_blend3.glb")
}

class RemoteGarmentReconstruction @Inject constructor(
    val getSharedPreferenceUseCase: GetSharedPreferenceUseCase,
) : IResourceRetriever<ByteBuffer> {

    protected lateinit var channel: Channel
    protected lateinit var stub: GarmentReconstructionGrpcKt.GarmentReconstructionCoroutineStub

    fun connect() {
        val preferences = getSharedPreferenceUseCase.execute()
        val host = GetSharedPreferenceUseCase.getHost(preferences)
        val port = GetSharedPreferenceUseCase.getPort(preferences)

        channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build()
        stub = GarmentReconstructionGrpcKt.GarmentReconstructionCoroutineStub(channel)
    }

    class RequestBuilder {
        val reqBuilder = GarmentReconstructionRequest.newBuilder()

        fun supplyImage(bitmap: Bitmap): RequestBuilder {
            val mat = floatMat {
                this.numDim = 3

                this.shape.add(bitmap.height)
                this.shape.add(bitmap.width)
                this.shape.add(3)

                for (y in 0 until bitmap.height) {
                    for (x in 0 until bitmap.width) {
                        val color = bitmap.getColor(x, y)
                        this.data.add(color.red())
                        this.data.add(color.green())
                        this.data.add(color.blue())
                    }
                }
            }

            reqBuilder.setGarmentImg(mat)

            return this
        }

        fun supplyPose(pose: FloatArray): RequestBuilder {
            reqBuilder.setPose(
                floatMat {
                    this.numDim = 1
                    this.shape.add(pose.size)
                    for (fl in pose) {
                        this.data.add(fl)
                    }
                }
            )
            return this
        }

        fun build(): GarmentReconstructionRequest {
            return reqBuilder.build()
        }
    }

    override suspend fun postRetrievalRequest(data: Any?): ByteBuffer {
        // call remote service
        connect()
        val bytes = stub.withDeadlineAfter(3, TimeUnit.MINUTES)
            .reconstruct(data as GarmentReconstructionRequest)

        return ByteBuffer.wrap(byteArrayOf())
    }
}

class RemotePoseReconstruction @Inject constructor(
    val getSharedPreferenceUseCase: GetSharedPreferenceUseCase,
) : IResourceRetriever<ArrayList<Mat4>> {

    protected lateinit var channel: Channel
    protected lateinit var stub: PoseDetectionGrpcKt.PoseDetectionCoroutineStub

    fun connect() {
        if (::channel.isInitialized) return

        val preferences = getSharedPreferenceUseCase.execute()
        val host = GetSharedPreferenceUseCase.getHost(preferences)
        val port = GetSharedPreferenceUseCase.getPort(preferences)

        channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build()
        stub = PoseDetectionGrpcKt.PoseDetectionCoroutineStub(channel)
    }

    override suspend fun postRetrievalRequest(data: Any?): ArrayList<Mat4> {
        // call remote service
        connect()

        val buffer = data as Requests.ByteBuffer
        val floatMat = stub.withDeadlineAfter(3, TimeUnit.MINUTES)
            .getPose(buffer)

        val response = ArrayList<Mat4>()

        for (i in 0..23) {
            response.add(
                Mat4(
                    Float4(floatMat.getData(i*16    ), floatMat.getData(i*16 + 1), floatMat.getData(i*16 + 2), floatMat.getData(i*16 + 3)),
                    Float4(floatMat.getData(i*16 + 4), floatMat.getData(i*16 + 5), floatMat.getData(i*16 + 6), floatMat.getData(i*16 + 7)),
                    Float4(floatMat.getData(i*16 + 8), floatMat.getData(i*16 + 9), floatMat.getData(i*16 + 10), floatMat.getData(i*16 + 11)),
                    Float4(floatMat.getData(i*16 + 12), floatMat.getData(i*16 + 13), floatMat.getData(i*16 + 14), floatMat.getData(i*16 + 15)),
                )
            )
        }

        return response
    }
}