package com.amasugfu.vton.data.repo

import android.app.Application
import android.net.Uri
import androidx.lifecycle.ViewModel
import com.amasugfu.vton.data.repo.IGarmentRepository.Companion.DEFAULT_MAX_QUEUE_SIZE
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentLinkedQueue
import javax.inject.Inject


data class GarmentData(
    val resourceUri: Uri,
    val resourceType: ResourceType,
)

enum class ResourceType {
    IMAGE, MESH
}

interface IGarmentRepository {
    companion object {
        val DEFAULT_MAX_QUEUE_SIZE = 1
    }

    suspend fun loadResource(data: Any?): ByteBuffer
    fun pushDataTo(data: GarmentData, viewModelClass: Class<out ViewModel>)
    fun getData(viewModelClass: Class<out ViewModel>): GarmentData?
    fun getLoadedResources(): List<ByteBuffer>

    operator fun get(key: String): ByteBuffer?
    operator fun set(key: String, data: ByteBuffer)
}

class GarmentRepository @Inject constructor(
    application: Application,
    val resourceRetriever: IResourceRetriever<ByteBuffer>
) : IGarmentRepository {

    // to communicate between view models
    private val dataQueue: HashMap<Class<out ViewModel>, ConcurrentLinkedQueue<GarmentData>> = HashMap()

    // saved state
    private val loadedResources: HashMap<String, ByteBuffer> = HashMap()

    override suspend fun loadResource(data: Any?): ByteBuffer = resourceRetriever.postRetrievalRequest(data)

    override fun pushDataTo(data: GarmentData, viewModelClass: Class<out ViewModel>) {
        if (!dataQueue.containsKey(viewModelClass)) {
            dataQueue[viewModelClass] = ConcurrentLinkedQueue()
        }
        val q = dataQueue[viewModelClass]!!
        if (q.size == DEFAULT_MAX_QUEUE_SIZE) {
            q.poll()
        }
        q.add(data)
    }

    override fun getData(viewModelClass: Class<out ViewModel>): GarmentData? {
        return dataQueue[viewModelClass]?.poll()
    }

    override fun getLoadedResources(): List<ByteBuffer> = loadedResources.values.toList()

    override fun get(key: String) = loadedResources[key]
    override fun set(key: String, data: ByteBuffer) = loadedResources.set(key, data)
}