package com.amasugfu.vton.di

import android.app.Application
import android.content.res.AssetManager
import com.amasugfu.vton.data.repo.*
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import java.nio.ByteBuffer
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
class ApplicationModule {
    @Provides
    @Singleton
    fun provideAssetManager(application: Application): AssetManager {
        return application.assets
    }
}

@Module
@InstallIn(SingletonComponent::class)
abstract class ApplicationInterfaceModule {
    @Singleton
    @Binds
    abstract fun bindGarmentRepo(
        garmentRepository: GarmentRepository
    ): IGarmentRepository

    @Singleton
    @RuntimeReconstruction
    @Binds
    abstract fun bindGarmentRetriever(
        remoteGarmentReconstruction: RemoteGarmentReconstruction
    ) : IResourceRetriever<ByteBuffer>

    @SMPL
    @Binds
    abstract fun bindSMPLRetriever(
        smplModelRetriever: SMPLModelRetriever
    ): IResourceRetriever<ByteBuffer>

    @Binds
    abstract fun bindPoseDetector(
        remotePoseReconstruction: RemotePoseReconstruction
    ) : IResourceRetriever<FloatArray>
}