package com.amasugfu.vton.di

import com.amasugfu.vton.data.IPoseDetectionSession
import com.amasugfu.vton.data.PoseDetectionSession
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.android.components.ViewModelComponent

@Module
@InstallIn(ViewModelComponent::class)
abstract class ViewModelModule {
    @Binds
    abstract fun bindPoseDetectionSession(
        poseDetectionSession: PoseDetectionSession
    ): IPoseDetectionSession

    // debug
//    @Binds
//    abstract fun bindGarmentModelRetriever(
//        localGarmentModelRetriever: DebugLocalGarmentModelRetriever
//    ): IResourceRetriever
}