package com.amasugfu.vton.view

import com.amasugfu.vton.data.domain.navigation.NavigateUsingContextUseCase
import com.amasugfu.vton.data.domain.navigation.NavigationUseCase
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class NavigationController @Inject constructor() {
    private val activityFactory: HashMap<String, NavigationUseCase> = HashMap()

    fun navigateTo(namespace: String) {
        activityFactory[namespace]?.navigate()
    }

    fun registerNavigation(namespace: String, callback: () -> Unit) {
        activityFactory[namespace] = object : NavigationUseCase {
            override fun navigate() {
                callback()
            }
        }
    }

    fun registerNavigation(namespace: String, useCase: NavigationUseCase) {
        activityFactory[namespace] = useCase
    }
}