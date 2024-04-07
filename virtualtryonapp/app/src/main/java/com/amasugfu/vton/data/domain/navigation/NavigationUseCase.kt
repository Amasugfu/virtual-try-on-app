package com.amasugfu.vton.data.domain.navigation

import androidx.activity.ComponentActivity

interface NavigationUseCase {
    fun navigate()
}

abstract class NavigateUsingContextUseCase(
    val context: ComponentActivity,
    val onCompleted: (Any?) -> Unit = {}
) : NavigationUseCase