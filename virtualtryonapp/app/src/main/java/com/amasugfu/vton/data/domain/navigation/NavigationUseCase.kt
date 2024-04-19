package com.amasugfu.vton.data.domain.navigation

import android.content.Intent
import androidx.activity.ComponentActivity

interface NavigationUseCase {
    fun navigate()
}

abstract class NavigateUsingContextUseCase(
    val context: ComponentActivity,
    val onCompleted: (Any?) -> Unit = {}
) : NavigationUseCase

class NavigateToActivity(
    private val clazz: Class<out ComponentActivity>,
    context: ComponentActivity,
    onCompleted: (Any?) -> Unit = {}
) : NavigateUsingContextUseCase(context, onCompleted) {
    override fun navigate() {
        Intent(context.applicationContext, clazz).also { context.startActivity(it) }
    }
}