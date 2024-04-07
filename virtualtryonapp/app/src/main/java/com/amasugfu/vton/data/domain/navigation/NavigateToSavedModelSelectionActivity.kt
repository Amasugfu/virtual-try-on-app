package com.amasugfu.vton.data.domain.navigation

import androidx.activity.ComponentActivity

class NavigateToSavedModelSelectionActivity(
    context: ComponentActivity,
    onCompleted: (Any?) -> Unit
) : NavigateUsingContextUseCase(context, onCompleted) {
    override fun navigate() {
        TODO("Not yet implemented")
    }
}