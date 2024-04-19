package com.amasugfu.vton.data.domain

import android.app.Application
import android.content.SharedPreferences
import androidx.preference.PreferenceManager
import com.amasugfu.vton.view.theme.ConnectionSettingsHostKey
import com.amasugfu.vton.view.theme.ConnectionSettingsPortKey
import com.amasugfu.vton.view.theme.DefaultConnectionAddress
import com.amasugfu.vton.view.theme.DefaultConnectionPort
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class GetSharedPreferenceUseCase @Inject constructor(
    val application: Application
) {
    fun execute(): SharedPreferences {
        return PreferenceManager.getDefaultSharedPreferences(application)
    }

    companion object {
        fun getHost(sharedPreferences: SharedPreferences): String {
            return sharedPreferences.getString(ConnectionSettingsHostKey, DefaultConnectionAddress)!!
        }

        fun getPort(sharedPreferences: SharedPreferences): Int {
            return sharedPreferences.getInt(ConnectionSettingsPortKey, DefaultConnectionPort)
        }
    }
}