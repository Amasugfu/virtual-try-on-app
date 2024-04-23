import com.google.protobuf.gradle.id
import com.google.protobuf.gradle.proto

plugins {
    kotlin("kapt")
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.dagger.hilt.android")
    id("com.google.protobuf") version "0.9.4"
}

android {
    namespace = "com.amasugfu.vton"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.amasugfu.vton"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    sourceSets.getByName("main") {
        proto {
            srcDir("../../public")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.4.3"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

kotlin {
    jvmToolchain(17)
}

val protobufVersion = "3.25.1"
val grpcVersion = "1.62.2"
val grpcKtVersion = "1.4.1"

protobuf {
    protoc {
        artifact = "com.google.protobuf:protoc:${protobufVersion}"
    }

    plugins {
        id("grpc") {
            artifact = "io.grpc:protoc-gen-grpc-java:${grpcVersion}" // CURRENT_GRPC_VERSION
        }

        id("grpckt") {
            artifact = "io.grpc:protoc-gen-grpc-kotlin:${grpcKtVersion}:jdk8@jar"
        }

        id("java") {
            artifact = "io.grpc:protoc-gen-grpc-java:${grpcVersion}"
        }
    }

    generateProtoTasks {
        all().forEach { task ->
            task.builtins {
                id("kotlin")
            }
            task.plugins {
                id("grpc") {
                    option("lite")
                    outputSubDir = "grpc_java"
                }

                id("grpckt") {
                    option("lite")
                    outputSubDir = "grpc_kt"
                }

                id("java") {
                    option("lite")
                    outputSubDir = "grpc_java"
                }

                id("python") {
                    outputSubDir = "grpc_py"
                }
            }
        }
    }
}

dependencies {
    implementation("com.google.guava:guava:33.1.0-android")

    implementation("androidx.core:core-ktx:1.9.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.6.1")
    implementation("androidx.activity:activity-compose:1.7.0")
    implementation("androidx.preference:preference-ktx:1.2.1")
    implementation("com.google.mlkit:pose-detection:18.0.0-beta4")
//    implementation("com.google.mlkit:pose-detection-accurate:18.0.0-beta4")

    implementation("io.coil-kt:coil-compose:2.6.0")

    val cameraxVersion = "1.4.0-alpha04"
    implementation("androidx.camera:camera-camera2:${cameraxVersion}")
    implementation("androidx.camera:camera-view:${cameraxVersion}")
    implementation("androidx.camera:camera-lifecycle:${cameraxVersion}")

    val cameraxVisionVersion = "1.3.0-beta02"
    implementation("androidx.camera:camera-mlkit-vision:${cameraxVisionVersion}")

    // filament for 3D rendering
    val filamentVersion = "1.51.2"
    implementation("com.google.android.filament:filament-android:${filamentVersion}")
    implementation("com.google.android.filament:gltfio-android:${filamentVersion}")
    implementation("com.google.android.filament:filament-utils-android:${filamentVersion}")

    // gRPC
    implementation("io.grpc:grpc-okhttp:${grpcVersion}")
    implementation("io.grpc:grpc-protobuf-lite:${grpcVersion}")
    implementation("io.grpc:grpc-stub:${grpcVersion}")
    implementation("io.grpc:grpc-kotlin-stub:${grpcKtVersion}")
    implementation("com.google.protobuf:protobuf-javalite:${protobufVersion}")
    implementation("com.google.protobuf:protobuf-kotlin-lite:${protobufVersion}")

    implementation("com.google.dagger:hilt-android:2.44")
    kapt("com.google.dagger:hilt-android-compiler:2.44")

    implementation(platform("androidx.compose:compose-bom:2023.03.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")

    testImplementation("junit:junit:4.13.2")

    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation(platform("androidx.compose:compose-bom:2023.03.00"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")

    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}

kapt {
    correctErrorTypes = true
}