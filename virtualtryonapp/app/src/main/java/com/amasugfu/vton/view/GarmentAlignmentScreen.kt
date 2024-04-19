package com.amasugfu.vton.view

import android.app.Activity
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInRoot
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.amasugfu.vton.R
import com.amasugfu.vton.view.theme.AlphaGray
import com.amasugfu.vton.viewmodel.GarmentAlignmentViewModel

@Composable
fun GarmentAlignmentScreen() {
    val viewModel: GarmentAlignmentViewModel = viewModel()
    val resultShowCase: Boolean by viewModel.resultShowCase

    Box {
        Model3DView(viewModel)

        if (!resultShowCase) {
            // for alignment
            AsyncImage(
                model = viewModel.imageUri,
                contentDescription = null,
                modifier = Modifier
                    .align(Alignment.Center)
                    .onGloballyPositioned {coordinates ->
                        viewModel.imageSize = coordinates.size
                        viewModel.imagePosition = coordinates.positionInRoot()
                    }
                    .fillMaxWidth(),
                alpha = if (viewModel.showGarmentOpaque.value) 1f else 0.3f
            )

            OptionMenu(
                options = {
                    AlignmentOptions()
                },
                extras = {
                    PoseSettings()
                }
            )
        }
        else {
            // for result showcase
            OptionMenu(
                options = {
                    ShowCaseOptions()
                }
            )
        }
    }

    if (viewModel.loading.value) {
        LoadingScreen()
    }

    var alert: Boolean by viewModel.showAlert

    if (alert) {
        AlertDialog(
            icon = {
                Icon(Icons.Default.Info, contentDescription = null)
            },
            title = {
                Text(text = "Connection Failed")
            },
            text = {
                Text(text = "The server might be down or you have no internet connection. Please try again later.")
            },
            onDismissRequest = {
                alert = false
                viewModel.loading.value = false
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        alert = false
                        viewModel.loading.value = false
                    }
                ) {
                    Text("Confirm", color = MaterialTheme.colorScheme.secondary)
                }
            },
        )
    }
}

@Composable
fun OptionMenu(options: @Composable () -> Unit, extras: @Composable () -> Unit = {}) {
    Box(
        modifier = Modifier.fillMaxSize()
    ) {
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(start = 30.dp, end = 30.dp, bottom = 20.dp)
        ) {
            Row(
                modifier = Modifier.align(Alignment.CenterHorizontally),
            ) {
                options()
            }

            extras()
        }
    }
}

@Composable
fun AlignmentOptions() {
    val viewModel: GarmentAlignmentViewModel = viewModel()
    var showGarment by viewModel.showGarmentOpaque
    var poseSettingsOpened by viewModel.poseSettingsOpened
    val activity = LocalContext.current as Activity

    // menu buttons
    ResetPoseButton(viewModel)

    RoundIconButton(
        { poseSettingsOpened = !poseSettingsOpened },
        R.drawable.ic_pose,
        isToggle = { poseSettingsOpened }
    )

    RoundIconButton(
        { showGarment = !showGarment },
        if (showGarment) R.drawable.ic_shirt else R.drawable.ic_shirt_outline,
        isToggle = { showGarment }
    )

    RoundIconButton(
        {
            poseSettingsOpened = false
            viewModel.submitForReconstruction(activity.contentResolver) {
                viewModel.showResult()
                viewModel.replaceAsset(it)
            }
        },
        R.drawable.ic_upload
    )
}

@Composable
fun ResetPoseButton(viewModel: GarmentAlignmentViewModel) {
    AnimatedVisibility(
        visible = viewModel.poseSettingsOpened.value,
        enter = expandIn { it } + fadeIn(initialAlpha = 0.3f),
        exit = shrinkOut() + fadeOut()
    ) {

        RoundIconButton(
            { viewModel.resetPose() },
            R.drawable.ic_reset
        )
    }
}

@Composable
fun PoseSettings() {
    val viewModel: GarmentAlignmentViewModel = viewModel()

    AnimatedVisibility(
        visible = viewModel.poseSettingsOpened.value,
        enter = expandVertically(
            // Expand from the top.
            expandFrom = Alignment.Bottom
        ) + fadeIn(
            // Fade in with the initial alpha of 0.3f.
            initialAlpha = 0.3f
        ),
        exit = shrinkVertically() + fadeOut()
    ) {
        val interactionSource = remember { MutableInteractionSource() }
        val touched by interactionSource.collectIsPressedAsState()

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 20.dp, bottom = 10.dp)
                .clip(shape = RoundedCornerShape(10.dp))
                .background(AlphaGray)
                .clickable (
                    onClick = { viewModel.allowOrbit = touched },
                    indication = null,
                    interactionSource = interactionSource
                ),
        ) {
            Column(
                modifier = Modifier.align(Alignment.Center).padding(25.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                PoseSlider("left arm", viewModel.poseLeftArm, GarmentAlignmentViewModel.rangeLeftArm)
                PoseSlider("right arm", viewModel.poseRightArm, GarmentAlignmentViewModel.rangeRightArm)
                PoseSlider("left leg", viewModel.poseLeftLeg, GarmentAlignmentViewModel.rangeLeftLeg)
                PoseSlider("right leg", viewModel.poseRightLeg, GarmentAlignmentViewModel.rangeRightLeg)
            }
        }
    }
}

@Composable
fun PoseSlider(name: String, sliderPosition: MutableState<Float>, bounds: ClosedFloatingPointRange<Float>) {
    var pos by sliderPosition

    Column {
        Text(
            name,
            color = Color.LightGray,
            modifier = Modifier.padding(0.dp)
        )
        Slider(
            value = pos,
            onValueChange = { pos = it },
            colors = SliderDefaults.colors(
                thumbColor = MaterialTheme.colorScheme.primary,
                activeTrackColor = MaterialTheme.colorScheme.primary,
                inactiveTrackColor = Color.LightGray,
            ),
            valueRange = bounds
        )
    }
}

@Composable
fun LoadingScreen() {
    Box(
        modifier = Modifier
            .background(AlphaGray)
            .fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        CircularProgressIndicator(
            modifier = Modifier
                .width(80.dp),
            color = MaterialTheme.colorScheme.secondary,
        )
    }
}

@Composable
fun ShowCaseOptions() {
    val viewModel: GarmentAlignmentViewModel = viewModel()

    RoundIconButton(
        onClick = {
            viewModel.retry()
        },
        id = R.drawable.ic_reset
    )

    RoundIconButton(
        onClick = {
            viewModel.confirmResult()
        },
        id = R.drawable.ic_tick
    )
}