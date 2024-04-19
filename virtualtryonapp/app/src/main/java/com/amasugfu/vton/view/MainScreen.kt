package com.amasugfu.vton.view

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.amasugfu.vton.R
import com.amasugfu.vton.viewmodel.MainViewModel

@Composable
fun MainScreen() {
    val viewModel: MainViewModel = viewModel()

    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier.fillMaxSize().padding(vertical = 50.dp)
    ) {
        // center button
        Column (
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(bottom = 100.dp)
        ) {
            // upload button
            when (viewModel.inputMode.value) {
                MainViewModel.InputMode.UPLOAD_IMAGE ->
                    SquareDashedIconButton(viewModel::openImageSelection, R.drawable.ic_image, "Upload an Image")
                MainViewModel.InputMode.UPLOAD_MODEL ->
                    SquareDashedIconButton(viewModel::openModelSelection, R.drawable.ic_storage, "Upload a Saved Model")
                MainViewModel.InputMode.CAMERA ->
                    SquareDashedIconButton(viewModel::openCamera, R.drawable.ic_camera, "Take a Photo")
            }

            Spacer(Modifier.height(100.dp))

            // button menu
            Row (
                verticalAlignment = Alignment.CenterVertically,
            ) {
                MenuButton(
                    MainViewModel.InputMode.UPLOAD_IMAGE.name,
                    R.drawable.ic_image,
                    viewModel.inputMode,
                    0
                )
                MenuButton(
                    MainViewModel.InputMode.UPLOAD_MODEL.name,
                    R.drawable.ic_storage,
                    viewModel.inputMode,
                    1
                )
                MenuButton(
                    MainViewModel.InputMode.CAMERA.name,
                    R.drawable.ic_camera,
                    viewModel.inputMode,
                    2
                )
            }
        }
    }
}

@Composable
fun MenuButton(name: String, id: Int, target: MutableState<MainViewModel.InputMode>, pos: Int) {
    val radius = 15.dp
    val iconSize = 30.dp

    val inputMode = MainViewModel.InputMode.valueOf(name)
    val selected = target.value == inputMode

    val scale = animateFloatAsState(if (selected) 1.4f else 1.0f)
//    val offset = if (selected) 1.2f else 0f

    Button(
        onClick = { target.value = inputMode },
        content = {
            Image(
                painterResource(id),
                null,
                modifier = Modifier
                    .size(iconSize)
                    .graphicsLayer(
                        scaleX = scale.value,
                        scaleY = scale.value,
//                        translationY = offset,
                    ),
                colorFilter = ColorFilter.tint(if (selected) MaterialTheme.colorScheme.secondary else Color.White)
            )
        },
        shape = when (pos) {
            0 -> RoundedCornerShape(radius, 0.dp, 0.dp, radius)
            2 -> RoundedCornerShape(0.dp, radius, radius, 0.dp)
            else -> RectangleShape
        },
        colors = ButtonDefaults.buttonColors(MaterialTheme.colorScheme.primary)
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppBar(
    navigation: NavigationController,
    title: String = stringResource(R.string.app_display_name),
    supplementaryTitle: String? = null,
    actionsVisible: Boolean = true,
    content: @Composable () -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                colors = TopAppBarDefaults.smallTopAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    titleContentColor = MaterialTheme.colorScheme.onSurface
                ),
                modifier = Modifier.shadow(8.dp),
                title = {
                    Row {
                        val size = 5.em

                        Text(
                            title,
                            style = TextStyle.Default.copy(
                                fontSize = size,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = -3.sp
                            ),
                        )

                        if (supplementaryTitle != null) {
                            Text(
                                " | $supplementaryTitle",
                                style = TextStyle.Default.copy(
                                    fontSize = 5.em,
                                    fontWeight = FontWeight.Bold,
                                ),
                            )
                        }
                    }
                },
                actions = {
                    if (actionsVisible) {
                        IconButton(
                            onClick = { navigation.navigateTo("Settings") }
                        ) {
                            Icon(
                                tint = MaterialTheme.colorScheme.onSurface,
                                imageVector = Icons.Filled.Settings,
                                contentDescription = "Settings"
                            )
                        }
                    }
                }
            )
        }
    ) {
        Box(modifier = Modifier.padding(it)) {
            content()
        }
    }
}