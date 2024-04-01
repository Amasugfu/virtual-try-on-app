package com.amasugfu.vton.ui

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.TextUnitType
import androidx.compose.ui.unit.dp
import com.amasugfu.vton.R
import com.amasugfu.vton.ui.theme.CustomCyan
import com.amasugfu.vton.ui.theme.VirtualtryonappTheme
import com.amasugfu.vton.viewmodel.MainViewModel

@Composable
fun MainScreen(viewModel: MainViewModel = MainViewModel()) {
    Box(
        contentAlignment = Alignment.TopCenter,
        modifier = Modifier.fillMaxSize()
    ) {
        // center button
        Column (
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(Modifier.height(100.dp))

            // upload button
            when (viewModel.inputMode.value) {
                MainViewModel.InputMode.UPLOAD_IMAGE ->
                    SquareButton(viewModel::openImageSelection, R.drawable.ic_image, "Upload an Image")
                MainViewModel.InputMode.UPLOAD_MODEL ->
                    SquareButton(viewModel::openModelSelection, R.drawable.ic_storage, "Upload a Saved Model")
                MainViewModel.InputMode.CAMERA ->
                    SquareButton(viewModel::openCamera, R.drawable.ic_camera, "Take a Photo")
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
                colorFilter = ColorFilter.tint(if (selected) Color.Black else Color.White)
            )
        },
        shape = when (pos) {
            0 -> RoundedCornerShape(radius, 0.dp, 0.dp, radius)
            2 -> RoundedCornerShape(0.dp, radius, radius, 0.dp)
            else -> RectangleShape
        },
        colors = ButtonDefaults.buttonColors(CustomCyan)
    )
}

@Composable
fun SquareButton(onClick: () -> Unit, id: Int, hint: String = "") {
    OutlinedButton(
        onClick = onClick,
        modifier = Modifier
            .size(300.dp)
            .background(Color.Transparent)
            .drawBehind {
                drawRoundRect(
                    color = Color.LightGray,
                    cornerRadius = CornerRadius(20.dp.toPx()),
                    style = Stroke(
                        width = 5.dp.toPx(),
                        pathEffect = PathEffect.dashPathEffect(
                            floatArrayOf(10.dp.toPx(), 10.dp.toPx())
                    ),
                ))
            },
        shape = RectangleShape,
        border = null
    ) {
        Column (
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Image(
                painter = painterResource(id = id),
                contentDescription = null,
                modifier = Modifier.size(100.dp),
                colorFilter = ColorFilter.tint(CustomCyan)
            )

            Spacer(Modifier.height(25.dp))

            Text(
                text = hint,
                modifier = Modifier
                    .border(3.dp, CustomCyan, RoundedCornerShape(10.dp))
                    .width(200.dp)
                    .padding(3.dp, 5.dp, 3.dp, 5.dp),
                color = CustomCyan,
                textAlign = TextAlign.Center,
                fontSize = TextUnit(6f, TextUnitType.Em)
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun MainScreenPreview() {
    VirtualtryonappTheme {
        MainScreen()
    }
}