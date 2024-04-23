package com.amasugfu.vton.view

import android.view.SurfaceView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.TextUnitType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.amasugfu.vton.viewmodel.Model3DViewModel

@Composable
fun SquareDashedIconButton(onClick: () -> Unit, id: Int, hint: String = "") {
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
                    )
                )
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
                colorFilter = ColorFilter.tint(MaterialTheme.colorScheme.primary)
            )

            Spacer(Modifier.height(25.dp))

            Text(
                text = hint,
                modifier = Modifier
                    .border(3.dp, MaterialTheme.colorScheme.primary, RoundedCornerShape(10.dp))
                    .width(200.dp)
                    .padding(3.dp, 5.dp, 3.dp, 5.dp),
                color = MaterialTheme.colorScheme.primary,
                textAlign = TextAlign.Center,
                fontSize = TextUnit(5f, TextUnitType.Em)
            )
        }
    }
}

@Composable
fun RoundIconButton(
    onClick: () -> Unit,
    id: Int,
    background: Color = MaterialTheme.colorScheme.primary,
    tint: Color = MaterialTheme.colorScheme.secondary,
    isToggle: () -> Boolean = { false }
) {
    var background_ = background
    var tint_ = tint

    if (isToggle.invoke()) {
        background_ = tint_.also { tint_ = background_ }
    }

    Box (
        modifier = Modifier.padding(10.dp)
    ) {
        IconButton(
            onClick = onClick,
            modifier = Modifier
                .size(40.dp)
                .shadow(3.dp, CircleShape)
                .background(background_, CircleShape),
            content = {
                Icon(
                    modifier = Modifier.padding(8.dp),
                    painter = painterResource(id = id),
                    contentDescription = null,
                    tint = tint_
                )
            }
        )
    }
}

@Composable
fun Model3DView(viewModel: Model3DViewModel, transparent: Boolean = false) {
    val context = LocalContext.current

    AndroidView(
        factory = {
            SurfaceView(context)
                .also {
                    viewModel.bindSurface(it)
                }
                .also {
                    viewModel.displayScene(it, transparent)
                }
        },
        modifier = Modifier
            .fillMaxSize()
    )
}