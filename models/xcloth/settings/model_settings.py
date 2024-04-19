from dataclasses import dataclass
from typing import Tuple

import numpy as np

@dataclass
class xClothSettings:
    """
    @param input_h: input height
    @param input_w: input width
    @param n_intput_channels: input channels
    @param n_peelmaps: number of peeled representations
    """
    input_h: int = 512
    input_w: int = 512
    n_intput_channels: int = 3
    n_peelmaps: int = 4

    """
    @param n_updown_sampling: number of down/upsampling layers used for encoding & decoding
    @param size_kernel_conv: kernel size of the convolution layer
    @param size_kernel_sampling: kernel size of the down.upsampling layers
    """
    n_updown_sampling: int = 2
    size_kernel_conv: int = 7
    size_kernel_sampling: int = 3
    
    """
    @param resnet_expansion: expansion factor of resnet blocks: output channels = resnet_expansion*output_channels
    @param n_resnet_blocks: number of resnet blocks the encodes uses
    """
    resnet_expansion: int = 1
    n_resnet_blocks: int = 18

    # whether to downsample the result of the encoder
    # reserved for future use
    downsample_encoding_result: bool = False
    downsample_encoding_result_tgt: Tuple[int, int, int] = (256, 128, 128)


DEFAULT_XCLOTH_SETTINGS = xClothSettings()

    
@dataclass
class CameraSettings:
    z: float = 1
    fov: float | Tuple[float, float] = np.pi / 3
    

DEFAULT_CAMERA_SETTINGS = CameraSettings()