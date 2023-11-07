import torch
from torch import nn

from .sampling import build_sampling_layers, ConvUpSampling
from ..settings.model_settings import DEFAULT_XCLOTH_SETTINGS, xClothSettings

class BaseDecoder(nn.Module):
    def __init__(self, output_factor:int = 1, activation: nn.Module = nn.Sigmoid(), settings: xClothSettings = DEFAULT_XCLOTH_SETTINGS) -> None:
        super().__init__()

        # upsampling
        # self.upsampling1 = nn.ConvTranspose2d(256, 128, 3, stride=2)
        # self.upsampling2 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.upsampling = build_sampling_layers(ConvUpSampling, 256, 3)

        # decode conv
        self.conv2d = nn.Conv2d(64, settings.n_peelmaps*output_factor, 7, padding=3)

        # activation function
        self.act = activation

    def forward(self, x: torch.Tensor):
        """
        @param x: output of the shared encoder
        """
        x = self.upsampling(x)
        x = self.conv2d(x)
        x = self.act(x)

        return x


class DepthDecoder(BaseDecoder):
    def __init__(self, settings: xClothSettings = DEFAULT_XCLOTH_SETTINGS) -> None:
        super().__init__(settings=settings)

    
class NormDecoder(BaseDecoder):
    """
    @return: batch(N) x channels(C = n_peelmaps*4) x 512 x 512 
    """
    def __init__(self, settings: xClothSettings = DEFAULT_XCLOTH_SETTINGS) -> None:
        super().__init__(output_factor=4, settings=settings)


class RGBDecoder(BaseDecoder):
    """
    @return: batch(N) x channels(C = n_peelmaps*3) x 512 x 512 
    """
    def __init__(self, settings: xClothSettings = DEFAULT_XCLOTH_SETTINGS) -> None:
        super().__init__(output_factor=3, activation=nn.Tanh(), settings=settings)

