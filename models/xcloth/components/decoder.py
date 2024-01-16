import torch
from torch import Tensor, nn

from .sampling import build_sampling_layers, ConvUpSampling
from ..settings.model_settings import DEFAULT_XCLOTH_SETTINGS, xClothSettings

class BaseDecoder(nn.Module):
    def __init__(self, n_peelmaps, out_channels:int = 1, activation: nn.Module = nn.Sigmoid()) -> None:
        super().__init__()

        self._n_peelmaps = n_peelmaps
        self._out_channels = out_channels

        # upsampling
        # self.upsampling1 = nn.ConvTranspose2d(256, 128, 3, stride=2)
        # self.upsampling2 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.upsampling = build_sampling_layers(ConvUpSampling, 256, 3)

        # decode conv
        self.conv2d = nn.Conv2d(64, n_peelmaps*out_channels, 7, padding=3)

        # activation function
        self.act = activation

    def forward(self, x: torch.Tensor):
        """
        @param x: output of the shared encoder

        @return: B x P x C x H x W tensor
        """
        x = self.upsampling(x)
        x = self.conv2d(x)
        x = self.act(x)

        return x.reshape(-1, self._n_peelmaps, self._out_channels, x.shape[-2], x.shape[-1])


class DepthDecoder(BaseDecoder):
    def __init__(self, settings: xClothSettings = DEFAULT_XCLOTH_SETTINGS) -> None:
        super().__init__(n_peelmaps=settings.n_peelmaps)

    # def forward(self, x: Tensor):
    #     x = super().forward(x)

    #     # # scale to range [-1, 0.5]
    #     # x_min, _ = torch.min(x, dim=1, keepdim=True)
    #     # x_max, _ = torch.max(x, dim=1, keepdim=True)
    #     # x = (x - x_min) * (1.5) / (x_max - x_min) - 1

    #     return x

    
class NormDecoder(BaseDecoder):
    """
    @return: batch(B) x channels(C = n_peelmaps*3) x 512 x 512 
    """
    def __init__(self, settings: xClothSettings = DEFAULT_XCLOTH_SETTINGS) -> None:
        super().__init__(out_channels=3, n_peelmaps=settings.n_peelmaps)


class RGBDecoder(BaseDecoder):
    """
    @return: batch(B) x channels(C = (n_peelmaps - 1)*3) x 512 x 512 
    """
    def __init__(self, settings: xClothSettings = DEFAULT_XCLOTH_SETTINGS) -> None:
        super().__init__(out_channels=3, n_peelmaps=settings.n_peelmaps - 1, activation=nn.Tanh())

    def forward(self, x: Tensor):
        return (super().forward(x) + 1)/2

