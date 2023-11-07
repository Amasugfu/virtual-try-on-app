import torch
from torch import nn
from typing import Type, Union

class ConvDownSampling(nn.Module):
    def __init__(self, channels_in, kernel_size, factor=2) -> None:
        super().__init__()

        channel_out = channels_in * factor
        self.conv = nn.Conv2d(
            channels_in, 
            channel_out, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1)//2, 
            stride=kernel_size - 1
        )
        self.norm = nn.BatchNorm2d(channel_out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        self.activation(x)
        return x


class ConvUpSampling(nn.Module):
    def __init__(self, channels_in, kernel_size, factor=2) -> None:
        super().__init__()

        channel_out = channels_in // factor
        self.conv = nn.ConvTranspose2d(
            channels_in, 
            channel_out, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1)//2, 
            output_padding=1,
            stride=kernel_size - 1
        )
        self.norm = nn.BatchNorm2d(channel_out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        self.activation(x)
        return x
    

def build_sampling_layers(
        layer: Type[Union[ConvDownSampling, ConvUpSampling]],
        channels_in: int, 
        kernel_size: int, 
        factor: int = 2, 
        n: int = 2,
    ) -> nn.Sequential:
    return nn.Sequential(
        *(layer(channels_in*factor**(i), kernel_size, factor) for i in range(n))
    ) if layer == ConvDownSampling else nn.Sequential(
        *(layer(channels_in // (factor**(i)), kernel_size, factor) for i in range(n))
    )


