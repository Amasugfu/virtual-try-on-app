import torch
from torch import nn
from torchvision.models.resnet import Bottleneck, conv1x1

from .sampling import build_sampling_layers, ConvDownSampling
from ..settings.model_settings import DEFAULT_XCLOTH_SETTINGS, xClothSettings

"""
adapted from the original torch implementation of resnet
reference: pytorch, (June, 2023). https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L225. Github. 
"""
def _make_layer(
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        
        downsample = None
        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(
            Bottleneck(
                inplanes, planes, stride, downsample
            )
        )
        inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    inplanes,
                    planes,
                )
            )

        return nn.Sequential(*layers)


class Encoder(nn.Module):
    
    def __init__(self, settings: xClothSettings = DEFAULT_XCLOTH_SETTINGS) -> None:
        super().__init__()

        # rgb image to encoding with 64 channela
        self.conv2d = nn.Conv2d(
             settings.n_intput_channels + settings.n_peelmaps, 
             64, 
             settings.size_kernel_conv, 
             padding=(settings.size_kernel_conv - 1)//2
        )

        # strided conv downsampling
        # self.downsampling1 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        # self.downsampling2 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.downsampling = build_sampling_layers(
            ConvDownSampling, 64, settings.size_kernel_sampling, 2, settings.n_updown_sampling
        )

        # resnet encoding
        # input: batch(N) x 256(C) x 128(H) x 128(W)
        Bottleneck.expansion = settings.resnet_expansion
        self.res_blocks1 =  _make_layer(256, 256, settings.n_resnet_blocks)

        if settings.downsample_encoding_result:
             self.downsampling_last = self.build_final_downsampling_layer(settings.downsample_encoding_result_tgt)
             
    def build_final_downsampling_layer(self, tgt):
         pass

    def forward(self, x_img: torch.Tensor, x_smpl: torch.Tensor):
        """
        @param x_img: batch(N) x 4(C) x 512(H) x 512(W) rgba image
        @param x_smpl: batch(N) x 4(C) x 512(H) x 512(W) peel map
        """

        # concat
        x = torch.concat((x_img, x_smpl), dim=1)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.conv2d(x)
        x = self.downsampling(x)
        x = self.res_blocks1(x)
        # x = self.res_blocks2(x)

        return x