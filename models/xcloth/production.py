import torch
from torch import nn
from .settings.model_settings import xClothSettings, DEFAULT_XCLOTH_SETTINGS
from .components.encoder import Encoder
from .components.decoder import DepthDecoder, NormDecoder, RGBDecoder
from .components.utils import GarmentModel3D


class XCloth(nn.Module):
    def __init__(self, settings: xClothSettings = DEFAULT_XCLOTH_SETTINGS) -> None:
        super().__init__()
        self.settings = settings
        self.encoder = Encoder(settings)
        self.parallel_decoders = nn.ModuleDict({
            "Depth": DepthDecoder(settings),
            "Norm": NormDecoder(settings),
            "RGB": RGBDecoder(settings)
        })

    def get_smpl_prior(self, x_img: torch.Tensor):
        return x_img

    def forward(self, x_img: torch.Tensor):
        x_smpl = self.get_smpl_prior(x_img)
        x = self.encoder(x_img, x_smpl)
        y = [decoder(x) for name, decoder in self.parallel_decoders.items()]
        return y
    
    def reconstruct3d(self, x_img: torch.Tensor):
        model = GarmentModel3D(*self(x_img))
        return model