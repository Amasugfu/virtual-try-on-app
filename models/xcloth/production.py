import torch
from torch import nn
from .settings.model_settings import xClothSettings, DEFAULT_XCLOTH_SETTINGS
from .components.encoder import Encoder
from .components.decoder import DepthDecoder, NormDecoder, RGBDecoder
from .components.reconstruct import GarmentModel3D

class XCloth(nn.Module):
    def __init__(self, settings: xClothSettings = DEFAULT_XCLOTH_SETTINGS) -> None:
        super().__init__()
        self._settings = settings
        self._encoder = Encoder(settings)
        self._parallel_decoders = nn.ModuleDict({
            "Depth": DepthDecoder(settings),
            "Norm": NormDecoder(settings),
            "RGB": RGBDecoder(settings)
        })
    
    @property
    def n_peelmaps(self):
        return self._settings.n_peelmaps

    def get_smpl_prior(self, x_img: torch.Tensor):
        from .train.preprocessing import make_peelmaps
        return x_img

    def forward(self, x_img: torch.Tensor, x_smpl: torch.Tensor = None):
        if x_smpl is None: x_smpl = self.get_smpl_prior(x_img)

        if len(x_img.shape) == 3:           # C x H x W
            x_img = x_img.unsqueeze(0)      # 1 x C x H x W

        if len(x_smpl.shape) == 3:          
            x_smpl = x_smpl.unsqueeze(0)      

        x = self._encoder(x_img, x_smpl)
        y = {name: decoder(x) for name, decoder in self._parallel_decoders.items()}
        y["Img"] = x_img
        return y
    
    def reconstruct3d(self, x_img: torch.Tensor):
        model = GarmentModel3D(*self(x_img))
        return model
    
    def save(self, path, n=None, loss_hist=None):
        torch.save({
            "epoch": n,
            "loss_hist": loss_hist,
            "state": self.state_dict(),
        }, path)

    def load(self, path):
        chkpt = torch.load(path)
        self.load_state_dict(chkpt["state"])
        return chkpt["epoch"], chkpt["loss_hist"]
    