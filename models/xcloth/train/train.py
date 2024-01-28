import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from matplotlib import pyplot as plt
from typing import Type, List, Dict, Any

from ..production import XCloth
from ..train.data import MeshDataSet


def __epoch(
        model: XCloth, 
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        weight: List[float],
        reduction: str|None,
        separate_bg: bool):
    """
    @param: X_train: N x B x C[4 (image) + P (peelmaps)] x H[512] x W[512]
    @param: Y_train: Dict[N x B x P x C[1 (depth) | 3 (normal) | 3 (rgb)] x H x W]
    
    P: number of peelmaps
    N: number of batches
    B: batch size
    C: number of channels
    H: height
    W: width
    """

    loss_hist = []

    loss_l1 = nn.L1Loss(reduction=reduction)
    loss_l2 = nn.MSELoss(reduction=reduction)
    
    for X, y_depth, y_norm, y_rgb in dataloader:
        optimizer.zero_grad()
        
        # B x P x H x W
        X, y_depth, y_norm, y_rgb = X.cuda(), y_depth.cuda(), y_norm.cuda(), y_rgb.cuda()
        result = model(X[:, :3], X[:, 3:])
        
        loss_d = 0
        loss_norm = 0
        loss_rgb = 0

        # sum of loss of peelmap layers
        for j in range(model.n_peelmaps):

            # separate foreground and background loss
            if separate_bg:
                fg_mask = (y_depth[:, j] != 0).int()
                loss_d += weight[4]*loss_l1(result["Depth"][:, j]*fg_mask, y_depth[:, j]) \
                            + weight[5]*loss_l1(result["Depth"][:, j]*(1-fg_mask), y_depth[:, j])
            else:
                loss_d += loss_l1(result["Depth"][:, j], y_depth[:, j])

            loss_norm += loss_l2(result["Norm"][:, j], y_norm[:, j])

            if j < model.n_peelmaps - 1:
                loss_rgb += loss_l1(result["RGB"][:, j], y_rgb[:, j])

        # update weights per batch
        loss = weight[0]*loss_d + weight[2]*loss_norm + weight[3]*loss_rgb
        loss_hist.append(loss.item())       
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

    return np.mean(loss_hist)


def train_model(
        model: XCloth, 
        dataset: MeshDataSet,
        batch_size: int = 4,
        n_epoch: int = 20,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 5e-4,
        weight: List[float] = [1., 0.1, 1., 0.05],
        verbose: bool = False,
        plot_path: str|None = None,
        reduction: str|None = "sum",
        params_path: str|None = None,
        separate_bg: bool = False):
    
    """
    @param: X_train: N x B x C[8] x H[512] x W[512]
    @param: Y_train: Dict[N x B x P x C[1 (depth) | 3 (normal) | 3 (rgb)] x H x W]
    @param: weight: loss weight in the order of [depth, seg, norm, rgb, [fg, bg]]
    
    P: number of peelmaps
    N: number of batches
    B: batch size
    C: number of channels
    H: height
    W: width
    """
    loss_hist = []
    optim = optimizer(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.95)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    
    model.train()
    for epoch in range(1, n_epoch + 1):
        loss = __epoch(
            model,
            dataloader,
            optim,
            weight,
            reduction,
            separate_bg
        )

        loss_hist.append(loss)
        if verbose: print(f"epoch {epoch}: loss: {loss}")

        if params_path is not None:
            model.save(params_path)
            
        scheduler.step()

    if plot_path is not None:
        plt.plot(range(1, n_epoch + 1), loss_hist)
        plt.xticks(range(1, n_epoch + 1))
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.savefig(plot_path)


        
    
    
