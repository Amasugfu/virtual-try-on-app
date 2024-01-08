import torch
import torch.nn as nn
import numpy as np

from matplotlib import pyplot as plt
from typing import Type, List, Dict

from ..production import XCloth


def __epoch(
        model: XCloth, 
        X_train: torch.Tensor, 
        Y_train: Dict[str, torch.Tensor], 
        optimizer: torch.optim.Optimizer,
        weight: List[float],
        reduction: str|None):
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

    N = X_train.shape[0]
    loss_hist = []

    loss_l1 = nn.L1Loss(reduction=reduction)
    loss_l2 = nn.MSELoss(reduction=reduction)
    
    for i in range(N):
        optimizer.zero_grad()
        
        # B x P x H x W
        result = model(X_train[i, :, :3], X_train[i, :, 3:])
        
        loss_d = 0
        loss_norm = 0
        loss_rgb = 0

        # sum of loss of peelmap layers
        for j in range(model.n_peelmaps):
            loss_d += loss_l1(result["Depth"][:, j], Y_train["Depth"][i, :, j])
            loss_norm += loss_l2(result["Norm"][:, j], Y_train["Norm"][i, :, j])

            if j < model.n_peelmaps - 1:
                loss_rgb += loss_l1(result["RGB"][:, j], Y_train["RGB"][i, :, j])

        # update weights per batch
        loss = weight[0]*loss_d + weight[2]*loss_norm + weight[3]*loss_rgb
        loss_hist.append(loss)       
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

    return loss_hist


def train_model(
        model: XCloth, 
        X_train: torch.Tensor, 
        Y_train: Dict[str, torch.Tensor], 
        n_epoch: int = 20,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 5e-4,
        weight: List[float] = [1., 0.1, 1., 0.05],
        verbose: bool = False,
        plot: bool = False,
        reduction: str|None = "sum",
        params_path: str|None = None):
    
    """
    @param: X_train: N x B x C[8] x H[512] x W[512]
    @param: Y_train: Dict[N x B x P x C[1 (depth) | 3 (normal) | 3 (rgb)] x H x W]
    @param: weight: loss weight in the order of [depth, seg, norm, rgb]
    
    P: number of peelmaps
    N: number of batches
    B: batch size
    C: number of channels
    H: height
    W: width
    """

    loss_hist = []
    optim = optimizer(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9)
    
    for epoch in range(1, n_epoch + 1):
        loss = __epoch(
            model,
            X_train,
            Y_train,
            optim,
            weight,
            reduction
        )

        loss_hist.append(loss)
        if verbose: print(f"epoch {epoch}: loss: {loss}")

        if params_path is not None:
            model.save(params_path)
            
        # scheduler.step()

    if plot:
        plt.plot(range(1, n_epoch + 1), loss_hist)



        
    
    
