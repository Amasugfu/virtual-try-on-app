import torch
import torch.nn as nn
import numpy as np

from matplotlib import pyplot as plt
from typing import Type, List

from ..production import XCloth


def __epoch(
        model: XCloth, 
        X_train: torch.Tensor, 
        Y_train: torch.Tensor, 
        optimizer: torch.optim.Optimizer,
        weight: List[float]):
    """
    @param: X_train: N x B x C[4 (image) + P (peelmaps)] x H[512] x W[512]
    @param: Y_train: N x B x P x C[1 (depth) + 3 (normal) + 3 (rgb)] x H x W
    
    P: number of peelmaps
    N: number of batches
    B: batch size
    C: number of channels
    H: height
    W: width
    """

    N = X_train.shape[0]
    loss_hist = []

    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()
    
    for i in range(N):
        optimizer.zero_grad()
        
        result = model(X_train[i, :, :4], X_train[i, :, 4:])
        
        loss_d = 0
        loss_norm = 0
        loss_rgb = 0

        # sum of loss of peelmap layers
        for j in range(model.n_peelmaps):
            loss_d += loss_l1(result["Depth"][j].squeeze(), Y_train[i, :, j, 0])
            loss_norm += loss_l2(result["Norm"][j], Y_train[i, :, j, 1:4])
            loss_rgb += loss_l1(result["RGB"][j], Y_train[i, :, j, 4:])

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
        Y_train: torch.Tensor, 
        n_epoch: int = 20,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        weight: List[float] = [1., 0.1, 1., 0.05],
        verbose: bool = False,
        plot: bool = False):
    
    """
    @param: X_train: N x B x C[8] x H[512] x W[512]
    @param: Y_train: N x B x P x C[1 (depth) + 3 (normal) + 3 (rgb)] x H x W
    @param: weight: loss weight in the order of [depth, seg, norm, rgb]
    
    P: number of peelmaps
    N: number of batches
    B: batch size
    C: number of channels
    H: height
    W: width
    """

    loss_hist = []
    optim = optimizer(model.parameters())
    
    for epoch in range(n_epoch):
        loss = __epoch(
            model,
            X_train,
            Y_train,
            optim,
            weight
        )

        loss_hist.append(loss)
        if verbose: print(f"epoch {epoch}: loss: {loss}")

    if plot:
        plt.plot(range(n_epoch), loss_hist)



        
    
    
