import torch
import torch.nn as nn

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
    @param: X_train: N x B x C[8] x H[512] x W[512]
    @param: Y_train: N x B x C[1 (depth) + 4 (normal) + 3 (rgb)] x H x W
    
    N: number of batches
    B: batch size
    C: number of channels
    H: height
    W: width
    """

    loss = 0

    N = X_train.shape[0]
    for i in range(N):
        result = model(X_train[i, :, :4], X_train[i, :, 4:])

        loss_d = (result["Depth"] - Y_train[i, :, 0]).abs().sum()
        loss_norm = ((result["Norm"] - Y_train[i, :, 1:4*4+1])**2).sum()
        loss_rgb = (result["RGB"] - Y_train[i, :, 4*4+1:]).abs().sum()

    loss += weight[0]*loss_d + weight[2]*loss_norm + weight[3]*loss_rgb

    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()

    torch.cuda.empty_cache()

    return loss


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
    @param: Y_train: N x B x C[1 (depth) + 4 (normal) + 3 (rgb)] x H x W
    @param: weight: loss weight in the order of [depth, seg, norm, rgb]
    
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

        loss_hist.append(loss.item())
        if verbose: print(f"epoch {epoch}: loss: {loss}")

    if plot:
        plt.plot(range(n_epoch), loss_hist)



        
    
    
