import numpy as np

import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker

from DA2Lite.compression.filter_decomposition.methods.vmbf import EVBMF

def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """

    weights = layer.weight.data.cpu().numpy()
    unfold_0 = tl.base.unfold(weights, 0) 
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = EVBMF(unfold_0)
    _, diag_1, _, _ = EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]

    return ranks


def tucker_decomposition(layer, rank, device):

    if isinstance(rank, int):
        ranks = rank
    elif rank == 'VBMF':
        ranks = estimate_ranks(layer)

    core, [last, first] = \
        partial_tucker(layer.weight.data.cpu().numpy(),
                        modes=[0, 1],
                        rank=ranks,
                        init='svd')
    
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                out_channels=first.shape[1],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                dilation=layer.dilation,
                                bias=False)

    # A regular 2D convolution layer with R3 input channels 
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],
                                out_channels=core.shape[0],
                                kernel_size=layer.kernel_size,
                                stride=layer.stride,
                                padding=layer.padding,
                                dilation=layer.dilation,
                                bias=False)

    # A pointwise convolution that increases the channels from R4 to T


    if layer.bias is not None:
        last_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                    out_channels=last.shape[0],
                                    kernel_size=1, 
                                    stride=1,
                                    padding=0, 
                                    dilation=layer.dilation, 
                                    bias=True)
        last_layer.bias.data = layer.bias.data
    else:
        last_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                    out_channels=last.shape[0],
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    dilation=layer.dilation,
                                    bias=False)
    

    first_tensor = torch.from_numpy(first.copy()).to(device)
    last_tensor = torch.from_numpy(last.copy()).to(device)
    core_tensor = torch.from_numpy(core.copy()).to(device)

    first_layer.weight.data = torch.transpose(first_tensor, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last_tensor.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core_tensor

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers), ranks