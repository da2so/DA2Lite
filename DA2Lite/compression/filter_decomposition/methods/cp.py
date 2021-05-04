"""
"Tensor rank learning in CP decomposition via convolutional neural network"
"""
  
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorly as tl
from tensorly.decomposition import parafac

from DA2Lite.compression.filter_decomposition.methods.vmbf import EVBMF

def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """
    weights = layer.weight.data.cpu().numpy()
    mode3 = tl.base.unfold(weights, 0)
    mode4 = tl.base.unfold(weights, 1)
    
    diag_0 = EVBMF(mode3)
    diag_1 = EVBMF(mode4)

    # round to multiples of 16
    ranks = int(np.ceil(max([diag_0.shape[0], diag_1.shape[0]]) / 16) * 16)

    return ranks

def cp_decomposition(layer, rank, device):

    if isinstance(rank, int):
        ranks = rank
    elif rank == 'VBMF':
        ranks = estimate_ranks(layer)

    last, first, vertical, horizontal = parafac(layer.weight.data.cpu().numpy(),
                                        rank=ranks,
                                        init='random')
    
    s_to_r_layer = nn.Conv2d(in_channels=first.shape[0],
                            out_channels=first.shape[1],
                            kernel_size=1,
                            padding=0,
                            bias=False)

    r_to_r_layer = nn.Conv2d(in_channels=ranks,
                            out_channels=ranks,
                            kernel_size=vertical.shape[0],
                            stride=layer.stride,
                            padding=layer.padding,
                            dilation=layer.dilation,
                            groups=ranks,
                            bias=False)
                                       
    if layer.bias is not None:
        r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                    out_channels=last.shape[0],
                                    kernel_size=1, 
                                    stride=1,
                                    padding=0, 
                                    dilation=layer.dilation, 
                                    bias=True)
        r_to_t_layer.bias.data = layer.bias.data
    else:
        r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                    out_channels=last.shape[0],
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    dilation=layer.dilation,
                                    bias=False)

    sr = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    rt = last.unsqueeze_(-1).unsqueeze_(-1)
    rr = torch.stack([vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(0, i, 1) for i in range(rank)]).unsqueeze_(1)

    s_to_r_layer.weight.data = sr 
    r_to_t_layer.weight.data = rt
    r_to_r_layer.weight.data = rr

    new_layers = [s_to_r_layer, r_to_r_layer, r_to_t_layer]
    return new_layers