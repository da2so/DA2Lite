import torch.nn as nn

def _exclude_layer(layer):

    if isinstance(layer, nn.Sequential):
        return True
    if not 'torch.nn' in str(layer.__class__):
        return True

    return False
