import torch.nn as nn

def _exclude_layer(layer):

    if isinstance(layer, nn.Sequential):
        return True
    if not 'torch.nn' in str(layer.__class__):
        return True

    return False



def get_layer_type(layer):
    
    if isinstance(layer, nn.modules.conv.Conv2d) and layer.groups <= 1:
        return 'Conv'
    elif isinstance(layer, nn.modules.conv.Conv2d) and layer.groups > 1:
        return 'GroupConv'
    elif isinstance(layer, nn.modules.batchnorm.BatchNorm2d):
        return 'BN'
    elif isinstance(layer, nn.modules.linear.Linear):
        return 'Linear'
    