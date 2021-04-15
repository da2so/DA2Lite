import torch.nn as nn

def parse_layer_name(name):
    
    name_split = name.split('.')
    name_list = []
    
    for i_name in name_split:
        if i_name.isnumeric():
            name_list.append(int(i_name))
        else:
            name_list.append(i_name)
    
    return name_list

def get_module_of_layer(new_model, l_name):

    l_name_list = parse_layer_name(l_name)
    last_name = l_name_list.pop()

    module = new_model

    for i_name in l_name_list:
        if isinstance(i_name, str):
            module = getattr(module, i_name)
        elif isinstance(i_name, int):
            module = module[i_name]
        else:
            raise ValueError(f'layer name is unvalid: {i_name}')
    
    return module, last_name
            

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
    