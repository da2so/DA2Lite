import copy

import torch.nn as nn

from DA2Lite.compression.utils import _exclude_layer
from DA2Lite.compression.filter_decomposition import methods

cfg_to_method = {'Tucker': 'tucker_decomposition'}

class FilterDecomposition(object):
    def __init__(self,
                compress_cfg,
                model,
                device,
                **kwargs):
        
        self.origin_model = model
        self.device = device

        self.start_idx = compress_cfg.START_IDX
        self.fd_name = compress_cfg.NAME
        self.rank = compress_cfg.RANK
    
    def _get_method(self, method_name):
        
        try:
            method_func = getattr(methods, method_name)
        except:
            raise ValueError(f'Invalid FD method: {method_name}')
        
        return method_func

    def parse_layer_name(self, name):
        
        name_split = name.split('.')
        name_list = []
        
        for i_name in name_split:
            if i_name.isnumeric():
                name_list.append(int(i_name))
            else:
                name_list.append(i_name)
        
        return name_list

    def get_module_of_layer(self, new_model, l_name):

        l_name_list = l_name = self.parse_layer_name(l_name)
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
                
    def build(self):
        
        fd_method = self._get_method(cfg_to_method[self.fd_name])
        
        new_model = copy.deepcopy(self.origin_model)
        
        current_idx = 1

        for name, layer in new_model.named_modules():
            if _exclude_layer(layer):
                continue

            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                if self.start_idx <= current_idx:
                    module, last_name = self.get_module_of_layer(new_model, name)
                    module._modules[str(last_name)] = fd_method(layer, self.rank, self.device)
                    
                current_idx += 1
            
        from torchsummary import summary
        summary(new_model, (3,32,32))

        return new_model