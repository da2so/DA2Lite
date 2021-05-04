import copy

import torch.nn as nn

from DA2Lite.core.layer_utils import _exclude_layer, get_layer_type, get_module_of_layer
from DA2Lite.compression.filter_decomposition import methods
from DA2Lite.core.log import get_logger

logger = get_logger(__name__)

cfg_to_method = {'Tucker': 'tucker_decomposition',
                'CP': 'cp_decomposition'}

class FilterDecomposition(object):
    def __init__(self,
                compress_cfg,
                model,
                device,
                **kwargs):
        
        self.origin_model = model
        self.device = device
        
        # configs for Filter Decomposition (FD)
        self.start_idx = compress_cfg.START_IDX
        self.fd_name = compress_cfg.DECOMPOSITION
        self.rank = compress_cfg.RANK

    def _get_method(self, method_name):
        try:
            method_func = getattr(methods, method_name)
        except:
            raise ValueError(f'Invalid FD method: {method_name}')
        
        return method_func

    def build(self):
        fd_method = self._get_method(cfg_to_method[self.fd_name])
        
        new_model = copy.deepcopy(self.origin_model)
        current_idx = 1

        for name, layer in new_model.named_modules():
            layer_type = get_layer_type(layer)

            if _exclude_layer(layer):
                continue

            if layer_type == 'Conv':
                if self.start_idx <= current_idx and layer.out_channels >= layer.in_channels:

                    module, last_name = get_module_of_layer(new_model, name)
                    module._modules[str(last_name)], ranks = fd_method(layer, self.rank, self.device)
                    
                    in_channels, out_channels = layer.in_channels, layer.out_channels
                    
                    logger.info(f'In, Out channels are decomposed: [{in_channels}, {out_channels}] -> [{ranks[1]}, {ranks[0]}] at "{name}" layer')
                    
                current_idx += 1

        return new_model