import random

import torch

from DA2Lite.compression.pruning.methods.criteria_base import CriteriaBase
from DA2Lite.core.layer_utils import _exclude_layer, get_layer_type


class Slimming(CriteriaBase):
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.check_point = -1
    
    def get_prune_idx(self, i_node, pruning_ratio=0.0): 
        is_check = False
        for idx, (name, layer) in enumerate(self.model.named_modules()):

            if idx <= self.check_point or _exclude_layer(layer):
                continue

            if i_node['id'] == hash(name):
                self.check_point = idx
                is_check = True
            
            if self.check_point + 1 == idx and get_layer_type(layer) == 'BN' and is_check == True:

                assert layer.num_features == i_node['layer'].out_channels
                gamma = layer.weight.clone()

                gamma_norm = torch.abs(gamma)

                n_to_prune = int(pruning_ratio*len(gamma))
                if n_to_prune == 0:
                    return []
                threshold = torch.kthvalue(gamma_norm, k=n_to_prune).values 

                indices = torch.nonzero(gamma_norm <= threshold).view(-1).tolist()        

                return indices    

