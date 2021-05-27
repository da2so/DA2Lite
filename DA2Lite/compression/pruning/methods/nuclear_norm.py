import numpy as np
from collections import defaultdict

import torch

from DA2Lite.compression.pruning.methods.criteria_base import CriteriaBase

class NuclearNorm(CriteriaBase):
    def __init__(self, **kwargs):
        pass
            
    def get_prune_idx(self, i_node, pruning_ratio, **kwargs): 

        f_maps = kwargs['f_maps']
        device = kwargs['device']
        
        if pruning_ratio <= 0: return []
        
        
        size_of_fmaps = f_maps.size()
        num_samples = size_of_fmaps[0]
        num_features = size_of_fmaps[1]
        h = size_of_fmaps[2]
        w = size_of_fmaps[3]
        
        print(num_features)
        singular_values = torch.zeros((num_features)).to(device)
        tt =  torch.zeros((num_features)).to(device)
        
        if num_samples > h*w:
            beta = h * w / (num_samples)
        else:
            beta = num_samples / (h * w)

        w_beta = 0.56 * beta ** 3 - 0.95 * beta **2 + 1.82 * beta + 1.43 

        for i in range(num_features):
            i_sample = f_maps[:, i, :, :]

            i_sample = i_sample.view(num_samples, -1)
            # print(i_sample.size())
            i_u, i_s, i_vh = torch.svd(i_sample)
            
            i_s_median = torch.median(i_s)
            threshold_sigma = i_s_median * w_beta
            #print(len(i_s))
            #threshold_sigma = torch.kthvalue(i_s, k= int( len(i_s) * 0.25) ).values 
            idx = torch.nonzero(i_s <= threshold_sigma).view(-1).tolist()        
            i_s_tmp = i_s[idx]
            tt[i] = i_s_tmp.sum()
            singular_values[i] = i_s.sum()
        
        n_to_prune = int(pruning_ratio*num_features)
        

        print(f'n_to_prune: {n_to_prune}')
        threshold = torch.kthvalue(tt, k=n_to_prune).values 
        print(threshold)
        removed_indices = torch.nonzero(tt <= threshold).view(-1).tolist()        
        print(len(removed_indices))
        print('\n\n')
        return removed_indices

                