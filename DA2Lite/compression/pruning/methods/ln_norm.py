import numpy as np

import torch

from DA2Lite.compression.pruning.methods.criteria_base import CriteriaBase


class LNCriteria(CriteriaBase):
    def __init__(self, p):
        self.p = p

    def get_prune_idx(self, weights, pruning_ratio=0.0):
        if pruning_ratio <= 0: return []
        n = len(weights)

        #lN_norm = torch.norm( weights.view(n, -1), p=self.p, dim=1 )
        if self.p == 1:
            lN_norm = torch.sum(torch.abs(weights), dim=(1,2,3))
        else:
            lN_norm = torch.sum(torch.sqrt(weights**2), dim=(1,2,3))
         
        n_to_prune = int(pruning_ratio*n)
        if n_to_prune == 0:
            return []
        threshold = torch.kthvalue(lN_norm, k=n_to_prune).values 

        indices = torch.nonzero(lN_norm <= threshold).view(-1).tolist()        

        return indices
    

class L1Criteria(LNCriteria):
    def __init__(self, **kwargs):
        super(L1Criteria, self).__init__(p=1)

class L2Criteria(LNCriteria):
    def __init__(self, **kwargs):
        super(L2Criteria, self).__init__(p=2)