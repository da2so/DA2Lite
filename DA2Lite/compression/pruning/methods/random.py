import random

import torch

from DA2Lite.compression.pruning.methods.common import CriteriaBase



class RandomCriteria(CriteriaBase):

    def get_prune_idx(self, weights, amount=0.0): 
        if amount<=0: return []

        n = len(weights)
        n_to_prune = int(amount*n)
        indices = random.sample( list( range(n) ), k=n_to_prune )
        return indices
