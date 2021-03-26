from abc import abstractclassmethod, ABC
import random

import torch


class CriteriaBase(ABC):

    @abstractclassmethod
    def __call__(self, weights, amount=0.0):
        raise NotImplementedError
    

class RandomCriteria(CriteriaBase):

    def __call__(self, weights, amount=0.0): 
        if amount<=0: return []

        n = len(weights)
        n_to_prune = int(amount*n)
        indices = random.sample( list( range(n) ), k=n_to_prune )
        return indices

class LNCriteria(CriteriaBase):
    def __init__(self, p):
        self.p = p

    def __call__(self, weights, amount=0.0):
        if amount<=0: return []
        n = len(weights)
        lN_norm = torch.norm( weights.view(n, -1), p=self.p, dim=1 )
        n_to_prune = int(amount*n)
        if n_to_prune == 0:
            return []
        threshold = torch.kthvalue(lN_norm, k=n_to_prune).values 
        indices = torch.nonzero(lN_norm <= threshold).view(-1).tolist()
        return indices

class L1Criteria(LNCriteria):
    def __init__(self):
        super(L1Criteria, self).__init__(p=1)

class L2Criteria(LNCriteria):
    def __init__(self):
        super(L2Criteria, self).__init__(p=2)