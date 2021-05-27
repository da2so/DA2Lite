from abc import ABC, abstractmethod
from collections import OrderedDict
import random
import time
import numpy as np


#vgg_16_bn = [0.4] + [0.5] + [0.6] + [0.5] + [0.7] + [0.6]*2 + [0.8]*3 + [0.9]*3
vgg_16_bn = [0.3] + [0.4] + [0.5] + [0.6] + [0.6] + [0.7]*2 + [0.7]*3 + [0.7]*3

class StragtegyBase(ABC):
    def __init__(self, group_set):
        self.group_set = group_set
        np.random.seed(int(time.time()))

    def build(self):
        group_to_ratio = OrderedDict()
        for idx, i_group in enumerate(self.group_set):
            random_rate = self._get_ratio()
            group_to_ratio[i_group] = vgg_16_bn[idx]


        return group_to_ratio
    
    def _get_ratio(self):
        return 0.

class MinMaxStrategy(StragtegyBase):
    def __init__(self, group_set, pruning_ratio):
        super().__init__(group_set)
        
        assert len(pruning_ratio) == 2
        assert pruning_ratio[0] < pruning_ratio[1]
        self.min_ratio = pruning_ratio[0]
        self.max_ratio = pruning_ratio[1]
        
    def _get_ratio(self):
        return  (self.max_ratio - self.min_ratio) * (np.random.rand(1)) + self.min_ratio

class RandomStrategy(StragtegyBase):
    def __init__(self, group_set, pruning_ratio):
        super().__init__(group_set)

    def _get_ratio(self):
        return np.random.rand(1)


class StaticStrategy(StragtegyBase):
    def __init__(self, group_set, pruning_ratio):
        super().__init__(group_set)

        assert len(pruning_ratio) == 1
        self.pruning_ratio = pruning_ratio[0]
    
    def _get_ratio(self):
        return self.pruning_ratio