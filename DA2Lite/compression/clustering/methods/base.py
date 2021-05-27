from abc import abstractclassmethod, ABC
import random

import torch

class ClusteringBase(ABC):

    @abstractclassmethod
    def get_cluster_idx(self, i_node, clustering_ratio):
        raise NotImplementedError
