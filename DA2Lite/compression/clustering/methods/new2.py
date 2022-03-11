import numpy as np
from collections import defaultdict

import torch
from kmeans_pytorch import kmeans

from DA2Lite.compression.clustering.methods.base import ClusteringBase
from DA2Lite.core.layer_utils import _exclude_layer, get_layer_type


class New2Clustering(ClusteringBase):
    def __init__(self, **kwargs):
        
        self.threshold = kwargs['cluster_args'].THRESHOLD
    
    def get_cluster_idx(self, i_node, f_maps, device): 
        
        
        B, C, _, _ = f_maps.size()
        # print(f'Channels: {C}')
        
        f_maps = f_maps.view(B, C, -1)
        # per channel SVD
        f_hats = []
        for c in range(C):
            f = f_maps[:, c, :]
            u, s, _ = torch.svd(f)
            # UxS
            f_hat = u @ torch.diag(s)
            f_hats.append(f_hat)
        # calulate simmat
        simmat = torch.zeros((C, C))
        for i in range(C):
            for j in range(i + 1, C):
                sim = abs(
                    torch.nn.CosineSimilarity(dim=0)(f_hats[i].view(-1), f_hats[j].view(-1))
                )
                simmat[i, j] = simmat[j, i] = sim
        # print(simmat)
        # remove with threshold

        
        remove_indices = set()
        while simmat.max() > self.threshold:
            argmax = torch.argmax(simmat)
            argmax_row = argmax // C
            argmax_col = argmax % C

            for i_col in range(C):
                
                if simmat[argmax_row][i_col] > self.threshold:
                    
                    simmat[argmax_row][i_col] = 0.0
                    simmat[i_col][argmax_row] = 0.0
                    remove_indices.add(i_col)

        # print(remove_indices)
        return list(remove_indices)

        """
        size_of_fmaps = f_maps.size()
        num_samples = size_of_fmaps[0]
        num_features = size_of_fmaps[1]
        h = size_of_fmaps[2]
        w = size_of_fmaps[3]
        
        singular_values = torch.zeros((num_features)).to(device)
        tt = torch.zeros((num_features)).to(device)

        beta = num_samples / (h * w)

        print(h*w)
        w_beta = 0.56 * beta ** 3 - 0.95 * beta **2 + 1.82 * beta + 1.43 
        
        for i in range(num_features):
            i_sample = f_maps[:, i, :, :]

            i_sample = i_sample.view(num_samples, -1)
            i_u, i_s, i_vh = torch.svd(i_sample)
            
            i_s_median = torch.median(i_s)
            threshold = i_s_median * w_beta
            indices = torch.nonzero(i_s >= threshold).view(-1).tolist()        
            i_s_tmp = i_s[indices]
            tt[i] = i_s_tmp.sum()
            singular_values[i] = i_s.sum()
        
        print(tt / num_features)
        singular_values /= num_features
        print(singular_values)
        
        print('\n\n')


        return None

                
        """