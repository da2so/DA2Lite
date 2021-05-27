import numpy as np
from collections import defaultdict

import torch
from kmeans_pytorch import kmeans

from DA2Lite.compression.clustering.methods.base import ClusteringBase
from DA2Lite.core.layer_utils import _exclude_layer, get_layer_type


class NewClustering(ClusteringBase):

    def get_cluster_idx(self, i_node, clustering_ratio): 

        weights = i_node['layer'].weight.clone()
        
        if clustering_ratio <= 0: return []
        n = len(weights)

        out_channels = weights.size()[0]
        
        weights = weights.view(-1, out_channels)
        
        sim_dict = {}

        i_u, i_s, i_vh = torch.svd(weights)
 
        n_to_cluster = n - int(clustering_ratio * n)
        print(i_u.size())
        print(i_s.size())
        print(i_vh.size())
        i_sv = torch.matmul(i_vh, torch.diag(i_s))

        # kmeans
        cluster_ids_x, cluster_centers = kmeans(
            X=i_sv, num_clusters=n_to_cluster, distance='cosine'
        )
        
        group_to_mindist = 1e6 * np.ones(n_to_cluster)
        group_to_id = np.ones(n_to_cluster)
        for idx in range(len(cluster_ids_x)):
            i_cluster = cluster_ids_x[idx]

            i_center = cluster_centers[i_cluster, :]
            i_val = i_sv[idx, :]
            dist = torch.dist(i_val, i_center)

            if group_to_mindist[i_cluster] > dist:
                group_to_mindist[i_cluster] = dist
                group_to_id[i_cluster] = idx
            

        indices = {x for x in range(n)} - set(group_to_id)

        return list(indices)

                
