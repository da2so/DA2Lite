import numpy as np
import copy
import random

import torch.nn as nn

from DA2Lite.compression.pruning.graph_generator import GraphGenerator
from DA2Lite.compression.pruning.utils import load_strategy, load_criteria

class EagleEye(object):
    def __init__(self,
                cfg_util,
                model,
                train_loader,
                test_loader,
                device):

        cfg = cfg_util.cfg
        self.pruning_config = cfg.PRUNING
        self.img_shape = cfg.DATASET.IMG_SHAPE
        self.save_dir = cfg.SAVE_DIR
        self.min_rate = 0.0
        self.max_rate = 0.5
        self.num_candidates = 10

        self.model = model
        

    def build(self):

        pruned_model_list = []
        val_acc_list = []

        for i in range(self.num_candidates):

            #channel_config = get_strategy(self.model, self.min_rate, self.max_rate)  
            node_graph, group_set = GraphGenerator(self.model, 
                                                self.img_shape, 
                                                self.save_dir).build()
            
            group_to_ratio = load_strategy(self.pruning_config.STRATEGY.NAME,
                                        group_set,
                                        self.pruning_config.STRATEGY.PRUNING_RATIO
                                        ).build()

            criteria = load_criteria(self.pruning_config.CRITERIA.NAME)
        
            for idx, key in enumerate(node_graph.keys()):

                i_node = node_graph[key]
                if isinstance(i_node['layer'], nn.Conv2d):
                    weight_copy = i_node['layer'].weight
                    prune_idx = criteria(weight_copy, group_to_ratio[i_node['group']])
                    
                    total_channel = i_node['layer'].out_channels
                    remaining_channel = total_channel - len(prune_idx)
                    i_node['prune_idx'] = prune_idx

                    print(f'layer index: {idx} \t total channel: {total_channel} \t remaining channel: {remaining_channel}')
                
                elif isinstance(i_node['layer'], nn.BatchNorm2d):
                    down_key = 1

                    while 'prune_idx' not in node_graph[key - down_key]:
                        down_key += 1
                    
                    i_node['prune_idx'] = node_graph[key - down_key]['prune_idx']

                #print(i_node['group'])
            
            new_model = copy.deepcopy(self.model)

            
            idx = 0

            
            def _exclude_layer(layer):

                if isinstance(layer, nn.Sequential):
                    return True
                if not 'torch.nn' in str(layer.__class__):
                    return True

                return False
            i = 0
            for idx, layer in enumerate(new_model.modules()):
                if idx == 0 or _exclude_layer(layer):
                    continue
                if isinstance(layer, nn.Conv2d):

                    if 'input_conv_layers' in node_graph[i]:
                        prev_prune_idx = self.get_prev_prune_idx(node_graph[i]['input_conv_layers'], node_graph)
                    else:
                        prev_prune_idx = []
                    prune_idx = node_graph[i]['prune_idx']

                    keep_prev_idx = list(set(range(layer.in_channels)) - set(prev_prune_idx))
                    keep_idx = list(set(range(layer.out_channels)) - set(prune_idx))

                    w = layer.weight.data[:, keep_prev_idx, :, :].clone()
                    w = w[keep_idx, :, :, :].clone()

                    layer.weight.data = w.clone()

                elif isinstance(layer, nn.BatchNorm2d):
                    prune_idx = node_graph[i]['prune_idx']
                    keep_idx = list(set(range(layer.num_features)) - set(prune_idx))

                    layer.running_mean.data = layer.running_mean.data[keep_idx].clone()
                    layer.running_var.data = layer.running_var.data[keep_idx].clone()
                    if layer.affine:
                        layer.weight.data = layer.weight.data[keep_idx].clone()
                        layer.bias.data = layer.bias.data[keep_idx].clone()


                elif isinstance(layer, nn.Linear):

                    prev_prune_idx = self.get_prev_prune_idx(node_graph[i]['input_conv_layers'], node_graph)
                    keep_idx = list(set(range(layer.in_features)) - set(prev_prune_idx))
                    layer.weight.data = layer.weight.data[:, keep_idx].clone()

                i += 1

            return new_model

    
    def get_prev_prune_idx(self, in_layer, node_graph):
        tmp_in_layer = in_layer
        prev_prune_idx = set()
        for key in node_graph.keys():
            if 'name' not in node_graph[key]:
                continue
            if node_graph[key]['name'] in in_layer:
                prev_prune_idx.update(node_graph[key]['prune_idx'])
                tmp_in_layer.remove(node_graph[key]['name'])

            
            if len(tmp_in_layer) == 0:
                num_prev_prune = len(node_graph[key]['prune_idx'])
                break
    
        indices = random.sample( list( range(len(prev_prune_idx)) ), k=len(prev_prune_idx)-num_prev_prune )
        prev_prune_idx = list(prev_prune_idx)
        for i in sorted(indices, reverse=True):
            del prev_prune_idx[i]
            
        return prev_prune_idx
