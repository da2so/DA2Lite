import numpy as np
import copy
import random

import torch
import torch.nn as nn

from DA2Lite.compression.utils import _exclude_layer
from DA2Lite.compression.pruning.graph_generator import GraphGenerator
from DA2Lite.compression.pruning.utils import load_strategy, load_criteria

class Pruner(object):
    def __init__(self,
                compress_cfg,
                model,
                device,
                **kwargs):

        self.pruning_config = compress_cfg

        self.criteria_args = None
        if 'CRITERIA_ARGS' in self.pruning_config:
            self.criteria_args = self.pruning_config.CRITERIA_ARGS
        self.device = device        

        self.model = model
        
        self.train_loader = kwargs['train_loader']
        self.img_shape = kwargs['cfg'].DATASET.IMG_SHAPE
        self.save_dir = kwargs['cfg'].SAVE_DIR
    
    def prune(self):
        
        pruning_info = []
        node_graph, group_set = GraphGenerator(self.model, 
                                                self.img_shape, 
                                                self.save_dir).build()
            
        group_to_ratio = load_strategy(self.pruning_config.STRATEGY,
                                    group_set,
                                    self.pruning_config.STRATEGY_ARGS.PRUNING_RATIO
                                    ).build()

        pruning_method = load_criteria(criteria_name=self.pruning_config.CRITERIA, 
                                    critera_args=self.criteria_args)
    
        for idx, key in enumerate(node_graph.keys()):

            i_node = node_graph[key]
            if isinstance(i_node['layer'], nn.Conv2d):
                weight_copy = i_node['layer'].weight
                prune_idx = pruning_method.get_prune_idx(weights=weight_copy, 
                                                        pruning_ratio=group_to_ratio[i_node['group']])
                
                total_channel = i_node['layer'].out_channels
                remaining_channel = total_channel - len(prune_idx)
                i_node['prune_idx'] = prune_idx

                pruning_info.append(f'layer index: {idx:4d} \t total channel: {total_channel:4d} \t remaining channel: {remaining_channel:4d}')
            
            elif isinstance(i_node['layer'], nn.BatchNorm2d):
                down_key = 1

                while 'prune_idx' not in node_graph[key - down_key]:
                    down_key += 1
                
                i_node['prune_idx'] = node_graph[key - down_key]['prune_idx']

        new_model = copy.deepcopy(self.model)

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
                layer.weight.data = w[keep_idx, :, :, :].clone()
                if layer.bias is not None:
                        layer.bias = nn.Parameter(layer.bias.data.clone()[keep_idx])
                
            elif isinstance(layer, nn.BatchNorm2d):
                prune_idx = node_graph[i]['prune_idx']
                keep_idx = list(set(range(layer.num_features)) - set(prune_idx))

                layer.running_mean.data = layer.running_mean.data[keep_idx].clone()
                layer.running_var.data = layer.running_var.data[keep_idx].clone()
                if layer.affine:
                    layer.weight.data = layer.weight.data[keep_idx].clone()
                    layer.bias.data = layer.bias.data[keep_idx].clone()


            elif isinstance(layer, nn.Linear):
                # [error] operate code when having only one linear layer
                prev_prune_idx = self.get_prev_prune_idx(node_graph[i]['input_conv_layers'], node_graph)
                keep_idx = list(set(range(layer.in_features)) - set(prev_prune_idx))
                layer.weight.data = layer.weight.data[:, keep_idx].clone()


            i += 1

        return new_model, pruning_info


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


            
    def build(self):

        best_model = {'acc': -1., 'model': None, 'idx': -1}
        for idx in range(self.pruning_method.num_candidates):

            new_model, pruning_info = self.prune()

            best_model = self.pruning_method.get_model(pruned_model=new_model,
                                                    pruning_info=pruning_info,
                                                    best_model=best_model,
                                                    train_loader=self.train_loader,
                                                    device=self.device,
                                                    idx=idx)

        if idx >= 2:
            print(f'\nThe best candidate is {best_model["index"]}-th prunned model (Train Acc: {best_model["acc"]})')
        else:
            print(f'\nThe prunned model (Train Acc: {best_model["acc"]})')

        [print(line) for line in best_model['pruning_info']]
        
        return best_model['model']