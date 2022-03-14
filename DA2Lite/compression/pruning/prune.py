import numpy as np
import copy
import random
from collections import Counter, OrderedDict, defaultdict

import torch
import torch.nn as nn

from DA2Lite.core.layer_utils import _exclude_layer, get_layer_type
from DA2Lite.core.graph_generator import GraphGenerator
from DA2Lite.compression.pruning.utils import load_strategy, load_criteria
from DA2Lite.core.log import get_logger

logger = get_logger(__name__)

act_based_pruning = {'NuclearNorm'}

class Pruner(object):
    def __init__(self,
                compress_cfg,
                model,
                device,
                **kwargs):            
        self.device = device        
        self.model = model
        self.train_loader = kwargs['train_loader']
        self.img_shape = kwargs['cfg'].DATASET.IMG_SHAPE
        self.save_dir = kwargs['cfg'].SAVE_DIR
    
        self.pruning_cfg = compress_cfg

        self.criteria_args = None
        if 'CRITERIA_ARGS' in self.pruning_cfg:
            self.criteria_args = self.pruning_cfg.CRITERIA_ARGS

        #get network node graph using onnx framework
        graph_model = copy.deepcopy(model)
        self.node_graph, self.group_set = GraphGenerator(model=graph_model, 
                                                        img_shape=self.img_shape, 
                                                        save_dir=self.save_dir
                                                        ).build()

        # for key, val in self.node_graph.items():
        #     print(key)
        #     print(val)
        del graph_model
        self.criteria_class = load_criteria(criteria_name=self.pruning_cfg.CRITERIA, 
                                            criteria_args=self.criteria_args,
                                            model=self.model)

        self.activations = defaultdict()
        self.conv2target_conv = defaultdict()
        self.hook_layers = []

        if self.pruning_cfg.CRITERIA in act_based_pruning:
            self.set_hooking()
        
    def set_prune_idx(self, group_to_ratio, node_graph):
        pruning_info = []
        group_frequency = dict()
        for idx, key in enumerate(node_graph.keys()):

            i_node = node_graph[key]
            layer_type = get_layer_type(i_node['layer'])

            if layer_type == 'Conv':
                
                if self.pruning_cfg.CRITERIA not in act_based_pruning:
                    prune_idx = self.criteria_class.get_prune_idx(i_node=i_node, 
                                                                pruning_ratio=group_to_ratio[i_node['group']])
                    
                    i_node['prune_idx'] = prune_idx

                    # For integrating prune indexes
                    if i_node['group'] not in group_frequency:
                        group_frequency[i_node['group']] = [key]
                    else:
                        group_frequency[i_node['group']].append(key)
                
                else:
                    if i_node['group'] in group_frequency:
                        prune_idx = group_frequency[i_node['group']]
                    else:
                        f_maps = self.activations[self.conv2target_conv[i_node['name']]]
                        prune_idx = self.criteria_class.get_prune_idx(i_node=i_node,
                                                                    pruning_ratio=group_to_ratio[i_node['group']],
                                                                    f_maps=f_maps,
                                                                    device=self.device)
                        group_frequency[i_node['group']] = prune_idx

                    i_node['prune_idx'] = prune_idx

            elif layer_type == 'GroupConv':
                down_key = -1
                while 'prune_idx' not in node_graph[key + down_key]:
                    down_key -= 1
                
                i_node['prune_idx'] = node_graph[key + down_key]['prune_idx']

        if self.pruning_cfg.CRITERIA not in act_based_pruning:
            node_graph = self._integrate_prune_idx(node_graph=node_graph,
                                                group_frequency=group_frequency)

        return node_graph

    def _integrate_prune_idx(self, node_graph, group_frequency):
        
        for group_num in group_frequency.keys():
            if len(group_frequency[group_num]) >= 2:
                total_prune_idx = []
                for key in group_frequency[group_num]:
                    total_prune_idx.extend(node_graph[key]['prune_idx'])
                
                limit_num_idx = len(node_graph[key]['prune_idx'])
                count_prune_idx = dict(Counter(total_prune_idx).most_common(limit_num_idx))
                cutted_prune_idx = []
                
                # only prune filters if prune indexes of layers are duplicated
                for prune_idx, num in count_prune_idx.items():
                    if num >= 2:
                        cutted_prune_idx.append(prune_idx)

                # redefine prune indexes for same groups
                for key in group_frequency[group_num]:
                    node_graph[key]['prune_idx'] = cutted_prune_idx
   
        return node_graph
                
    def prune(self, node_graph):
        group_to_ratio = load_strategy(strategy_name=self.pruning_cfg.STRATEGY,
                                    group_set=self.group_set,
                                    pruning_ratio=self.pruning_cfg.STRATEGY_ARGS.PRUNING_RATIO).build()

        # print(group_to_ratio)
        #group_to_ratio = 
        node_graph = self.set_prune_idx(group_to_ratio, node_graph)

        new_model = copy.deepcopy(self.model)

        i = 0
        pruning_info = []
        for idx, data in enumerate(new_model.named_modules()):
            name, layer = data
            if idx == 0 or _exclude_layer(layer):
                continue
            
            layer_type = get_layer_type(layer)
            if layer_type == 'Conv':
                prev_prune_idx = []
                if 'input_convs' in node_graph[i]:
                    prev_prune_idx = self.get_prev_prune_idx(node_graph=node_graph,
                                                            index=i)
                prune_idx = node_graph[i]['prune_idx']

                keep_prev_idx = list(set(range(layer.in_channels)) - set(prev_prune_idx))
                keep_idx = list(set(range(layer.out_channels)) - set(prune_idx))

                w = layer.weight.data[:, keep_prev_idx, :, :].clone()
                layer.weight.data = w[keep_idx, :, :, :].clone()
                if layer.bias is not None:
                        layer.bias.data = layer.bias.data[keep_idx].clone()
                
                
                pruning_info.append(f'Out channels are pruned: [{layer.out_channels:4d}] -> [{len(keep_idx):4d}] at "{name}" layer')
                layer.out_channels = len(keep_idx)
                layer.in_channels = len(keep_prev_idx)

            elif layer_type == 'GroupConv':
                prune_idx = node_graph[i]['prune_idx']

                keep_idx = list(set(range(layer.out_channels)) - set(prune_idx))

                layer.weight.data = layer.weight.data[keep_idx, :, :, :].clone()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[keep_idx].clone()

                pruning_info.append(f'Out channels are pruned: [{layer.out_channels:4d}] -> [{len(keep_idx):4d}] at "{name}" layer')
                layer.out_channels = len(keep_idx)
                layer.in_channels = len(keep_idx)
                layer.groups = len(keep_idx)

            elif layer_type == 'BN':
                prev_prune_idx = self.get_prev_prune_idx(node_graph=node_graph,
                                                        index=i)
                keep_idx = list(set(range(layer.num_features)) - set(prev_prune_idx))

                layer.running_mean.data = layer.running_mean.data[keep_idx].clone()
                layer.running_var.data = layer.running_var.data[keep_idx].clone()
                if layer.affine:
                    layer.weight.data = layer.weight.data[keep_idx].clone()
                    layer.bias.data = layer.bias.data[keep_idx].clone()
                
                pruning_info.append(f'Out channels are pruned: [{layer.num_features:4d}] -> [{len(keep_idx):4d}] at "{name}" layer')
                layer.num_features = len(keep_idx)

            elif layer_type == 'Linear':
                if 'input_convs' in node_graph[i]:
                    prev_prune_idx = self.get_prev_prune_idx(node_graph=node_graph,
                                                            index=i)
                    keep_idx = list(set(range(layer.in_features)) - set(prev_prune_idx))
                    
                    layer.weight.data = layer.weight.data[:, keep_idx].clone()

                    layer.in_features = len(keep_idx)

            i += 1

        return new_model, pruning_info, node_graph


    def get_prev_prune_idx(self, node_graph, index):
        in_layers_name = node_graph[index]['input_convs']
        if in_layers_name == None:
            return []
        
        in_layer_num = len(in_layers_name)
        find_in_layer_num = 0
        tmp_in_layers = defaultdict()
        in_layers = []
        for key in node_graph.keys():
            if in_layer_num == find_in_layer_num:
                break
            if 'name' not in node_graph[key]:
                continue
            
            if node_graph[key]['name'] in in_layers_name:
                find_in_layer_num += 1

                l_idx = in_layers_name.index(node_graph[key]['name'])
                tmp_in_layers[l_idx] = node_graph[key]
        for idx in range(in_layer_num):
            in_layers.append(tmp_in_layers[idx])
        
        concat_in_layers = []
        if 'concat_op' in node_graph[index]:
            concat_in_layers = node_graph[index]['concat_op']

            
        prev_prune_idx = list()
        total_channels = 0
        for idx, i_layer in enumerate(in_layers):
            if idx == 0:
                prev_prune_idx.extend(i_layer['prune_idx'])
                total_channels += i_layer['layer'].out_channels
            else:
                if i_layer['name'] in concat_in_layers:
                    for i_prune in i_layer['prune_idx']:
                        prev_prune_idx.extend([i_prune + total_channels]) # For concat operation
                    total_channels += i_layer['layer'].out_channels
                else:
                    break
        return prev_prune_idx
            
    def build(self):
        best_model = {'acc': -1., 'model': None, 'idx': -1}
       
        for idx in range(self.criteria_class.num_candidates):
            node_graph = copy.deepcopy(self.node_graph)

            new_model, pruning_info, node_graph = self.prune(node_graph)

            best_model = self.criteria_class.get_model(pruned_model=new_model,
                                                    pruning_info=pruning_info,
                                                    node_graph=node_graph,
                                                    best_model=best_model,
                                                    train_loader=self.train_loader,
                                                    device=self.device,
                                                    idx=idx)

        if idx >= 2:
            logger.info(f'The best candidate is {best_model["index"]}-th prunned model (Train Acc: {best_model["acc"]})\n')

        [logger.info(line) for line in best_model['pruning_info']]
        logger.info(" ")
        self.best_node_graph = best_model['node_graph']

        return best_model['model']


    def get_pruning_node_info(self):
        return self.best_node_graph

    

    def set_hooking(self):

        def save_fmaps(key):
            def forward_hook(module, inputs, outputs):
                
                if key not in self.activations:
                    self.activations[key] = inputs[0]
                else:

                    self.activations[key] = torch.cat((self.activations[key], inputs[0]), dim=0)
            return forward_hook
        
        prev_names = []
        prev_layers = []
        last_conv = 0
        for name, layer in reversed(list(self.model.named_modules())):
            
            if _exclude_layer(layer):
                continue
            layer_type = get_layer_type(layer)
            
            if layer_type == 'Conv':
                if last_conv == 0:
                    
                    prev_i_layer = prev_layers[last_conv]
                    prev_i_name = prev_names[last_conv]

                    while get_layer_type(prev_i_layer) == 'BN':
                        last_conv += 1
                        prev_i_layer = prev_layers[last_conv]
                        prev_i_name = prev_names[last_conv]

                    target_layer = prev_layers[last_conv-1]
                    target_name = prev_names[last_conv-1]
                    self.hook_layers.append(target_layer.register_forward_hook(save_fmaps(target_name)))    
                    last_conv =  True
                self.hook_layers.append(layer.register_forward_hook(save_fmaps(name)))    

            if last_conv == 0:
                prev_names.append(name)
                prev_layers.append(layer)

        batches = 0
        batch_size = self.train_loader.batch_size
        with torch.no_grad():
            for images, labels in self.train_loader:
                batches += batch_size
                images = images.to(self.device)
                out = self.model(images)

                if batches >= self.criteria_args.NUM_SAMPLES:
                    break

        for key, val in self.node_graph.items():
            if 'group' in val:
                if val['input_convs'] != None:
                    
                    for i_input in val['input_convs']:
                        self.conv2target_conv[i_input] = val['torch_name']

            if get_layer_type(val['layer']) == "Linear" and 'input_convs' in val:
                if 'Conv' in val['input_convs'][0]:
                    for i_input in val['input_convs']:
                        self.conv2target_conv[i_input] = target_name
                else:
                    for i_input in val['input_convs']:
                        self.conv2target_conv[i_input] = val['torch_name']

        for i_hook in self.hook_layers:
            i_hook.remove()
