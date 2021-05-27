import numpy as np
import copy
import random
from collections import Counter, OrderedDict, defaultdict

import torch
import torch.nn as nn

from DA2Lite.core.layer_utils import _exclude_layer, get_layer_type
from DA2Lite.core.graph_generator import GraphGenerator
from DA2Lite.compression.clustering.utils import load_strategy, load_method
from DA2Lite.core.log import get_logger

logger = get_logger(__name__)

act_based_clustering = {'New2Clustering'}

class Clustering(object):
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
    
        self.clustering_cfg = compress_cfg

        self.cluster_args = None
        if 'CLUSTER_ARGS' in self.clustering_cfg:
            self.cluster_args = self.clustering_cfg.CLUSTER_ARGS

        #get network node graph using onnx framework
        graph_model = copy.deepcopy(model)
        self.node_graph, self.group_set = GraphGenerator(model=graph_model, 
                                                        img_shape=self.img_shape, 
                                                        save_dir=self.save_dir
                                                        ).build()
        del graph_model
        self.criteria_class = load_method(method_name=self.clustering_cfg.CLUSTER, 
                                            cluster_args=self.cluster_args,
                                            model=self.model)

        self.activations = defaultdict()
        self.conv2target_conv = defaultdict()
        self.hook_layers = []

        if self.clustering_cfg.CLUSTER in act_based_clustering:
            self.set_hooking()
        
    def set_cluster_idx(self, node_graph):
        clustering_info = []
        group_frequency = dict()
        for idx, key in enumerate(node_graph.keys()):

            i_node = node_graph[key]
            layer_type = get_layer_type(i_node['layer'])

            if layer_type == 'Conv':
                
                if self.clustering_cfg.CLUSTER not in act_based_clustering:
                    cluster_idx = self.criteria_class.get_cluster_idx(i_node=i_node)
                    
                    i_node['cluster_idx'] = cluster_idx

                    # For integrating prune indexes
                    if i_node['group'] not in group_frequency:
                        group_frequency[i_node['group']] = [key]
                    else:
                        group_frequency[i_node['group']].append(key)
                
                else:
                    if i_node['group'] in group_frequency:
                        cluster_idx = group_frequency[i_node['group']] 
                    else:
                        f_maps = self.activations[self.conv2target_conv[i_node['name']]]
                        cluster_idx = self.criteria_class.get_cluster_idx(i_node=i_node,
                                                    f_maps=f_maps,
                                                    device=self.device)
                        group_frequency[i_node['group']] = cluster_idx

                    i_node['cluster_idx'] = cluster_idx

            elif layer_type == 'GroupConv':
                down_key = -1
                while 'cluster_idx' not in node_graph[key + down_key]:
                    down_key -= 1
                
                i_node['cluster_idx'] = node_graph[key + down_key]['cluster_idx']

        if self.clustering_cfg.CLUSTER not in act_based_clustering:
            node_graph = self._integrate_cluster_idx(node_graph=node_graph,
                                                    group_frequency=group_frequency)

        return node_graph

    def _integrate_cluster_idx(self, node_graph, group_frequency):
        
        for group_num in group_frequency.keys():
            if len(group_unefrequency[group_num]) >= 2:
                total_cluster_idx = []
                for key in group_frequency[group_num]:
                    total_cluster_idx.extend(node_graph[key]['cluster_idx'])
                
                limit_num_idx = len(node_graph[key]['cluster_idx'])
                count_cluster_idx = dict(Counter(total_cluster_idx).most_common(limit_num_idx))
                cutted_cluster_idx = []
                
                # only prune filters if prune indexes of layers are duplicated
                for cluster_idx, num in count_cluster_idx.items():
                    if num >= 2:
                        cutted_cluster_idx.append(cluster_idx)

                # redefine prune indexes for same groups
                for key in group_frequency[group_num]:
                    node_graph[key]['cluster_idx'] = cutted_cluster_idx
   
        return node_graph
                
    def cluster(self, node_graph):

        node_graph = self.set_cluster_idx(node_graph)

        new_model = copy.deepcopy(self.model)

        i = 0
        clustering_info = []
        for idx, data in enumerate(new_model.named_modules()):
            name, layer = data
            if idx == 0 or _exclude_layer(layer):
                continue
            
            layer_type = get_layer_type(layer)
            if layer_type == 'Conv':
                prev_cluster_idx = []
                if 'input_convs' in node_graph[i]:
                    prev_cluster_idx = self.get_prev_cluster_idx(node_graph=node_graph,
                                                            index=i)
                cluster_idx = node_graph[i]['cluster_idx']

                keep_prev_idx = list(set(range(layer.in_channels)) - set(prev_cluster_idx))
                keep_idx = list(set(range(layer.out_channels)) - set(cluster_idx))
                
                w = layer.weight.data[:, keep_prev_idx, :, :].clone()
                layer.weight.data = w[keep_idx, :, :, :].clone()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[keep_idx].clone()

                
                clustering_info.append(f'Out channels are clustered: [{layer.out_channels:4d}] -> [{len(keep_idx):4d}] at "{name}" layer')
                layer.out_channels = len(keep_idx)
                layer.in_channels = len(keep_prev_idx)

            elif layer_type == 'GroupConv':
                cluster_idx = node_graph[i]['cluster_idx']

                keep_idx = list(set(range(layer.out_channels)) - set(pruncluster_idxe_idx))

                layer.weight.data = layer.weight.data[keep_idx, :, :, :].clone()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[keep_idx].clone()

                clustering_info.append(f'Out channels are clustered: [{layer.out_channels:4d}] -> [{len(keep_idx):4d}] at "{name}" layer')
                layer.out_channels = len(keep_idx)
                layer.in_channels = len(keep_idx)
                layer.groups = len(keep_idx)

            elif layer_type == 'BN':
                prev_cluster_idx = self.get_prev_cluster_idx(node_graph=node_graph,
                                                        index=i)
                keep_idx = list(set(range(layer.num_features)) - set(prev_cluster_idx))

                layer.running_mean.data = layer.running_mean.data[keep_idx].clone()
                layer.running_var.data = layer.running_var.data[keep_idx].clone()
                if layer.affine:
                    layer.weight.data = layer.weight.data[keep_idx].clone()
                    layer.bias.data = layer.bias.data[keep_idx].clone()
                
                clustering_info.append(f'Out channels are clustered: [{layer.num_features:4d}] -> [{len(keep_idx):4d}] at "{name}" layer')
                layer.num_features = len(keep_idx)

            elif layer_type == 'Linear':
                if 'input_convs' in node_graph[i]:
                    prev_cluster_idx = self.get_prev_cluster_idx(node_graph=node_graph,
                                                            index=i)
                    keep_idx = list(set(range(layer.in_features)) - set(prev_cluster_idx))
                    
                    layer.weight.data = layer.weight.data[:, keep_idx].clone()

                    layer.in_features = len(keep_idx)
            i += 1

        return new_model, clustering_info, node_graph


    def get_prev_cluster_idx(self, node_graph, index):
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

            
        prev_cluster_idx = list()
        total_channels = 0
        for idx, i_layer in enumerate(in_layers):
            if idx == 0:
                prev_cluster_idx.extend(i_layer['cluster_idx'])
                total_channels += i_layer['layer'].out_channels
            else:
                if i_layer['name'] in concat_in_layers:
                    for i_cluster in i_layer['cluster_idx']:
                        prev_cluster_idx.extend([i_cluster + total_channels]) # For concat operation
                    total_channels += i_layer['layer'].out_channels
                else:
                    break
        return prev_cluster_idx
            
    def build(self):

        new_model, clustering_info, node_graph = self.cluster(self.node_graph)

        [logger.info(line) for line in clustering_info]
        logger.info(" ")
        self.best_node_graph = node_graph

        return new_model



    def get_clustering_node_info(self):
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

                if batches >= self.cluster_args.NUM_SAMPLES:
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
