import os
import numpy as np
import time
from collections import OrderedDict

import onnx
import torch.nn as nn
import torch
from torchsummary import summary

from DA2Lite.core.layer_utils import _exclude_layer, get_layer_type


class GraphGenerator(object):
    def __init__(self, model, img_shape, save_dir):
        self.model = model.eval().cpu()
        self.dummy_input = torch.randn(img_shape).unsqueeze_(0) # For onnx model export
        self.onnx_save_path = os.path.join(save_dir, 'test.onnx')
        
    def build(self):
        layer_info = self._get_layer_info(torch_model=self.model)
        
        onnx_graph = self._get_onnx_graph(torch_model=self.model, 
                                        dummy_input=self.dummy_input,
                                        save_path=self.onnx_save_path)
        parsed_onnx_graph = self._onnx_graph_parser(node_graph=onnx_graph)
        
        node_graph, group_set = self._get_combined_graph(layer_info=layer_info, 
                                                        onnx_graph=parsed_onnx_graph)
        
        return node_graph, group_set

    def _get_layer_info(self, torch_model):
        layer_info = OrderedDict()
        i = 0 
        for idx, (name, layer) in enumerate(torch_model.named_modules()):
            if idx == 0 or _exclude_layer(layer):
                continue
            layer_info[i] = {'layer': layer, 'id': hash(name)}
            i += 1

        return layer_info            
    
    def _get_onnx_graph(self, torch_model, dummy_input, save_path):
        """
        It brings connected input layer(s) and saves an operation type and name in each layer.
        """
        node_graph = OrderedDict()

        torch.onnx.export(torch_model, dummy_input, save_path, verbose=False)
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)

        for i in range(len(onnx_model.graph.node)):
            i_node = onnx_model.graph.node[i]

            if 'Conv' == i_node.op_type or 'Gemm' == i_node.op_type: # Gemm indicates linear(fc) layer.
                node_graph[i_node.output[0]] = {'input': [i_node.input[0]], 'op_type': i_node.op_type, 'name': i_node.name}
            else:
                node_graph[i_node.output[0]] = {'input': i_node.input, 'op_type': i_node.op_type, 'name': i_node.name}

        os.remove(save_path) # Remove onnx model

        return node_graph
        
    def _onnx_graph_parser(self, node_graph):
        """
        It connects input and output layers in consideration of Add and Concat layers
        """
        group_idx = 0
        linear_group_idx = -1

        for key in node_graph.keys():
            if node_graph[key]['op_type'] == 'Conv':
                node_graph[key]['group'] = group_idx
                
                if group_idx != 0: # If a layer is not first come.
                    node_graph[key]['input_convs'] = []
                    node_graph[key]['concat_op'] = [] # Use when if there is a concat operation from input layer(s).
                    
                    i_input = node_graph[key]['input'][0] 
                    while 'group' not in node_graph[i_input]: # Find a connected input layer.
                        i_input = node_graph[i_input]['input'][0]
                    
                    if node_graph[i_input]['op_type'] == 'Add':
                        node_graph[key]['input_convs'].extend(node_graph[i_input]['input_convs'])
                    elif node_graph[i_input]['op_type'] == 'Concat':
                        node_graph[key]['input_convs'].extend(node_graph[i_input]['input_convs'])
                        node_graph[key]['concat_op'].extend(node_graph[i_input]['input_convs'])
                    elif node_graph[i_input]['op_type'] == 'Conv':
                        node_graph[key]['input_convs'].extend([node_graph[i_input]['name']])

                group_idx += 1

            elif node_graph[key]['op_type'] == 'Add':
                assert len(node_graph[key]['input']) >= 2
                
                group_min_val = 1e4
                input_list = []
                node_graph[key]['input_convs'] = []
                #node_graph[key]['concat_op'] = []

                for i_input in node_graph[key]['input']:

                    if i_input not in node_graph:
                        break
                    while 'group' not in node_graph[i_input]:
                        i_input = node_graph[i_input]['input'][0]
                    
                    input_list.append(i_input)
                    if -1 < node_graph[i_input]['group'] < group_min_val:
                        group_min_val = node_graph[i_input]['group'] # Find smallest group index among connected layers.

                    if node_graph[i_input]['op_type'] == 'Add':
                        node_graph[key]['input_convs'].extend(node_graph[i_input]['input_convs'])
                    elif node_graph[i_input]['op_type'] == 'Conv':
                        node_graph[key]['input_convs'].extend([node_graph[i_input]['name']])
                
                # Set to a unified group index among connected layers from an add layer.
                for i_input in input_list:
                    node_graph[i_input]['group'] = group_min_val   
                node_graph[key]['group'] = group_min_val

                group_idx += 1

            elif node_graph[key]['op_type'] == 'Concat':
                assert len(node_graph[key]['input']) >= 2
                
                node_graph[key]['input_convs'] = []
                node_graph[key]['concat_op'] = []

                group_num_for_concat = -1
                for i_input in node_graph[key]['input']:
                    if i_input not in node_graph:
                        break
                    while 'group' not in node_graph[i_input]:
                        i_input = node_graph[i_input]['input'][0]
                    
                    if node_graph[i_input]['op_type'] == 'Add':
                        node_graph[key]['input_convs'].extend(node_graph[i_input]['input_convs'])
                    elif node_graph[i_input]['op_type'] == 'Concat':
                        node_graph[key]['input_convs'].extend(node_graph[i_input]['input_convs'])
                        node_graph[key]['concat_op'].extend(node_graph[i_input]['input_convs'])
                    elif node_graph[i_input]['op_type'] == 'Conv':
                        node_graph[key]['input_convs'].extend([node_graph[i_input]['name']])
                        node_graph[key]['concat_op'].extend([node_graph[i_input]['name']])

                    node_graph[key]['group'] = group_num_for_concat

            elif node_graph[key]['op_type'] == 'Gemm':
                
                node_graph[key]['input_convs'] = []
                node_graph[key]['concat_op'] = []

                for i_input in node_graph[key]['input']:
                    if i_input not in node_graph:
                        continue
                    
                    while 'group' not in node_graph[i_input]:
                        i_input = node_graph[i_input]['input'][0]

                    if node_graph[i_input]['op_type'] == 'Add':
                        node_graph[key]['input_convs'].extend(node_graph[i_input]['input_convs'])
                    elif node_graph[i_input]['op_type'] == 'Concat':
                        node_graph[key]['input_convs'].extend(node_graph[i_input]['input_convs'])
                        node_graph[key]['concat_op'].extend(node_graph[i_input]['input_convs'])
                    elif node_graph[i_input]['op_type'] == 'Conv':
                        node_graph[key]['input_convs'].extend([node_graph[i_input]['name']])
                        node_graph[key]['concat_op'].extend([node_graph[i_input]['name']])

                node_graph[key]['group'] = linear_group_idx
                linear_group_idx -= 1
        
        node_graph = {k: v for k, v in node_graph.items() \
                    if node_graph[k]['op_type'] == 'Conv' or node_graph[k]['op_type'] == 'Gemm'}

        return node_graph


    def _find_input_convs(self, key, layer_info, key_out_channels):
        down_key = -1
        concat_op = None
        while 'name' not in layer_info[key + down_key]:
            down_key -= -1
        
        if layer_info[key + down_key]['layer'].out_channels != key_out_channels: # If bn - conv order, not conv - bn order
            up_key = 1 
            while 'name' not in layer_info[key + up_key]:
                up_key += 1
            
            if 'concat_op' in layer_info[key + up_key]:
                concat_op = layer_info[key + up_key]['concat_op']

            return layer_info[key + up_key]['input_convs'], concat_op
        else:
            if 'concat_op' in layer_info[key + down_key]:
                concat_op = layer_info[key + down_key]['concat_op']

            return [layer_info[key + down_key]['name']], concat_op

    def _get_combined_graph(self, layer_info, onnx_graph):
        """
        It combines onnx and pytorch node graphs
        """
        is_first_linear = True
        group_set = set()

        for key in layer_info.keys():
            layer_type = get_layer_type(layer_info[key]['layer'])

            if layer_type == 'Conv' or layer_type == 'GroupConv':
                assert 'group' in onnx_graph[next(iter(onnx_graph))]

                i_graph = onnx_graph[next(iter(onnx_graph))]

                layer_info[key]['group'] = i_graph['group']
                layer_info[key]['name'] = i_graph['name']
                if 'input_convs' in i_graph:
                    layer_info[key]['input_convs'] = i_graph['input_convs']
                else:
                    layer_info[key]['input_convs'] = None
                if 'concat_op' in i_graph:
                    layer_info[key]['concat_op'] = i_graph['concat_op']

                group_set.add(i_graph['group'])
                del onnx_graph[next(iter(onnx_graph))]
            
            elif layer_type == 'Linear':
                assert 'group' in onnx_graph[next(iter(onnx_graph))]

                i_graph = onnx_graph[next(iter(onnx_graph))]
                layer_info[key]['name'] = i_graph['name']
                if 'input_convs' in i_graph:
                    layer_info[key]['input_convs'] = i_graph['input_convs']
                else:
                    layer_info[key]['input_convs'] = None
                if 'concat_op' in i_graph:
                    layer_info[key]['concat_op'] = i_graph['concat_op']
                del onnx_graph[next(iter(onnx_graph))]

            else:
                continue
        
        for key in layer_info.keys():
            layer_type = get_layer_type(layer_info[key]['layer'])
            
            if layer_type == 'BN':
                bn_out_channels = layer_info[key]['layer'].num_features
                in_convs_name, in_convs_concat = self._find_input_convs(key, layer_info, bn_out_channels)
                
                layer_info[key]['input_convs'] = in_convs_name
                if in_convs_concat != None:
                    layer_info[key]['concat_op'] = in_convs_concat

            else:
                continue

        return layer_info, group_set

        


