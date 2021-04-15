import os
import numpy as np
import time
from collections import OrderedDict

import onnx
import torch.nn as nn
import torch
from torchsummary import summary

from DA2Lite.core.layer_utils import _exclude_layer, get_layer_type
#PRUNABLE_LAYERS = [ nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm, nn.Linear, nn.PReLU]


class GraphGenerator(object):
    def __init__(self, model, img_shape, save_dir):
        
        self.model = model
        self.model.eval().cpu()    
        self.dummy_input = torch.randn(img_shape).unsqueeze_(0)
        self.onnx_save_path = os.path.join(save_dir, 'test.onnx')
        
    def build(self):
        
        layer_info = self._get_layer_info(self.model)

        onnx_graph = self._get_onnx_graph(self.model, self.dummy_input, self.onnx_save_path)
        parsed_onnx_graph = self._onnx_graph_parser(onnx_graph)
        
        node_graph, group_set = self._get_combined_graph(layer_info, parsed_onnx_graph)
        

        return node_graph, group_set

    
    def _get_layer_info(self, model):

        layer_info = OrderedDict()
        i = 0 
        for idx, layer in enumerate(model.modules()):
            if idx == 0 or _exclude_layer(layer):
                continue

            layer_info[i] = {'layer': layer}
            i += 1
        return layer_info            
    
    def _get_onnx_graph(self, model, dummy_input, save_path):
        """
        get input and output layers in each module using onnx framework
        """
        node_graph = OrderedDict()
        torch.onnx.export(model, dummy_input, save_path, verbose=False)

        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)

        for i in range(len(onnx_model.graph.node)):
            i_node = onnx_model.graph.node[i]

            if 'Conv' == i_node.op_type:
                node_graph[i_node.output[0]] = {'input': [i_node.input[0]], 'op_type': i_node.op_type, 'name': i_node.name}
            else:
                node_graph[i_node.output[0]] = {'input': i_node.input, 'op_type': i_node.op_type, 'name': i_node.name}

        os.remove(save_path)

        return node_graph
        
    def _onnx_graph_parser(self, node_graph):
        idx = 0
        
        for key in node_graph.keys():

            if node_graph[key]['op_type'] == 'Conv':
                node_graph[key]['group'] = idx
                
                if  idx != 0:
                    node_graph[key]['input_convs'] = []
                    node_graph[key]['concat_op'] = []
                    
                    i_input = node_graph[key]['input'][0]
                    while 'group' not in node_graph[i_input]:
                        i_input = node_graph[i_input]['input'][0]
                    
                    if node_graph[i_input]['op_type'] == 'Add':
                        node_graph[key]['input_convs'].extend(node_graph[i_input]['input_convs'])
                    elif node_graph[i_input]['op_type'] == 'Concat':
                        node_graph[key]['input_convs'].extend(node_graph[i_input]['input_convs'])
                        node_graph[key]['concat_op'].extend(node_graph[i_input]['input_convs'])
                    elif node_graph[i_input]['op_type'] == 'Conv':
                        node_graph[key]['input_convs'].extend([node_graph[i_input]['name']])

                idx += 1

            elif node_graph[key]['op_type'] == 'Add':
                assert len(node_graph[key]['input']) >= 2
                
                group_min_val = 1e4
                input_list = []
                node_graph[key]['input_convs'] = []
                node_graph[key]['add_op'] = []
                node_graph[key]['concat_op'] = []

                for i_input in node_graph[key]['input']:
                    while 'group' not in node_graph[i_input]:
                        i_input = node_graph[i_input]['input'][0]
                    
                    input_list.append(i_input)
                    if node_graph[i_input]['group'] != -1 and node_graph[i_input]['group'] < group_min_val:
                        group_min_val = node_graph[i_input]['group']

                    if node_graph[i_input]['op_type'] == 'Add':
                        node_graph[key]['input_convs'].extend(node_graph[i_input]['input_convs'])
                    elif node_graph[i_input]['op_type'] == 'Conv':
                        node_graph[key]['input_convs'].extend([node_graph[i_input]['name']])
                        

                for i_input in input_list:
                    if node_graph[i_input]['group'] != group_min_val:
                        node_graph[i_input]['group'] = group_min_val            
                
                node_graph[key]['group'] = group_min_val
                idx += 1

            elif node_graph[key]['op_type'] == 'Concat':
                assert len(node_graph[key]['input']) >= 2
                
                node_graph[key]['input_convs'] = []
                node_graph[key]['concat_op'] = []

                group_num_for_concat = -1
                for i_input in node_graph[key]['input']:
                    if 'i_input' not in node_graph[i_input]:
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

                    node_graph[key]['group'] = group_num_for_concat


        node_graph = {k: v for k, v in node_graph.items() if node_graph[k]['op_type'] == 'Conv'}
        
        return node_graph


    def _find_input_convs(self, key, layer_info):
        down_key = 1

        while 'name' not in layer_info[key - down_key]:
            down_key += 1
        
        return down_key

    def _get_combined_graph(self, layer_info, onnx_graph):

        is_first_linear = True
        group_set = set()

        for key in layer_info.keys():
            layer_type = get_layer_type(layer_info[key]['layer'])

            if layer_type == 'Conv':
                assert 'group' in onnx_graph[next(iter(onnx_graph))]

                i_graph = onnx_graph[next(iter(onnx_graph))]

                layer_info[key]['group'] = i_graph['group']
                layer_info[key]['name'] = i_graph['name']
                if 'input_convs' in i_graph:
                    layer_info[key]['input_convs'] = i_graph['input_convs']
                else:
                    layer_info[key]['input_convs'] = None
                group_set.add(i_graph['group'])
                del onnx_graph[next(iter(onnx_graph))]
                
            elif layer_type == 'BN':
                
                down_key = self._find_input_convs(key, layer_info)
                
                layer_info[key]['input_convs'] = [layer_info[key - down_key]['name']]

        
            elif layer_type == 'Linear' and is_first_linear:

                down_key = self._find_input_convs(key, layer_info)
                
                layer_info[key]['input_convs'] = [layer_info[key - down_key]['name']]
                is_first_linear = False

        return layer_info, group_set

        


