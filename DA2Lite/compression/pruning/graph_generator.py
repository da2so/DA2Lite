import os
import numpy as np
import time
from collections import OrderedDict

import onnx
import torch.nn as nn
import torch
from torchsummary import summary

PRUNABLE_LAYERS = [ nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm, nn.Linear, nn.PReLU]

def _exclude_layer(layer):

    if isinstance(layer, nn.Sequential):
        return True
    if not 'torch.nn' in str(layer.__class__):
        return True

    return False

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
        for idx, data in enumerate(model.named_modules()):
            name, layer = data
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
                idx += 1
                
                if 'input' not in node_graph[key]['input'][0]:

                    node_graph[key]['input_conv_layers'] = []
                    for i_input in node_graph[key]['input']:
                        while 'group' not in node_graph[i_input]:
                            i_input = node_graph[i_input]['input'][0]
                    
                    if node_graph[i_input]['op_type'] == 'Add':
                        node_graph[key]['input_conv_layers'].extend(node_graph[i_input]['input_conv_layers'])
                    elif node_graph[i_input]['op_type'] == 'Conv':
                        node_graph[key]['input_conv_layers'].extend([node_graph[i_input]['name']])

            if node_graph[key]['op_type'] == 'Add':
                assert len(node_graph[key]['input']) >= 2
                
                group_list = []
                input_list = []
                node_graph[key]['input_conv_layers'] = []
                for i_input in node_graph[key]['input']:
                    while 'group' not in node_graph[i_input]:
                        i_input = node_graph[i_input]['input'][0]
                    
                    input_list.append(i_input)
                    group_list.append(node_graph[i_input]['group']) 

                    if node_graph[i_input]['op_type'] == 'Add':
                        node_graph[key]['input_conv_layers'].extend(node_graph[i_input]['input_conv_layers'])
                    elif node_graph[i_input]['op_type'] == 'Conv':
                        node_graph[key]['input_conv_layers'].extend([node_graph[i_input]['name']])
                        


                min_val = min(group_list)

                for i_input in input_list:
                    if node_graph[i_input]['group'] != min_val:
                        node_graph[i_input]['group'] = min_val            
                
                node_graph[key]['group'] = min_val
                idx += 1
        
        node_graph = {k: v for k, v in node_graph.items() if node_graph[k]['op_type'] == 'Conv'}
        
        return node_graph

    def _get_combined_graph(self, layer_info, onnx_graph):

        is_first_linear = True
        group_set = set()
        for key in layer_info.keys():
            
            if isinstance(layer_info[key]['layer'], nn.Conv2d):
                assert 'group' in onnx_graph[next(iter(onnx_graph))]

                i_graph = onnx_graph[next(iter(onnx_graph))]

                layer_info[key]['group'] = i_graph['group']
                layer_info[key]['name'] = i_graph['name']
                if 'input_conv_layers' in i_graph:
                    layer_info[key]['input_conv_layers'] = i_graph['input_conv_layers']
                group_set.add(i_graph['group'])
                del onnx_graph[next(iter(onnx_graph))]
                
            elif isinstance(layer_info[key]['layer'], nn.BatchNorm2d):
                down_key = 1
                
                while 'group' not in layer_info[key - down_key]:
                    down_key += 1
                
                layer_info[key]['group'] = layer_info[key - down_key]['group']
            
        
            elif isinstance(layer_info[key]['layer'], nn.Linear) and is_first_linear:
                down_key = 1

                while 'input_conv_layers' not in layer_info[key - down_key]:
                    down_key += 1
                
                layer_info[key]['input_conv_layers'] = [layer_info[key - down_key]['name']]
                is_first_linear = False

        return layer_info, group_set

        
