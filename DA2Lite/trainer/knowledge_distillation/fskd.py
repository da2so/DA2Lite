import numpy as np
from collections import OrderedDict
import copy

import torch
import torch.nn as nn

from DA2Lite.core.layer_utils import get_module_of_layer, _exclude_layer, get_layer_type


class FSKD(object):
	def __init__(self,
				train_cfg,
				origin_model,
				new_model,
				train_loader,
				test_loader,
				device,
				compress_name,
				**kwargs):

		self.origin_model = origin_model.to(device)
		self.new_model = new_model.to(device)

		self.train_loader = train_loader
		self.test_loader = test_loader

		self.device = device

		self.num_samples = train_cfg.NUM_SAMPLES
		self.compress_name = compress_name

		if self.compress_name == 'Pruner':
			self.pruning_node_info = kwargs['pruning_node_info']

			for key in self.pruning_node_info.keys():
				
				if 'prune_idx' in self.pruning_node_info[key]:
					if len(self.pruning_node_info[key]['prune_idx']) == 0:
						del self.pruning_node_info[key]['prune_idx']
		self.new_activations = []
		self.origin_activations = []

		self.origin_t_layers, self.new_t_layers = self.target_layers_hooking()


	def new_forward_hook(self, module, input, output):
		self.new_activations.append(output)
		return None
	def origin_forward_hook(self, module, input, output):
		self.origin_activations.append(output)
		return None

	def build(self):
		
		self.origin_model.eval()
		self.new_model.eval()
		
		from torchsummary import summary
		summary(self.new_model, (3,32,32))

		layer_idx = 0
		t_node_idx = 0
		check_idx = -10000
		for idx, data in enumerate(self.new_model.named_modules()):
			name, layer = data

			layer_type = get_layer_type(layer)
			if name in self.new_t_layers:		
				sample_count = 0

				for i, (inputs, targets) in enumerate(self.train_loader):
					
					if sample_count >= self.num_samples:
						break
					self.new_activations = []
					self.origin_activations = []
				
					input_var = torch.autograd.Variable(inputs).to(self.device)

					self.origin_model(input_var)
					self.new_model(input_var)

					origin_act = self.origin_activations[layer_idx]
					new_act = self.new_activations[layer_idx]
					origin_channels = origin_act.size(1)
					new_channels = new_act.size(1)

					if i == 0:
						origin_out = origin_act
						new_out = new_act
					else:
						origin_out = torch.cat((origin_out, origin_act), dim=0)
						new_out = torch.cat((new_out, new_act), dim=0)
					sample_count += self.train_loader.batch_size
					
					#del input_var
					#torch.cuda.empty_cache()
				
				print(origin_out.size())
				print(new_out.size())					
				origin_out = origin_out.permute(0, 2, 3, 1).contiguous().view(-1, origin_channels).data.cpu().numpy().astype(np.float32)
				new_out = new_out.permute(0, 2, 3, 1).contiguous().view(-1, new_channels).data.cpu().numpy().astype(np.float32)
				print(np.shape(origin_out))
				print(np.shape(new_out))		

				if self.compress_name == 'Pruner':
					while 'prune_idx' not in self.pruning_node_info[t_node_idx]:
						t_node_idx += 1

					keep_idx = list(set(range(origin_channels)) - set(self.pruning_node_info[t_node_idx]['prune_idx']))
					origin_out = origin_out[:, keep_idx]

				layer_idx += 1
				t_node_idx += 1

				ret = np.linalg.lstsq(new_out, origin_out, rcond=None)
				print(ret[1])
				pwconv_weights = np.transpose(ret[0])
				
				module, last_name = get_module_of_layer(self.new_model, name)
				module._modules[str(last_name)] = self.add_pwconv_layer(layer=layer, 
																		pwconv_weights=pwconv_weights,
																		channels=new_channels)

		from torchsummary import summary
		summary(self.new_model, (3,32,32))
		return self.new_model
		
	def target_layers_hooking(self):

		if self.compress_name == 'Pruner':
			origin_t_layers, new_t_layers = self.get_pruned_layers()
		elif self.compress_name == 'FilterDecomposition':
			origin_t_layers, new_t_layers = self.get_fd_layers()
		else:
			raise ValueError(f'Get an invalid compression name : {self.compress_name} in Knowledge distillation')

		return origin_t_layers, new_t_layers

	def get_fd_layers(self):
		
		origin_t_layers = OrderedDict()
		new_t_layers = OrderedDict()
		new_tmp_layers = OrderedDict()

		for name, layer in self.new_model.named_modules():

			if _exclude_layer(layer):
				continue

			layer_type = get_layer_type(layer)

			if layer_type == 'Conv':
				new_tmp_layers[name] = layer

		idx = 0
		for name, layer in self.origin_model.named_modules():
            
			if _exclude_layer(layer):
				continue
			"""
			layer_type = get_layer_type(layer)
			if layer_type == 'Conv' and name != new_tmp_layers.keys()[idx]:
				idx += 1
				while new_tmp_layers.keys()[idx]
				
				
				else:

			"""
		return origin_model_conv, new_model_conv

		
	def get_pruned_layers(self):
		self.origin_model.eval()
		self.new_model.eval()

		origin_t_layers = OrderedDict()
		new_t_layers = OrderedDict()

		check_idx = -100
		# suppose that the number of layers between origin and new models are same.
		for idx, (origin_data, new_data) in enumerate(zip(self.origin_model.named_modules(),\
														self.new_model.named_modules())):
			origin_name, origin_layer = origin_data
			new_name, new_layer = new_data

			if _exclude_layer(origin_layer):
				continue
			layer_type = get_layer_type(origin_layer)

			if layer_type == 'Conv' and origin_layer.out_channels != new_layer.out_channels:
				prev_origin_layer = origin_layer
				prev_origin_name = origin_name
				prev_new_layer = new_layer
				prev_new_name = new_name
				check_idx = idx

			if layer_type == 'BN' and check_idx + 1 == idx:
				origin_t_layers[origin_name] = origin_layer
				origin_layer.register_forward_hook(self.origin_forward_hook)

				new_t_layers[new_name] = new_layer
				new_layer.register_forward_hook(self.new_forward_hook)

			elif check_idx + 1 == idx:
				origin_t_layers[prev_origin_name] = prev_origin_layer
				prev_origin_layer.register_forward_hook(self.origin_forward_hook)

				new_t_layers[prev_new_name] = prev_new_layer
				prev_new_layer.register_forward_hook(self.new_forward_hook)
			
		return origin_t_layers, new_t_layers				

	
	def add_pwconv_layer(self, layer, pwconv_weights, channels):
		
		print(channels)
		print(np.shape(pwconv_weights))
		pwconv_layer = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		pwconv_layer.weight.data.copy_(torch.from_numpy(pwconv_weights).view(channels, channels, 1, 1))
		
		new_layers = [layer, pwconv_layer]
		return nn.Sequential(*new_layers)

