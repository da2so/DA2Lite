
from collections import OrderedDict


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


		self.origin_model = origin_model
		self.new_model = new_model

		self.train_loader = train_loader
		self.test_loader = test_loader

		self.device = device

		self.num_samples = train_cfg.NUM_SAMPLES
		self.compress_name = compress_name

		self.origin_t_layers, self.new_t_layers = self.get_target_layers()

		import sys
		sys.exit()
		origin_model_hook = []
		new_model_hook = []

		"""
		def forward_hook(module, input, output):
			return output

		for layer in origin_model.modules():
			if True:
			layer.register_forward_hook(forward_hook)
		"""
	def build(self):

		self.origin_model.eval()
		self.new_model.eval()
		idx = 0

		for layer in origin_model.modules():

			if _exclude_layer(layer): 
				continue

			layer_type = get_layer_type(layer)

			if layer_type == 'Conv' and 'prune_idx' in self.node_graph[idx]:



			idx += 1

	def get_target_layers(self):

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

			layer_type = get_layer_type(layer)
			if layer_type == 'Conv' and name != new_tmp_layers.keys()[idx]:
				idx += 1
				while new_tmp_layers.keys()[idx]
				
				
				else:

		
		return origin_model_conv, new_model_conv



	def get_pruned_layers(self):

		origin_t_layers = OrderedDict()
		new_t_layers = OrderedDict()
		new_tmp_layers = OrderedDict()


		for name, layer in self.new_model.named_modules():

			if _exclude_layer(layer):
				continue
			layer_type = get_layer_type(layer)

			if layer_type == 'Conv':
				new_tmp_layers[name] = layer
		

		for name, layer in self.new_model.named_modules():