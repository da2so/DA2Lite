
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
                    weight_copy = i_node['layer'].weight.data.cpu().numpy()
                    prune_idx = criteria(weight_copy, group_to_ratio[i_node['group']]) #error 가능성 (ratio가 np이니까)
                    
                    total_channel = i_node['layer'].out_channel
                    remaining_channel = total_channel - len(prune_idx)

                    i_node['prune_idx'] = prune_idx

                    print(f'layer index: {idx} \t total channel: {total_channel} \t remaining channel: {remaining_channel}')
                
                elif isinstance(i_node['layer'], nn.BatchNorm2d):
                    down_key = 1

                    while 'prune_idx' not in node_graph[key - down_key]:
                        down_key += 1
                    
                    i_node['prune_idx'] = node_graph[key - down_key]['prune_idx']

                
                        
            print(criteria)
            import sys
            sys.exit()


        
                    
        """
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.cpu().numpy()
                weight_norm = np.sum(np.abs(weight_copy), axis=(1, 2, 3))
                num_channel = len(weight_norm)
                if PRUNED_CHANNEL[layer_id] == 0:
                    thre = -1
                else:
                    thre = sorted(weight_norm)[int(num_channel * PRUNED_CHANNEL[layer_id]) - 1]
                mask = (weight_norm > thre).astype(np.int64)
                cfg.append(int(np.sum(mask)))
                cfg_mask.append(mask)
                layer_id += 1
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    format(k, len(mask), int(np.sum(mask))))
            elif isinstance(m, nn.MaxPool2d):
                cfg.append('M')
            import sys
            sys.exit()
        """