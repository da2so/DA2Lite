import copy

import torch.nn as nn

from DA2Lite.core.layer_utils import _exclude_layer, get_layer_type
from DA2Lite.compression.knowledge_distillation import methods
from DA2Lite.core.log import get_logger

logger = get_logger(__name__)

cfg_to_method = {'FSKD': 'FSKD'}

class KnowledgeDistillation(object):
    def __init__(self,
                kd_cfg,
                origin_model,
                new_model,
                train_loader,
                test_loader,
                device,
                **kwargs):
        
        self.origin_model = origin_model
        self.new_model = new_model

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = device
        
        self.kd_name = kd_cfg.KNOWLEDGE_DISTILLATION

    
    def _get_method(self, method_name):
        
        try:
            method_func = getattr(methods, method_name)
        except:
            raise ValueError(f'Invalid KD method: {method_name}')
        
        return method_func


    def build(self):
        
        kd_method = self._get_method(cfg_to_method[self.kd_name])

        kd_obj = kd_method()
        kd_obj.build()
        
        logger.info(f'In, Out channels are decomposed: [{in_channels}, {out_channels}] -> [{ranks[1]}, {ranks[0]}] at "{name}" layer')

        return new_model