import copy

import torch.nn as nn
import torch
from torch.autograd import Variable

from DA2Lite.trainer.common import TrainerBase
from DA2Lite.trainer import knowledge_distillation
from DA2Lite.core.log import get_logger

logger = get_logger(__name__)

cfg_to_class = {'fskd': 'FSKD'}

class KnowledgeDistillation(TrainerBase):
    def __init__(self,
                cfg_util,
                train_cfg,
                prefix,
                model,
                train_loader,
                test_loader,
                device,
                **kwargs):
        
        super().__init__(cfg_util,
                        prefix,
                        train_loader,
                        test_loader,
                        device)
        
        self.new_model = model
        self.origin_model = kwargs['origin_model']

        self.train_cfg = train_cfg
        self.loss = cfg_util.get_loss()
        cfg_util.train_config = train_cfg #change to train config of compression

        self.kd_name = train_cfg.NAME
        self.compress_name = kwargs['compress_name']
        self.pruning_node_info = kwargs['pruning_node_info']
        
    def _get_kd_class(self, method_name):
        
        try:
            method_func = getattr(knowledge_distillation, method_name)
        except:
            raise ValueError(f'Invalid KD method: {method_name}')
        
        return method_func


    def train(self, epoch):
            
        kd_class = self._get_kd_class(cfg_to_class[self.kd_name])
        
        kd_obj = kd_class(train_cfg=self.train_cfg,
                        origin_model=self.origin_model,
                        new_model=self.new_model,
                        train_loader=self.train_loader,
                        test_loader=self.test_loader,
                        device=self.device,
                        compress_name=self.compress_name,
                        pruning_node_info=self.pruning_node_info)

        self.new_model = kd_obj.build()  


    def test(self, epoch, print_log=True):

        self.new_model.eval()

        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.new_model(images)
                avg_loss += self.loss(outputs, labels).sum()
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

                del images, labels    
        avg_loss /= len(self.test_loader.dataset)
        acc = float(total_correct) / len(self.test_loader.dataset)
        
        if print_log == True:
            if epoch != -1:
                logger.debug(f'Test  - Epoch [{epoch}/{self.epochs}] Accuracy: {acc}, Loss: {avg_loss.data.item()}')
            else:
                logger.info(f'Test Accuracy: {acc}, Loss {avg_loss.data.item()}')
            
        return acc, avg_loss.data.item()
    

    def evaluate(self, print_log=True):
        return self.test(-1, print_log)
    
    def build(self):
        if self.kd_name == 'fskd':
            self._print_train_cfg()
            self.train(-1)
            test_acc, test_loss = self.test(-1)
        
        logger.info(f'The trained model is saved in {self.save_path}\n')        
        torch.save(self.new_model.state_dict(), self.save_path)
        
        self.model_summary(test_acc, test_loss, self.new_model)

        return self.new_model


    def _print_train_cfg(self):
        split_train_cfg = str(self.train_cfg).split('\n')
        
        num_dummy = 60
        train_txt = ' Train configuration '.center(num_dummy,' ')
        border_txt = '-'*num_dummy
       
        logger.info(f'+{border_txt}+')
        logger.info(f'|{train_txt}|')
        logger.info(f'+{border_txt}+')
        logger.info(f'|{" ".ljust(num_dummy)}|')
        for i_tr_cfg in split_train_cfg:
            if 'IS_USE' in i_tr_cfg:
                continue
            logger.info(f'| {i_tr_cfg.ljust(num_dummy-1)}|')
        logger.info(f'|{" ".ljust(num_dummy)}|')
        logger.info(f'+{border_txt}+\n')