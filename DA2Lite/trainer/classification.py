import os
from typing import List, Tuple
from tqdm import tqdm

import torch
from torch.autograd import Variable

from DA2Lite.trainer.common import TrainerBase
from DA2Lite.core.log import get_logger

logger = get_logger(__name__)


class Classification(TrainerBase):
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

        self.origin_summary = None
        if 'origin_summary' in kwargs:
            self.origin_summary = kwargs['origin_summary']
        
        self.model = model.to(device)
        self.train_cfg = train_cfg
        self.is_train = train_cfg.IS_USE
        cfg_util.train_config = train_cfg #change to train config of compression

        self.epochs = train_cfg.EPOCHS
        self.optimizer = cfg_util.get_optimizer(self.model)
        self.loss = cfg_util.get_loss()

        self.scheduler = None
        if train_cfg.SCHEDULER:
            self.scheduler = cfg_util.get_scheduler(self.optimizer)
    
    def train(self, epoch):

        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        total_correct = 0
        batches = 0
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader)-1, leave=False)
        for i, (images, labels) in loop:
            batches += len(labels)

            images, labels = Variable(images).to(self.device), Variable(labels).to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.loss(outputs, labels)

            pred = outputs.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

            loss.backward()
            self.optimizer.step()

            acc = float(total_correct) / batches
            loop.set_description(f"Train - Epoch [{epoch}/{self.epochs}]")
            loop.set_postfix(Accuracy=acc, Loss=loss.item())

            if i == len(self.train_loader) -1:
                logger.debug(f'Train - Epoch [{epoch}/{self.epochs}] Accuracy: {acc}, Loss: {loss.item()}')
    

    def test(self, epoch, print_log=True):

        self.model.eval()

        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                avg_loss += self.loss(outputs, labels)
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        avg_loss /= (i+1)
        acc = float(total_correct) / len(self.test_loader.dataset)
        
        if epoch != -1 and print_log == True:
            logger.debug(f'Test  - Epoch [{epoch}/{self.epochs}] Accuracy: {acc}, Loss: {avg_loss.data.item()}')
        #else:
        #    logger.info(f'Test Accuracy: {acc}, Loss {avg_loss.data.item()}')
        return acc, avg_loss.data.item()
    
    def evaluate(self, print_log=True):
        return self.test(-1, print_log)


    def build(self):
        logger.info(f'loading {self.prefix}_{self.model_name}...')

        if self.is_train:
            self._print_train_cfg()
            for epoch in range(1, self.epochs+1):
                self.train(epoch)
                test_acc, test_loss = self.test(epoch)
                
                if self.scheduler != None:
                    self.scheduler.step()
        else:
            test_acc, test_loss = self.evaluate()
        logger.info(f'The trained model is saved in {self.save_path}\n')        
        torch.save(self.model, self.save_path)
        
        summary_dict = self.model_summary(test_acc, test_loss, self.model, self.origin_summary)
        
        return self.model, summary_dict
        
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