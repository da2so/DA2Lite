import os
from typing import List, Tuple
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader 

from DA2Lite.trainer.common import TrainerBase
from DA2Lite.core.log import get_logger

logger = get_logger(__name__)


class Classification(TrainerBase):
    def __init__(self, 
                 cfg_util,
                 train_obj,
                 prefix,
                 model,
                 train_loader,
                 test_loader,
                 device):

        super().__init__(cfg_util,
                        prefix,
                        model,
                        train_loader,
                        test_loader,
                        device)

        self.is_train = train_obj.IS_USE

        self.epochs = train_obj.EPOCHS
        self.optimizer = cfg_util.get_optimizer(self.model)
        self.loss = cfg_util.get_loss()

        self.scheduler = None
        if train_obj.SCHEDULER:
            self.scheduler = cfg_util.get_scheduler(self.optimizer)
    
    def train(self, epoch):

        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

        total_correct = 0
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader)-1, leave=False)
        for i, (images, labels) in loop:
            self.optimizer.zero_grad()

            images, labels = Variable(images), Variable(labels)

            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss(outputs, labels)

            pred = outputs.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

            loss.backward()
            self.optimizer.step()

            acc = float(total_correct) / len(self.train_loader.dataset)
            loop.set_description(f"Train - Epoch [{epoch}/{self.epochs}]")
            loop.set_postfix(Accuracy=acc, Loss=loss.item())

            if i == len(self.train_loader) -1:
                logger.debug(f'Train - Epoch [{epoch}/{self.epochs}] Accuracy: {acc}, Loss: {loss.item()}')
    
    
    def test(self, epoch):

        self.model.eval()

        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = Variable(images).to(self.device), Variable(labels).to(self.device)

                outputs = self.model(images)
                avg_loss += self.loss(outputs, labels).sum()
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        avg_loss /= len(self.test_loader.dataset)
        acc = float(total_correct) / len(self.test_loader.dataset)
        
        if epoch != -1:
            logger.debug(f'Test  - Epoch [{epoch}/{self.epochs}] Accuracy: {acc}, Loss: {avg_loss.data.item()}')
        #else:
        #    logger.info(f'Test Accuracy: {acc}, Loss {avg_loss.data.item()}')
        return acc, avg_loss.data.item()
    
    def evaluate(self):
        return self.test(-1)


    def build(self):
        logger.info(f'loading {self.prefix}_{self.model_name}..')
        

        if self.is_train:
            for epoch in range(1, self.epochs+1):
                self.train(epoch)
                self.test(epoch)
                
                if self.scheduler != None:
                    self.scheduler.step()
        
        logger.info(f'The trained model is saved in {self.save_path}\n')        
        torch.save(self.model.state_dict(), self.save_path)
        
        self.model_summary()

        return self.model