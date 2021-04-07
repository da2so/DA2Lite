from abc import abstractclassmethod, ABC
import random

import torch


class CriteriaBase(ABC):

    def __init__(self, **kwargs):
        self.num_candidates = 1 
    
    @abstractclassmethod
    def get_prune_idx(self, weights, pruning_ratio=0.0):
        raise NotImplementedError

    
    def get_model(self, pruned_model, pruining_ratio, best_model, train_loader, device, **kwargs):

        pruned_model.to(device)
        
        pruned_model.eval()

        total_correct = 0            
        with torch.no_grad():
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                outputs = pruned_model(images)
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        acc = float(total_correct) / len(train_loader.dataset)

        if best_model['acc'] < acc:
            best_model['acc'] = acc
            best_model['model'] = pruned_model
            best_model['pruning_info'] = pruning_info
        
        return best_model