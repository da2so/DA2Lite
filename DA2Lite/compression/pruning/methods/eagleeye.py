
import torch

from DA2Lite.compression.pruning.methods.lN_norm import L1Criteria
from DA2Lite.compression.pruning.methods.common import CriteriaBase


class EagleEye(CriteriaBase):
    def __init__(self, **kwargs):
        criteria_args = kwargs['criteria_args']
        
        self.num_candidates = criteria_args.NUM_CANDIDATES

    def get_prune_idx(self, weights, amount=0.0):
        indices = L1Criteria(weights, amount)
        return indices

    def get_model(self, pruned_model, pruning_info, best_model, train_loader, device, **kwargs):

        idx = kwargs['idx']

        new_model.to(device)
        new_model.train()

        max_iters = 10
        max_samples = len(train_loader.dataset) // 30
        batch_size = train_loader.batch_size
        batches = 0

        for j in range(max_iters):
            for images, label in train_loader:
                batches += batch_size
                images = images.to(device)
                out = new_model(images)

                if batches >= max_samples:
                    break
        
        new_model.eval()

        total_correct = 0            
        with torch.no_grad():
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                outputs = new_model(images)
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        acc = float(total_correct) / len(train_loader.dataset)
        print(f'Adaptive-BN-based accuracy for {idx}-th prunned model: {acc}')

        if best_model['acc'] < acc:
            best_model['acc'] = acc
            best_model['model'] = new_model
            best_model['index'] = idx
            best_model['pruning_info'] = pruning_info
        
    return best_model