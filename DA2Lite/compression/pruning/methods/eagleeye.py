
import torch

from DA2Lite.compression.pruning.methods.ln_norm import L1Criteria
from DA2Lite.compression.pruning.methods.criteria_base import CriteriaBase
from DA2Lite.core.log import get_logger

logger = get_logger(__name__)

class EagleEye(CriteriaBase):
    def __init__(self, **kwargs):
        criteria_args = kwargs['criteria_args']
        
        self.num_candidates = criteria_args.NUM_CANDIDATES
        
    def get_prune_idx(self, i_node, pruning_ratio=0.0):
        
        indices = L1Criteria().get_prune_idx(i_node, pruning_ratio)
        return indices

    def get_model(self, pruned_model, pruning_info, node_graph, best_model, train_loader, device, **kwargs):

        idx = kwargs['idx']
    
        pruned_model.to(device)
        pruned_model.train()

        max_iters = 10
        max_samples = len(train_loader.dataset) // 30
        batch_size = train_loader.batch_size
        batches = 0

        for j in range(max_iters):
            for images, labels in train_loader:
                batches += batch_size
                images = images.to(device)
                out = pruned_model(images)

                if batches >= max_samples:
                    break
        
        pruned_model.eval()

        total_correct = 0            
        with torch.no_grad():
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                outputs = pruned_model(images)
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        acc = float(total_correct) / len(train_loader.dataset)
        logger.info(f'Adaptive-BN-based accuracy for {idx}-th prunned model: {acc}')

        if best_model['acc'] < acc:
            best_model['acc'] = acc
            best_model['model'] = pruned_model
            best_model['node_graph'] = node_graph
            best_model['index'] = idx
            best_model['pruning_info'] = pruning_info
        
        del images, labels
        return best_model