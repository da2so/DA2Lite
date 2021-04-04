import os
from abc import ABC, abstractmethod

from ptflops import get_model_complexity_info

class TrainerBase(ABC):
    def __init__(
        self,
        cfg_util,
        prefix,
        model,
        train_loader,
        test_loader,
        device
        ):
        
        self.cfg = cfg_util.cfg

        self.device = device
        self.img_shape = self.cfg.DATASET.IMG_SHAPE
        self.model = model.to(self.device)
        self.model_name = self.cfg.MODEL.NAME
        self.dataset_name = self.cfg.DATASET.NAME
        self.train_loader = train_loader
        self.test_loader = test_loader

        save_dir = os.path.join(self.cfg.SAVE_DIR, 'models/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_name = f'{prefix}_{self.dataset_name}_{self.model_name}.pt'
        self.save_path = os.path.join(save_dir, file_name)

    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raise NotImplementedError


    def model_summary(self):
        macs, params = get_model_complexity_info(self.model, self.img_shape, as_strings=True,
                                                verbose=True)
        print(f'The number of parameters: {params}')
        print(f'Computational complexity: {macs}')
        
