import os
import copy
from abc import ABC, abstractmethod
from ptflops import get_model_complexity_info

from DA2Lite.core.log import get_logger

logger = get_logger(__name__)

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
        self.prefix = prefix
        self.model_name = self.cfg.MODEL.NAME
        self.dataset_name = self.cfg.DATASET.NAME
        self.train_loader = train_loader
        self.test_loader = test_loader

        save_dir = os.path.join(self.cfg.SAVE_DIR, 'models/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_name = f'{self.prefix}_{self.dataset_name}_{self.model_name}.pt'
        self.save_path = os.path.join(save_dir, file_name)

    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def build(self):
        raise NotImplementedError


    def _get_file_size(self, file_path):
        size = os.path.getsize(file_path)
        return size

    def model_summary(self):

        copy_model = copy.deepcopy(self.model)
        macs, params = get_model_complexity_info(model=copy_model,
                                                input_res=tuple(self.img_shape),
                                                print_per_layer_stat=False,
                                                as_strings=False,
                                                verbose=False)
        
        acc, avg_loss = self.evaluate()
        file_size = self._get_file_size(self.save_path)

        num_dummy = 60
        model_name = '  '+self.prefix +'_model  '
        dummy_line = '-'*num_dummy

        acc =           f'Test Accuracy (%)            : {round(acc*100.0, 2)} %'
        loss =          f'Test loss                    : {round(avg_loss, 4)}'
        param_num =     f'The number of parameters (M) : {round(params * 1e-6,2)} M'
        complexity =    f'Computational complexity (G) : {round(macs * 1e-9,2)} G'
        file_size =     f'File size (MB)               : {round(file_size / (1024 * 1024), 2)} MB'

        logger.info(f'{dummy_line}')
        logger.info(f'{model_name.center(num_dummy,"-")}')
        logger.info(f'{" ".ljust(num_dummy)}')
        logger.info(f'{acc.ljust(num_dummy)}')
        logger.info(f'{loss.ljust(num_dummy)}')
        logger.info(f'{param_num.ljust(num_dummy)}')
        logger.info(f'{complexity.ljust(num_dummy)}')
        logger.info(f'{file_size.ljust(num_dummy)}')
        logger.info(f'{" ".ljust(num_dummy)}')
        logger.info(f'{dummy_line}\n')