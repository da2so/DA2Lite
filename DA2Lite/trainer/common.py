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
        train_loader,
        test_loader,
        device):
        
        self.cfg = cfg_util.cfg

        self.device = device
        self.img_shape = self.cfg.DATASET.IMG_SHAPE
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

    def model_summary(self, test_acc, test_loss, model, origin_summary):
        model = copy.deepcopy(model)
        macs, params = get_model_complexity_info(model=model,
                                                input_res=tuple(self.img_shape),
                                                print_per_layer_stat=False,
                                                as_strings=False,
                                                verbose=False)
        del model
        model_memory = self._get_file_size(self.save_path)

        acc_txt =           ' Test Accuracy (%)            '
        loss_txt =          ' Test loss                    '
        param_num_txt =     ' Number of parameters (M)     '
        complexity_txt =    ' Computational complexity (G) '
        file_size_txt =     ' File size (MB)               '

        len_txt = len(acc_txt)
        
        if origin_summary == None: # summary for an origin model 
            total_dummy = 50
            l_dummy = len_txt
            r_dummy = total_dummy - len_txt - 1
            edge_line = '-' * total_dummy

            model_name = ' '+self.prefix +'_model  '
            
            acc =           f' {round(test_acc*100.0, 2)} %'
            loss =          f' {round(test_loss, 4)}'
            param_num =     f' {round(params * 1e-6, 2)} M'
            complexity =    f' {round(macs * 1e-9, 2)} G'
            file_size =     f' {round(model_memory / (1024 * 1024), 2)} MB'

            logger.info(f'+{edge_line}+')
            logger.info(f'|{" ".ljust(l_dummy)}|{model_name.ljust(r_dummy)}|')
            logger.info(f'+{edge_line}+')
            logger.info(f'|{" ".ljust(l_dummy)}|{" ".ljust(r_dummy)}|')
            logger.info(f'|{acc_txt}|{acc.ljust(r_dummy)}|')
            logger.info(f'|{loss_txt}|{loss.ljust(r_dummy)}|')
            logger.info(f'|{param_num_txt}|{param_num.ljust(r_dummy)}|')
            logger.info(f'|{complexity_txt}|{complexity.ljust(r_dummy)}|')
            logger.info(f'|{file_size_txt}|{file_size.ljust(r_dummy)}|')
            logger.info(f'|{" ".ljust(l_dummy)}|{" ".ljust(r_dummy)}|')
            logger.info(f'+{edge_line}+\n')

            summary_dict = {'acc': test_acc, 'loss': test_loss, 'param_num': params,
                            'complexity': macs, 'file_size': model_memory, 'prefix': self.prefix}
        
        else: #summary for a compressed model
        
            total_dummy = 100
            l_dummy = len_txt
            r_dummy = (total_dummy - len_txt - 3 ) // 3

            edge_line = '-' * total_dummy

            origin_model_name = ' ' + origin_summary['prefix'] + '_model  '
            model_name = ' '+self.prefix + '_model  '
            enh_name = ' enhancement'

            ori_acc =           f' {round(origin_summary["acc"] * 100.0, 2)} %'
            ori_loss =          f' {round(origin_summary["loss"], 4)}'
            ori_param_num =     f' {round(origin_summary["param_num"] * 1e-6,2)} M'
            ori_complexity =    f' {round(origin_summary["complexity"] * 1e-9,2)} G'
            ori_file_size =     f' {round(origin_summary["file_size"] / (1024 * 1024), 2)} MB'

            acc =           f' {round(test_acc * 100.0, 2)} %'
            loss =          f' {round(test_loss, 4)}'
            param_num =     f' {round(params * 1e-6, 2)} M'
            complexity =    f' {round(macs * 1e-9, 2)} G'
            file_size =     f' {round(model_memory / (1024 * 1024), 2)} MB'

            enh_acc = f' {round((test_acc - origin_summary["acc"]) * 100.0, 2)} %'
            enh_loss = f' {round(abs(origin_summary["loss"] - test_loss), 2)}'
            enh_param_num = f' {round(origin_summary["param_num"] / params, 2)}x'
            enh_complexity = f' {round(origin_summary["complexity"] / macs, 2)}x'
            enh_file_size = f' {round(origin_summary["file_size"] / model_memory, 2)}x'

            logger.info(f'+{edge_line}+')
            logger.info(f'|{" ".ljust(l_dummy)}|{origin_model_name.ljust(r_dummy)}|{model_name.ljust(r_dummy)}|{enh_name.ljust(r_dummy)} |')
            logger.info(f'+{edge_line}+')
            logger.info(f'|{" ".ljust(l_dummy)}|{" ".ljust(r_dummy)}|{" ".ljust(r_dummy)}|{" ".ljust(r_dummy)} |')
            logger.info(f'|{acc_txt}|{ori_acc.ljust(r_dummy)}|{acc.ljust(r_dummy)}|{enh_acc.ljust(r_dummy)} |')
            logger.info(f'|{loss_txt}|{ori_loss.ljust(r_dummy)}|{loss.ljust(r_dummy)}|{enh_loss.ljust(r_dummy)} |')
            logger.info(f'|{param_num_txt}|{ori_param_num.ljust(r_dummy)}|{param_num.ljust(r_dummy)}|{enh_param_num.ljust(r_dummy)} |')
            logger.info(f'|{complexity_txt}|{ori_complexity.ljust(r_dummy)}|{complexity.ljust(r_dummy)}|{enh_complexity.ljust(r_dummy)} |')
            logger.info(f'|{file_size_txt}|{ori_file_size.ljust(r_dummy)}|{file_size.ljust(r_dummy)}|{enh_file_size.ljust(r_dummy)} |')
            logger.info(f'|{" ".ljust(l_dummy)}|{" ".ljust(r_dummy)}|{" ".ljust(r_dummy)}|{" ".ljust(r_dummy)} |')
            logger.info(f'+{edge_line}+\n')

            summary_dict = {'acc': test_acc, 'loss': test_loss, 'param_num': params,
                            'complexity': macs, 'file_size': model_memory}
            
        return summary_dict