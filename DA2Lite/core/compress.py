

from DA2Lite import compression
from DA2Lite import trainer
from DA2Lite.core.log import get_logger

logger = get_logger(__name__)

cfg_to_compress = {'PRUNING': 'Pruner', 'FD': 'FilterDecomposition'}

class Compressor(object):
    def __init__(self,
                cfg_util,
                model,
                train_loader,
                test_loader,
                device):
        
        self.cfg_util = cfg_util
        self.cfg = cfg_util.cfg
        self.compress_step = []
        for i_cfg in self.cfg:
            if i_cfg in cfg_to_compress:
                self.compress_step.append(i_cfg)

        self.origin_model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        logger.info(f'Compression Start!\n')

    def _get_compressor(self, compress_name):
        try:
            compressor_class = getattr(compression, compress_name)
        except:
            raise ValueError(f'Invalid compress name: {compress_name}')
        
        return compressor_class
    
    def _get_trainer(self, trainer_name):
        if trainer_name == 'basic':
            trainer_class = getattr(trainer, 'Classification')
        else:
            trainer_class = getattr(trainer, 'KnowledgeDistillation')
        
        return trainer_class

    def build(self):

        compressed_model = self.origin_model
        compress_num = 1
        for compress_name in self.compress_step:

            compress_class = self._get_compressor(cfg_to_compress[compress_name])
            
            compress_cfg = self.cfg[compress_name].METHOD

            self._print_compress_cfg(compress_cfg)

            compressor = compress_class(compress_cfg=compress_cfg,
                                        model=compressed_model,
                                        device=self.device,
                                        cfg=self.cfg,
                                        train_loader=self.train_loader)

            compressed_model = compressor.build()

            pruning_node_info = None
            if cfg_to_compress[compress_name] == 'Pruner':
                pruning_node_info = compressor.get_pruning_node_info()

            train_cfg = self.cfg[compress_name].POST_TRAIN
            prefix = f'compress_{compress_num}'

            trainer_class = self._get_trainer(train_cfg.NAME)

            trainer = trainer_class(cfg_util=self.cfg_util,
                                    train_cfg=train_cfg,
                                    prefix=prefix,
                                    model=compressed_model,
                                    train_loader=self.train_loader,
                                    test_loader=self.test_loader,
                                    device=self.device,
                                    origin_model=self.origin_model,
                                    compress_name=cfg_to_compress[compress_name],
                                    pruning_node_info=pruning_node_info
                                    )

            test_acc, test_loss = trainer.evaluate(print_log=False)
            logger.info(f'Test accuracy right after {compress_name}: {test_acc*1e2:.2f} %\n')

            compressed_model = trainer.build()
            compress_num += 1
        logger.info(f'Compression End!\n')

    def _print_compress_cfg(self, compress_cfg):
        split_compress_cfg = str(compress_cfg).split('\n')
        
        num_dummy = 60
        train_txt = ' Compress configuration '.center(num_dummy)
        border_txt = "-"*num_dummy
        logger.info(f'+{border_txt}+')
        logger.info(f'|{train_txt}|')
        logger.info(f'+{border_txt}+')
        logger.info(f'|{" ".ljust(num_dummy)}|')
        for i_comp_cfg in split_compress_cfg:
            logger.info(f'| {i_comp_cfg.ljust(num_dummy-1)}|')
        logger.info(f'|{" ".ljust(num_dummy)}|')
        logger.info(f'+{border_txt}+\n')
