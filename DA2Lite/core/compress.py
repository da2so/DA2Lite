

from DA2Lite import compression
from DA2Lite.trainer.classification import Classification
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
    
    def build(self):

        compressed_model = self.origin_model
        compress_num = 1
        for compress_name in self.compress_step:

            compress_class = self._get_compressor(cfg_to_compress[compress_name])
            
            compress_cfg = self.cfg[compress_name].METHOD

            
            compressor = compress_class(compress_cfg=compress_cfg,
                                        model=compressed_model,
                                        device=self.device,
                                        cfg=self.cfg,
                                        train_loader=self.train_loader)

            compressed_model = compressor.build()

            train_cfg = self.cfg[compress_name].POST_TRAIN
            prefix = f'compress_{compress_num}'

            trainer = Classification(cfg_util=self.cfg_util,
                                    train_obj=train_cfg,
                                    prefix=prefix,
                                    model=compressed_model,
                                    train_loader=self.train_loader,
                                    test_loader=self.test_loader,
                                    device=self.device)

            test_acc, test_loss = trainer.test(-1)
            logger.info(f'Test accuracy right after {compress_name}: {test_acc*1e2} %\n')
            compressed_model = trainer.build()

            import sys
            sys.exit()
