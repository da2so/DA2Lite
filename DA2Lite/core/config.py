from yacs.config import CfgNode as CN
import yaml

from torch.utils.data import DataLoader 
import torch

import DA2Lite.network as network
import DA2Lite.data as data
import DA2Lite.trainer.utils as utils


def get_cfg_defaults():

    _C = CN()
    
    ################
    # Train Config #
    ################
    
    _C.DATASET = CN()
    _C.DATASET.NAME = 'cifar10'
    _C.DATASET.IMG_SHAPE = [3, 32, 32]
    _C.DATASET.DATA_AUG = True

    _C.MODEL = CN()
    _C.MODEL.NAME = 'resnet18'
    _C.MODEL.NUM_CLASSES = 10
    _C.MODEL.PRE_TRAINED = CN()
    _C.MODEL.PRE_TRAINED.IS_USE = True
    _C.MODEL.PRE_TRAINED.RETRAINED = False
    _C.MODEL.PRE_TRAINED.PATH = './saved_models/cifar10_resnet18.pt'

    _C.TRAIN = CN()
    _C.TRAIN.OPTIMIZER = 'sgd'
    _C.TRAIN.LOSS = 'categorical_crossentropy'
    _C.TRAIN.SCHEDULER = 'stepLR'
    _C.TRAIN.BATCH_SIZE = 256
    _C.TRAIN.EPOCHS = 150
    _C.TRAIN.LR = 0.1

    _C.SAVE_DIR = './log/'
    
    _C.GPU = CN()
    _C.GPU.IS_USE = True

    ######################
    # Compression Config #
    ######################
    
    _C.PRUNING = CN()

    _C.PRUNING.POST_TRAIN = CN()
    _C.PRUNING.POST_TRAIN.IS_USE = True
    _C.PRUNING.POST_TRAIN.OPTIMIZER = 'sgd'
    _C.PRUNING.POST_TRAIN.LOSS = 'categorical_crossentropy'
    _C.PRUNING.POST_TRAIN.SCHEDULER = 'stepLR'
    _C.PRUNING.POST_TRAIN.BATCH_SIZE = 256
    _C.PRUNING.POST_TRAIN.EPOCHS = 30
    _C.PRUNING.POST_TRAIN.LR = 0.001

    _C.PRUNING.STRATEGY = CN()
    _C.PRUNING.STRATEGY.NAME = 'MinMaxStrategy'
    _C.PRUNING.STRATEGY.PRUNING_RATIO = [0.0, 0.7]  

    _C.PRUNING.CRITERIA = CN()
    _C.PRUNING.CRITERIA.NAME = 'L1Critera'
    
    return _C.clone()

class CfgUtil():

    def __init__(self, args, save_dir):

        self.cfg = get_cfg_defaults()
        for key ,value in vars(args).items():
            self.cfg.merge_from_file(value)

        self.cfg.SAVE_DIR = save_dir
        self.cfg.freeze()
        
        self.train_config = self.cfg.TRAIN

    def load_model(self):

        model_cfg = self.cfg.MODEL
        
        def get_model_arch():
            try:
                model_func = getattr(network, model_cfg.NAME) 
            except:
                raise ValueError(f'Invalid model name: {model_cfg.NAME}')
        
            model_arch = model_func(model_cfg.NUM_CLASSES)
            return model_arch

        if model_cfg.PRE_TRAINED.IS_USE:
            if torch.typename(torch.load(model_cfg.PRE_TRAINED.PATH)) == 'OrderedDict':

                """
                if you want to use customized model that has a type 'OrderedDict',
                you shoud load model object as follows:
                
                from Net import Net()
                model=Net()
                """ 
                model = get_model_arch()
                try:
                    model.load_state_dict(torch.load(model_cfg.PRE_TRAINED.PATH))
                except:
                    raise ValueError(f'An invalid model was loaded.')
            else:
                model = torch.load(model_cfg.PRE_TRAINED.PATH)
        else:
            model = get_model_arch()
        return model

    def load_dataset(self, data_dir='./dataset/'):
        dt = self.cfg.DATASET
        bs = self.train_config.BATCH_SIZE

        try: 
            dataset_func = getattr(data, dt.NAME)
        except:
            raise ValueError(f'Invalid dataset name: {dt.NAME}')

        train_dt, test_dt = dataset_func(dt.DATA_AUG, dt.IMG_SHAPE, data_dir)
        
        train_loader = DataLoader(train_dt, batch_size=bs, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dt, batch_size=bs, num_workers=0)
        
        return train_loader, test_loader

    
    def get_loss(self):

        loss_name = self.train_config.LOSS
        try:
            loss_func = getattr(utils, loss_name)
        except:
            raise ValueError(f'Invalid loss name: {loss_name}')
        
        loss = loss_func()

        return loss


    def get_optimizer(self, model):

        optimizer_name = self.train_config.OPTIMIZER
        lr = self.train_config.LR
        
        try: 
            optimizer_func = getattr(utils, optimizer_name)
        except:
            raise ValueError(f'Invalid optimizer name: {optimizer_name}')

        optimizer = optimizer_func(model, lr)
        return optimizer


    def get_scheduler(self, optimizer):

        scheduler_name = self.train_config.SCHEDULER
        
        try: 
            scheduler_func = getattr(utils, scheduler_name)
        except:
            raise ValueError(f'Invalid scheduler name: {scheduler_name}')

        scheduler = scheduler_func(optimizer)
        return scheduler

    def get_device(self):
        
        gpu = self.cfg.GPU
        device = torch.device("cuda" if gpu.IS_USE else "cpu")

        return device