import os
import argparse


from DA2Lite.trainer.classification import Classification
from DA2Lite.core.config import CfgUtil
from DA2Lite.core.log import setup_logger
from DA2Lite.core.compress import Compressor



def get_parser():
    parser = argparse.ArgumentParser(description="NET2net for builtin configs")
    parser.add_argument(
        "--train-config-file",
        default="configs/train/cifar10_densenet121.yaml",
        metavar="FILE",
        help="path to train config file",
    )
    parser.add_argument(
        "--compress-config-file",
        default="configs/compress/eagleeye.yaml",
        metavar="FILE",
        help="path to compress config file",
    )
    return parser


if __name__ == '__main__':
    save_dir = setup_logger()

    args = get_parser().parse_args()
    cfg_util = CfgUtil(args, save_dir)
    
    device = cfg_util.get_device()

    train_loader, test_loader = cfg_util.load_dataset()
    
    model = cfg_util.load_model()
    
    trainer = Classification(cfg_util=cfg_util,
                            train_obj=cfg_util.cfg.TRAIN,
                            prefix='origin',
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            device=device)

    trained_model = trainer.build()
    

    model = model.cuda()

    DA2lite = Compressor(cfg_util=cfg_util,
                        model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=device)
    DA2lite.build()
    