import os
import argparse


from DA2Lite.trainer.classification import Classification
from DA2Lite.core.config import CfgUtil
from DA2Lite.core.log import setup_logger

import sys


def get_parser():
    parser = argparse.ArgumentParser(description="NET2net for builtin configs")
    parser.add_argument(
        "--train-config-file",
        default="configs/train/a.yaml",
        metavar="FILE",
        help="path to train config file",
    )
    parser.add_argument(
        "--compress-config-file",
        default="configs/compress/a.yaml",
        metavar="FILE",
        help="path to compress config file",
    )
    return parser


if __name__ == '__main__':
    save_dir = setup_logger()

    args = get_parser().parse_args()
    cfg_util = CfgUtil(args, save_dir)
    
    device = cfg_util.get_device()

    #train_loader, test_loader = cfg_util.load_dataset()
    
    train_loader, test_loader = 1,1
    model = cfg_util.load_model()
    """
    trainer = Classification(cfg_util=cfg_util,
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            device=device)

    trained_model = trainer.build()
    """
    from DA2Lite.compression.pruning.eagleeye import EagleEye
    model = model.cuda()
    eagleeye_obj = EagleEye(cfg_util=cfg_util,
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            device=device)
    eagleeye_obj.build()