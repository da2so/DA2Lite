import os
import argparse


from DA2Lite.trainer.classification import Classification
from DA2Lite.core.config import CfgUtil
from DA2Lite.core.log import setup_logger, get_logger
from DA2Lite.core.compress import Compressor
from DA2Lite.converter.converter import Converter

logger = get_logger(__name__)

def get_parser():
    parser = argparse.ArgumentParser(description="NET2net for builtin configs")
    parser.add_argument(
        "--train_config_file",
        default="configs/train/cifar10/cifar10_vgg16_bn.yaml",
        metavar="FILE",
        help="path to train config file",
    )
    parser.add_argument(
        "--compress_config_file",
        default="configs/compress/eagleeye_fskd.yaml",
        metavar="FILE",
        help="path to compress config file",
    )
    return parser


if __name__ == '__main__':
    save_dir = setup_logger()

    args = get_parser().parse_args()
    cfg_util = CfgUtil(args, save_dir)
    
    device = cfg_util.get_device()

    logger.info(f'Loading {cfg_util.cfg.DATASET.NAME} dataset ...\n')

    train_loader, test_loader = cfg_util.load_dataset()
    
    model = cfg_util.load_model()
    
    trainer = Classification(cfg_util=cfg_util,
                            train_cfg=cfg_util.cfg.TRAIN,
                            prefix='origin',
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            device=device)

    trained_model, origin_summary = trainer.build()
    
    DA2lite = Compressor(cfg_util=cfg_util,
                        model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        origin_summary=origin_summary,
                        device=device)
    DA2lite.build()

    converter = Converter(model, DA2lite.trainer.save_path[:-3] + "_script.pt\n", (16, 3, 32, 32))     # TODO 2022.03.14 config 파일에 IMAGE_SHAPE 이용하도록 변경 해야함
    converter.to_torchscript()
