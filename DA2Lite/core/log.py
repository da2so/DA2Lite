import time
import logging
import sys
import os

BASE_DIR = './log/' 
BASIC_FMT = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")


def _get_logdir():
    now = time.localtime()
    sub_dir = "%04d-%02d-%02d.%02d:%02d:%02d/" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    save_dir = os.path.join(BASE_DIR, sub_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def setup_logger():
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(BASIC_FMT)
    root_logger.addHandler(ch)

    save_dir = _get_logdir()

    filename = os.path.join(save_dir, 'train.log')
    fh = logging.FileHandler(filename = filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(BASIC_FMT)
    root_logger.addHandler(fh)
    
    return save_dir


def get_logger(name):

    logger = logging.getLogger(name)
    logger.info(f"{name} Logger Initialized")
    return logger