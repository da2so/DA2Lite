import time
import logging
import sys
import os

BASE_DIR = './log/' 
BASIC_FMT = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%m/%d %M:%S")


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

    filename = os.path.join(save_dir, 'process.log')
    spr = logging.FileHandler(filename = filename)
    spr.setLevel(logging.INFO)
    spr.setFormatter(BASIC_FMT)
    root_logger.addHandler(spr)
    
    filename = os.path.join(save_dir, 'specific_process.log')
    pr = logging.FileHandler(filename = filename)
    pr.setLevel(logging.DEBUG)
    pr.setFormatter(BASIC_FMT)
    root_logger.addHandler(pr)

    return save_dir


def get_logger(name):

    logger = logging.getLogger(name)
    logger.info(f"{name} Logger Initialized")
    return logger