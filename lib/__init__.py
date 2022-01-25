"""
@Fire
https://github.com/fire717
"""
import horovod.torch as hvd
import os
import torch

from lib.data.data import Data
from lib.models.movenet_mobilenetv2 import MoveNet
from lib.task.task import Task


from lib.utils.utils import setRandomSeed, printDash


def init(cfg):

    hvd.init()

    if cfg["cfg_verbose"]:
        printDash()
        print(cfg)
        printDash()

    cfg["cuda"] = cfg["cuda"] and torch.cuda.is_available()

    if cfg["cuda"]:
        # Horovod: pin GPU to local rank.
        torch.cuda.device(hvd.local_rank())

    # horovod: limit # of CPU threads to be used per worker
    torch.set_num_threads(4)

    setRandomSeed(cfg['random_seed'])

    if not os.path.exists(cfg['save_dir']):
        os.makedirs(cfg['save_dir'])
