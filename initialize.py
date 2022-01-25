
import horovod.torch as hvd
import json

from pathlib import Path

from config import cfg
from lib import MoveNet, Task


def main(cfg):

    hvd.init()

    if cfg["num_classes"] == 17:
        fullname = 'mbv2_e105_valacc0.80255.pth'
        with open(Path(cfg['newest_ckpt']).resolve(), 'w') as f:
            json.dump(fullname, f, ensure_ascii=False)
    else:
        model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')

        run_task = Task(cfg, model)
        run_task.modelSave('e0_accu0.pth')


if __name__ == '__main__':
    main(cfg)
