"""
@Fire
https://github.com/fire717
"""

from config import cfg
from lib import init, Data, MoveNet, Task


def main(cfg):

    init(cfg)

    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    # print(model)

    data = Data(cfg)
    train_loader, train_sampler, val_loader, val_sampler = data.getTrainValDataloader()
    # data.showData(train_loader)

    run_task = Task(cfg, model)
    run_task.modelLoad(cfg["newest_ckpt"])
    run_task.train(train_loader, train_sampler, val_loader, val_sampler)


if __name__ == '__main__':
    main(cfg)
