"""
@Fire
https://github.com/fire717
"""
from lib import init, Data, MoveNet, Task
from config import cfg
from lib.utils.utils import arg_parser


def main(cfg):
    init(cfg)

    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    data = Data(cfg)
    train_loader, val_loader = data.getTrainValDataloader()

    run_task = Task(cfg, model)
    run_task.modelLoad(model_path=cfg["ckpt"])
    run_task.train(train_loader, val_loader)


if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)
