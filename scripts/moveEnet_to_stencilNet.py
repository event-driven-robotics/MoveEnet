
"""
@Fire
https://github.com/fire717
"""

from lib import init, MoveNet, Movenet_stencil, Task
from config import cfg
from lib.utils.utils import arg_parser



def main(cfg):
    init(cfg)

    cfg['ckpt'] = "/home/ggoyal/code/hpe-core/example/movenet/models/e97_valacc0.81209.pth"

    model_moveEnet = MoveNet(num_classes=13,
                    width_mult=cfg["width_mult"],
                    mode='train')

    model_stencil = Movenet_stencil(num_classes=13,
                    width_mult=cfg["width_mult"],
                    mode='train')
    model_stencil._initialize_weights()

    run_task = Task(cfg, model_moveEnet)
    run_task.modelLoad(model_path=cfg["ckpt"])

    run_task_other = Task(cfg, model_stencil)
    params_stencil = model_stencil.state_dict()
    params_original = model_moveEnet.state_dict()

    params_stencil_dict = dict(params_stencil)

    for name, value in params_original.items():
        # for submods in modules:
        if name in params_stencil_dict.keys():
            print(name)
            params_stencil[name] = value

    run_task_other.modelSave('/home/ggoyal/data/models/stencil_base.pth')

if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)


