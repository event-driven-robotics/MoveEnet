"""
@Fire
https://github.com/fire717
"""
from lib import init, Data, MoveNet, Task
from config import cfg
from lib.utils.utils import arg_parser, update_tuner_cfg
import os, json, yaml


def main(cfg):
    init(cfg)
    label_config = f'{os.path.join(cfg["save_dir"], cfg["label"])}/cfg.txt'
    if os.path.exists(label_config):
        with open(label_config, 'r') as file:
            cfg_temp = json.load(file)
        update_tuner_cfg(cfg,cfg_temp)
    cfg["ckpt"] = f'{cfg["save_dir"]}/{cfg["label"]}/newest.json'
    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    data = Data(cfg)
    train_loader, val_loader = data.getTrainValDataloader()

    run_task = Task(cfg, model)
    run_task.modelLoad(model_path=cfg["ckpt"])
    with open(f'{os.path.join(cfg["save_dir"], cfg["label"])}/cfg.yml', 'w') as convert_file:
        # convert_file.write(json.dumps(cfg)) #TODO make it yaml compatible
        yaml.dump(cfg, convert_file)
    run_task.train(train_loader, val_loader)


if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)
