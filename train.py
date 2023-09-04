"""
@Fire
https://github.com/fire717
"""
from lib import init, Data, MoveNet, Task, Movenet_stencil
from config import cfg
from lib.utils.utils import arg_parser, update_tuner_cfg
import os, json


def main(cfg):
    init(cfg)

    cfg['img_path'] = '/home/ggoyal/data/h36m_cropped/tester/h36m_stencil'
    cfg['train_label_path'] = '/home/ggoyal/data/h36m_cropped/tester/stencil_anno/poses.json'
    cfg['val_label_path'] = '/home/ggoyal/data/h36m_cropped/tester/stencil_anno/poses.json'
    cfg['batch_size'] = 1
    cfg['epochs'] = 1


    label_config = f'{os.path.join(cfg["save_dir"], cfg["label"])}/cfg.txt'
    if os.path.exists(label_config):
        with open(label_config, 'r') as file:
            cfg_temp = json.load(file)
        update_tuner_cfg(cfg,cfg_temp)
    cfg["ckpt"] = f'{cfg["save_dir"]}/{cfg["label"]}/newest.json'
    data = Data(cfg)


    if cfg['architecture'] == 'mobilenet2':
        model = MoveNet(num_classes=cfg["num_classes"], width_mult=cfg["width_mult"], mode='train')
        train_loader, val_loader = data.getTrainValDataloader()
    elif cfg['architecture'] == 'spiking':
        model = Movenet_stencil(num_classes=cfg["num_classes"], width_mult=cfg["width_mult"], mode='train')
        train_loader, val_loader = data.getTrainValDataloader_spike()
    else:
        print("Model architecture name invalid. please check options in the config file and retry.")


    run_task = Task(cfg, model)
    # run_task.modelLoad(model_path=cfg["ckpt"])
    model._initialize_weights()
    # with open(f'{os.path.join(cfg["save_dir"], cfg["label"])}/cfg.txt', 'w') as convert_file:
    #     convert_file.write(json.dumps(cfg))
    run_task.train(train_loader, val_loader)


if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)
