"""
@Fire
https://github.com/fire717
"""
from lib import init, Data, MoveNet, Task
from config import cfg
from lib.utils.utils import arg_parser, update_tuner_cfg
import os, json, torch
import torch.nn as nn


def main(cfg):
    init(cfg)

    old_classes = 13
    new_classes = 7

    label_config = f'{os.path.join(cfg["save_dir"], cfg["label"])}/cfg.txt'
    if os.path.exists(label_config):
        with open(label_config, 'r') as file:
            cfg_temp = json.load(file)
        update_tuner_cfg(cfg,cfg_temp)
    # cfg["ckpt"] = f'{cfg["save_dir"]}/{cfg["label"]}/newest.json'
    cfg['ckpt'] = "/home/ggoyal/code/hpe-core/example/movenet/models/e97_valacc0.81209.pth"

    model = MoveNet(num_classes=old_classes,
                    width_mult=cfg["width_mult"],
                    mode='train')
    # model2 = MoveNet(num_classes=7,
    #                 width_mult=cfg["width_mult"],
    #                 mode='train')


    # print(model.header.header_heatmaps[1])
    # data = Data(cfg)
    # train_loader, val_loader = data.getTrainValDataloader()
    #
    run_task = Task(cfg, model)
    run_task.modelLoad(model_path=cfg["ckpt"])

    header_heatmaps_save = model.header.header_heatmaps[1]
    header_regs_save = model.header.header_regs[1]
    header_offsets_save = model.header.header_offsets[1]

    model.header.header_heatmaps[1] = nn.Conv2d(96, new_classes, 1, 1, 0, bias=True)
    model.header.header_regs[1] = nn.Conv2d(96, new_classes*2, 1, 1, 0, bias=True)
    model.header.header_offsets[1] = nn.Conv2d(96, new_classes*2, 1, 1, 0, bias=True)
# with open(f'{os.path.join(cfg["save_dir"], cfg["label"])}/cfg.txt', 'w') as convert_file:
    #     convert_file.write(json.dumps(cfg))
    # run_task.train(train_loader, val_loader)
    with torch.no_grad():
        model.header.header_heatmaps[1].weight.copy_(header_heatmaps_save.weight[:new_classes,:,:,:])
        model.header.header_heatmaps[1].bias.copy_(header_heatmaps_save.bias[:new_classes])

        model.header.header_regs[1].weight.copy_(header_regs_save.weight[:new_classes*2,:,:,:])
        model.header.header_regs[1].bias.copy_(header_regs_save.bias[:new_classes*2])

        model.header.header_offsets[1].weight.copy_(header_offsets_save.weight[:new_classes*2,:,:,:])
        model.header.header_offsets[1].bias.copy_(header_offsets_save.bias[:new_classes*2])

    run_task.modelSave('/home/ggoyal/data/models/7joints_base.pth')

if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)
