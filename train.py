"""
@Fire
https://github.com/fire717
"""
from lib import init, Data, MoveNet, Task
from config import cfg
from lib.utils.utils import arg_parser, update_tuner_cfg
import os, json
import torch
import lightning as pl


def main(cfg):
    # init(cfg)
    # label_config = f'{os.path.join(cfg["save_dir"], cfg["label"])}/cfg.txt'
    # if os.path.exists(label_config):
    #     with open(label_config, 'r') as file:
    #         cfg_temp = json.load(file)
    #     update_tuner_cfg(cfg,cfg_temp)
    # cfg["ckpt"] = f'{cfg["save_dir"]}/{cfg["label"]}/newest.json'
    # model = MoveNet(num_classes=cfg["num_classes"],
    #                 width_mult=cfg["width_mult"],
    #                 mode='train',
    #                 cfg=cfg)
    # data = Data(cfg)
    # train_loader, val_loader = data.getTrainValDataloader()
    #
    # model.eval()
    # with torch.no_grad():
    #     batch = train_loader.dataset[0]
    #     pred = model(batch)
    # # dataloader = DataLoader(dataset)
    # # trainer = pl.Trainer()
    # # trainer.fit(model=model, train_dataloaders=train_loader)
    # # run_task = Task(cfg, model)
    # # run_task.modelLoad(model_path=cfg["ckpt"])
    # # with open(f'{os.path.join(cfg["save_dir"], cfg["label"])}/cfg.txt', 'w') as convert_file:
    # #     convert_file.write(json.dumps(cfg))
    # # run_task.train(train_loader, val_loader)

    init(cfg)
    cfg['ckpt'] = "/home/ggoyal/code/hpe-core/example/movenet/models/e97_valacc0.81209.pth"

    device = torch.device('cpu')
    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train',
                    cfg=cfg)  # .load_from_checkpoint(cfg['ckpt'],map_location=device)
    # with open("/home/ggoyal/code/hpe-core/example/movenet/models/e97_valacc0.81209.pth", 'rb') as f:
    #     model = pickle.load(f, encoding='ascii')
    torch.load(cfg['default_model'], map_location=device)

    data = Data(cfg)
    train_loader, val_loader = data.getTrainValDataloader()

    # run_task = Task(cfg, model)
    # run_task.modelLoad(cfg["ckpt"])

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    predictions = trainer.predict(model, dataloaders=test_loader)
    print(predictions[0])


if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)
