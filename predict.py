"""
@Fire
https://github.com/fire717
"""

from lib import init, Data, MoveNet, Task

from config import cfg
from lib.utils.utils import arg_parser
import torch
import lightning as pl
import pickle
# Script to create and save as images all the various outputs of the model


def main(cfg):

    init(cfg)
    cfg['ckpt'] = "/home/ggoyal/code/hpe-core/example/movenet/models/e97_valacc0.81209.pth"

    device = torch.device('cpu')
    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train',
                    cfg=cfg) #.load_from_checkpoint(cfg['ckpt'],map_location=device)
    # with open("/home/ggoyal/code/hpe-core/example/movenet/models/e97_valacc0.81209.pth", 'rb') as f:
    #     model = pickle.load(f, encoding='ascii')
    torch.load(cfg['ckpt'], map_location=device)
    
    data = Data(cfg)
    test_loader = data.getTestDataloader()
    train_loader,val_loader = data.getTrainValDataloader()


    # run_task = Task(cfg, model)
    # run_task.modelLoad(cfg["ckpt"])


    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)
    predictions = trainer.predict(model, dataloaders=test_loader)
    print(predictions[0])
    # run_task.predict(test_loader, cfg["predict_output_path"])
    # run_task.predict(test_loader, "output/predict")



if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)