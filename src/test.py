# -*- coding: utf-8 -*-
import time

import hydra
import pytorch_lightning as pl
from omegaconf import open_dict
from pytorch_lightning import Trainer

from globalenv import *
from utils.util import parse_config

pl.seed_everything(GLOBAL_SEED)


@hydra.main(config_path='config', config_name="config")
def main(opt):
    opt = parse_config(opt, TEST)
    print('Running config:', opt)
    from model.lcdpnet import LitModel as ModelClass
    ckpt = opt[CHECKPOINT_PATH]
    assert ckpt
    model = ModelClass.load_from_checkpoint(ckpt, opt=opt)
    # model.opt = opt
    with open_dict(opt):
        model.opt[IMG_DIRPATH] = model.build_test_res_dir()
        opt.mode = 'test'
    print(f'Loading model from: {ckpt}')

    from data.img_dataset import DataModule
    datamodule = DataModule(opt)

    trainer = Trainer(
        gpus=opt[GPU],
        strategy=opt[BACKEND],
        precision=opt[RUNTIME_PRECISION])

    beg = time.time()
    trainer.test(model, datamodule)
    print(f'[ TIMER ] Total time usage: {time.time() - beg}')
    print('[ PATH ] The results are in :')
    print(model.opt[IMG_DIRPATH])


if __name__ == "__main__":
    main()
