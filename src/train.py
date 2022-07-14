# -*- coding: utf-8 -*-
import os
import platform
import time
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import open_dict
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import utils.util as util
from globalenv import *
from utils.util import parse_config, init_logging
from data.img_dataset import DataModule

pl.seed_everything(GLOBAL_SEED)


@hydra.main(config_path='config', config_name="config")
def main(config):
    try:
        print('GPU status info:')
        os.system('nvidia-smi')
    except:
        ...

    # print(config)
    opt = parse_config(config, TRAIN)
    if opt.name == DEBUG:
        opt.debug = True

    if opt.debug:
        # ipdb.set_trace()
        mylogger = None
        if opt.checkpoint_path:
            continue_epoch = torch.load(
                opt.checkpoint_path, map_location=torch.device('cpu'))['global_step']
        debug_config = {
            DATALOADER_N: 0,
            NAME: DEBUG,
            LOG_EVERY: 1,
            VALID_EVERY: 1,
            NUM_EPOCH: 2 if not opt.checkpoint_path else continue_epoch + 2
        }
        opt.update(debug_config)
        debug_str = '[red]>>>> [[ WARN ]] You are in debug mode, update configs. <<<<[/red]'
        print(f'{debug_str}\n{debug_config}\n{debug_str}')

    else:
        # rename the exp
        spl = '_' if platform.system() == 'Windows' else ':'
        opt.name = f'{opt.runtime.modelname}{spl}{opt.name}@{opt.train_ds.name}'

        # trainer logger. init early to record all console output.
        mylogger = TensorBoardLogger(
            name=opt.name,
            save_dir=ROOT_PATH / 'tb_logs',
        )

    # init logging
    print('Running config:', opt)
    # opt[LOG_DIRPATH], opt.img_dirpath = init_logging(TRAIN, opt)
    with open_dict(opt):
        opt.log_dirpath, opt.img_dirpath = init_logging(TRAIN, opt)

    # load data
    # DataModuleClass = parse_ds_class(opt[TRAIN_DATA][CLASS])
    datamodule = DataModule(opt)

    # callbacks:
    callbacks = []
    if opt[EARLY_STOP]:
        print(
            f'Apply EarlyStopping when `{opt.checkpoint_monitor}` is {opt.monitor_mode}')
        callbacks.append(EarlyStopping(
            opt.checkpoint_monitor, mode=opt.monitor_mode))

    # callbacks:
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt[LOG_DIRPATH],
        save_last=True,
        save_top_k=5,
        mode=opt.monitor_mode,
        monitor=opt.checkpoint_monitor,
        save_on_train_epoch_end=True,
        every_n_epochs=opt.savemodel_every
    )
    callbacks.append(checkpoint_callback)

    if opt[AMP_BACKEND] != 'native':
        print(
            f'WARN: Running in APEX, mode: {opt[AMP_BACKEND]}-{opt[AMP_LEVEL]}')
    else:
        opt[AMP_LEVEL] = None

    # init trainer:
    trainer = pl.Trainer(
        gpus=opt[GPU],
        max_epochs=opt[NUM_EPOCH],
        logger=mylogger,
        callbacks=callbacks,
        check_val_every_n_epoch=opt[VALID_EVERY],
        num_sanity_val_steps=opt[VAL_DEBUG_STEP_NUMS],
        strategy=opt[BACKEND],
        precision=opt[RUNTIME_PRECISION],
        amp_backend=opt[AMP_BACKEND],
        amp_level=opt[AMP_LEVEL],
        **opt.flags
    )
    print('Trainer initailized.')

    # training loop
    from model.lcdpnet import LitModel as ModelClass
    if opt.checkpoint_path and not opt.resume_training:
        print('Load ckpt and train from step 0...')
        model = ModelClass.load_from_checkpoint(opt.checkpoint_path, opt=opt)
        trainer.fit(model, datamodule)
    else:
        model = ModelClass(opt)
        print(f'Continue training: {opt.checkpoint_path}')
        trainer.fit(model, datamodule, ckpt_path=opt.checkpoint_path)


if __name__ == "__main__":
    main()
