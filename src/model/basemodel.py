import os
import os.path as osp
import pathlib
from collections.abc import Iterable

import pytorch_lightning as pl
import torchvision
import wandb

import utils.util as util

try:
    from thop import profile
except Exception as e:
    print('ERR: import thop failed, skip. error msg:')
    print(e)

from globalenv import *

global LOGGER_BUFFER_LOCK


class BaseModel(pl.core.LightningModule):
    def __init__(self, opt, running_modes):
        '''
        logger_img_group_names: images group names in wandb logger. recommand: ['train', 'valid']
        '''

        super().__init__()
        self.save_hyperparameters(dict(opt))
        print('Running initialization for BaseModel')

        if IMG_DIRPATH in opt:
            # in training mode.
            # if in test mode, configLogging is not called.
            if TRAIN in running_modes:
                self.train_img_dirpath = osp.join(opt[IMG_DIRPATH], TRAIN)
                util.mkdir(self.train_img_dirpath)
            if VALID in running_modes and (len(opt[VALID_DATA].keys()) > 1 or opt[VALID_RATIO]):
                self.valid_img_dirpath = osp.join(opt[IMG_DIRPATH], VALID)
                util.mkdir(self.valid_img_dirpath)

        self.opt = opt
        self.learning_rate = self.opt[LR]

        self.MODEL_WATCHED = False  # for wandb watching model
        self.global_valid_step = 0
        self.iogt = {}  # a dict, saving input, output and gt batch

        assert isinstance(running_modes, Iterable)
        self.logger_image_buffer = {k: [] for k in running_modes}

    def show_flops_and_param_num(self, inputs):
        # inputs: arguments of `forward()`
        try:
            flops, params = profile(self, inputs=inputs)
            print('[ * ] FLOPs      =  ' + str(flops / 1000 ** 3) + 'G')
            print('[ * ] Params Num =  ' + str(params / 1000 ** 2) + 'M')
        except Exception as e:
            print(f'Err occured while calculating flops: {str(e)}')

    # def get_progress_bar_dict(self):
    #     items = super().get_progress_bar_dict()
    #     items.pop("v_num", None)
    #     # items.pop("loss", None)
    #     return items

    def build_test_res_dir(self):
        assert self.opt[CHECKPOINT_PATH]
        modelpath = pathlib.Path(self.opt[CHECKPOINT_PATH])

        # only `test_ds` is supported when testing.
        ds_type = TEST_DATA
        runtime_dirname = f'{self.opt.runtime.modelname}_{modelpath.parent.name}_{modelpath.name}@{self.opt.test_ds.name}'
        dirpath = modelpath.parent / TEST_RESULT_DIRNAME

        if (dirpath / runtime_dirname).exists():
            if len(os.listdir(dirpath / runtime_dirname)) == 0:
                # an existing but empty dir
                pass
            else:
                try:
                    input_str = input(
                        f'[ WARN ] Result directory "{runtime_dirname}" exists. Press ENTER to overwrite or input suffix '
                        f'to create a new one:\n> New name: {runtime_dirname}.')
                except Exception as e:
                    print(
                        f'[ WARN ] Excepion {e} occured, ignore input and set `input_str` empty.')
                    input_str = ''
                if input_str == '':
                    print(
                        f"[ WARN ] Overwrite result_dir: {runtime_dirname}")
                    pass
                else:
                    runtime_dirname += '.' + input_str
            # fname += '.new'

        dirpath /= runtime_dirname
        util.mkdir(dirpath)
        print('TEST - Result save path:')
        print(str(dirpath))

        util.save_opt(dirpath, self.opt)
        return str(dirpath)

    @staticmethod
    def save_img_batch(batch, dirpath, fname, save_num=1):
        util.mkdir(dirpath)
        imgpath = osp.join(dirpath, fname)

        # If you want to visiual a single image, call .unsqueeze(0)
        assert len(batch.shape) == 4
        torchvision.utils.save_image(batch[:save_num], imgpath)

    def calc_and_log_losses(self, loss_lambda_map):
        logged_losses = {}
        loss = 0
        for loss_name, loss_weight in self.opt[RUNTIME][LOSS].items():
            if loss_weight:
                current = loss_lambda_map[loss_name]()
                if current != None:
                    current *= loss_weight
                    logged_losses[loss_name] = current
                    loss += current

        logged_losses[LOSS] = loss
        self.log_dict(logged_losses)
        return loss

    def log_images_dict(self, mode, input_fname, img_batch_dict, gt_fname=None):
        """
        log input, output and gt images to local disk and remote wandb logger.
        mode: TRAIN or VALID
        """
        if self.opt[DEBUG]:
            return

        global LOGGER_BUFFER_LOCK
        if LOGGER_BUFFER_LOCK and self.opt.logger == 'wandb':
            # buffer is used by other GPU-thread.
            # print('Buffer locked!')
            return

        assert mode in [TRAIN, VALID]
        if mode == VALID:
            local_dirpath = self.valid_img_dirpath
            step = self.global_valid_step
            if self.global_valid_step == 0:
                print(
                    'WARN: Found global_valid_step=0. Maybe you foget to increase `self.global_valid_step` in `self.validation_step`?')
            # log_step = step  # to avoid valid log step = train log step
        elif mode == TRAIN:
            local_dirpath = self.train_img_dirpath
            step = self.global_step
            # log_step = None

        if step % self.opt[LOG_EVERY] == 0:
            suffiix = f'_epoch{self.current_epoch}_step{step}.png'
            input_fname = osp.basename(input_fname) + suffiix

            if gt_fname:
                gt_fname = osp.basename(gt_fname) + suffiix

            # ****** public buffer opration ******
            LOGGER_BUFFER_LOCK = True
            for name, batch in img_batch_dict.items():
                if batch is None or batch is False:
                    # image is None or False, skip.
                    continue

                # save local image:
                fname = input_fname
                if name == GT and gt_fname:
                    fname = gt_fname
                self.save_img_batch(
                    batch,
                    # e.g. ../train_log/train/output
                    osp.join(local_dirpath, name),
                    fname)

                # save remote image:
                if self.opt.logger == 'wandb':
                    self.add_img_to_buffer(mode, batch, mode, name, fname)
                else:
                    # tb logger
                    self.logger.experiment.add_image(f'{mode}/{name}', batch[0], step)

            if self.opt.logger == 'wandb':
                self.commit_logger_buffer(mode)

            # self.buffer_img_step += 1
            LOGGER_BUFFER_LOCK = False
            # ****** public buffer opration ******

    def add_img_to_buffer(self, group_name, batch, *caption):
        if len(batch.shape) == 3:
            # when input is not a batch:
            batch = batch.unsqueeze(0)

        self.logger_image_buffer[group_name].append(
            wandb.Image(batch[0], caption='-'.join(caption))
        )

    def commit_logger_buffer(self, groupname, **kwargs):
        assert self.logger
        self.logger.experiment.log({
            groupname: self.logger_image_buffer[groupname]
        }, **kwargs)

        # clear buffer after each commit for the next commit
        self.logger_image_buffer[groupname].clear()
