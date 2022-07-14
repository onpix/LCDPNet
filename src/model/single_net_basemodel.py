# -*- coding: utf-8 -*-

import os.path as osp

import cv2
import ipdb
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

import utils.util as util
from globalenv import *
from .basemodel import BaseModel


# dict_merge = lambda a, b: a.update(b) or a


class SingleNetBaseModel(BaseModel):
    # for models with only one self.net
    def __init__(self, opt, net, running_modes, valid_ssim=False, print_arch=True):
        super().__init__(opt, running_modes)
        self.net = net
        self.net.train()

        # config for SingleNetBaseModel
        if print_arch:
            print(str(net))
        self.valid_ssim = valid_ssim  # weather to compute ssim in validation
        self.tonemapper = cv2.createTonemapReinhard(2.2)

        self.psnr_func = torchmetrics.PeakSignalNoiseRatio(data_range=1)
        self.ssim_func = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1)

    def configure_optimizers(self):
        # self.parameters in LitModel is the same as nn.Module.
        # once you add nn.xxxx as a member in __init__, self.parameters will include it.
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        # optimizer = optim.Adam(self.net.parameters(), lr=self.opt[LR])

        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [schedular]

    def forward(self, x):
        return self.net(x)

    def training_step_forward(self, batch, batch_idx):
        if not self.MODEL_WATCHED and not self.opt[DEBUG] and self.opt.logger == 'wandb':
            self.logger.experiment.watch(
                self.net, log_freq=self.opt[LOG_EVERY] * 2, log_graph=True)
            self.MODEL_WATCHED = True
            # self.show_flops_and_param_num([batch[INPUT]])

        input_batch, gt_batch = batch[INPUT], batch[GT]
        output_batch = self(input_batch)
        self.iogt = {
            INPUT: input_batch,
            OUTPUT: output_batch,
            GT: gt_batch,
        }
        return input_batch, gt_batch, output_batch

    def validation_step(self, batch, batch_idx):
        input_batch, gt_batch = batch[INPUT], batch[GT]
        output_batch = self(input_batch)

        # log psnr
        output_ = util.cuda_tensor_to_ndarray(output_batch)
        y_ = util.cuda_tensor_to_ndarray(gt_batch)
        try:
            psnr = util.ImageProcessing.compute_psnr(output_, y_, 1.0)
        except:
            ipdb.set_trace()
        self.log(PSNR, psnr)

        # log SSIM (optional)
        if self.valid_ssim:
            ssim = util.ImageProcessing.compute_ssim(output_batch, gt_batch)
            self.log(SSIM, ssim)

        # log images
        if self.global_valid_step % self.opt.log_every == 0:
            self.log_images_dict(
                VALID,
                osp.basename(batch[INPUT_FPATH][0]),
                {
                    INPUT: input_batch,
                    OUTPUT: output_batch,
                    GT: gt_batch,
                },
                gt_fname=osp.basename(batch[GT_FPATH][0])
            )
        self.global_valid_step += 1
        return output_batch

    def log_training_iogt_img(self, batch, extra_img_dict=None):
        """
        Only used in training_step
        """
        if extra_img_dict:
            img_dict = {**self.iogt, **extra_img_dict}
        else:
            img_dict = self.iogt

        if self.global_step % self.opt.log_every == 0:
            self.log_images_dict(
                TRAIN,
                osp.basename(batch[INPUT_FPATH][0]),
                img_dict,
                gt_fname=osp.basename(batch[GT_FPATH][0])
            )

    @staticmethod
    def logdomain2hdr(ldr_batch):
        return 10 ** ldr_batch - 1

    def on_test_start(self):
        self.total_psnr = 0
        self.total_ssim = 0
        self.global_test_step = 0

    def on_test_end(self):
        print(
            f'Test step: {self.global_test_step}, Manual PSNR: {self.total_psnr / self.global_test_step}, Manual SSIM: {self.total_ssim / self.global_test_step}')

    def test_step(self, batch, batch_ix):
        """
        save test result and calculate PSNR and SSIM for `self.net` (when have GT)
        """
        # test without GT image:
        self.global_test_step += 1
        input_batch = batch[INPUT]
        assert input_batch.shape[0] == 1
        output_batch = self(input_batch)
        save_num = 1
        # visualized_batch = torch.cat([input_batch, output_batch])
        # save_num = 2

        # test with GT:
        # if GT in batch:
        #     gt_batch = batch[GT]
        #     if output_batch.shape != batch[GT].shape:
        #         print(
        #             f'[[ WARN ]] output.shape is {output_batch.shape} but GT.shape is {batch[GT].shape}. Resize to get PSNR.')
        #         gt_batch = F.interpolate(batch[GT], output_batch.shape[2:])
        #
        #     visualized_batch = torch.cat([visualized_batch, gt_batch])
        #     save_num = 3
        #
        #     # calculate metrics:
        #     # psnr = float(self.psnr_func(output_batch, gt_batch).cpu().numpy())
        #     # ssim = float(self.ssim_func(output_batch, gt_batch).cpu().numpy())
        #     # ipdb.set_trace()
        #
        #     # output_ = util.cuda_tensor_to_ndarray(output_batch)
        #     # y_ = util.cuda_tensor_to_ndarray(gt_batch)
        #     # psnr = util.ImageProcessing.compute_psnr(output_, y_, 1.0)
        #     # ssim = util.ImageProcessing.compute_ssim(output_, y_)
        #     # self.log_dict({
        #     #     'test-' + PSNR: psnr,
        #     #     'test-' + SSIM: ssim
        #     # }, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        #     # self.total_psnr += psnr
        #     # self.total_ssim += ssim
        #     # print(
        #     #     f'{batch[INPUT_FPATH][0].split("/")[-1]}: psnr: {psnr:.4f}, ssim: {ssim:.4f}, avgpsnr: {self.total_psnr / self.global_test_step:.4f}, avgssim: {self.total_ssim / self.global_test_step:.4f}')
        #
        #     # if batch[GT_FPATH][0].endswith('.hdr'):
        #     #     # output and GT -> HDR domain -> tonemap back to LDR
        #     #     output_vis = util.cuda_tensor_to_ndarray(
        #     #         self.logdomain2hdr(visualized_batch[1]).permute(1, 2, 0))
        #     #     gt_vis = util.cuda_tensor_to_ndarray(
        #     #         self.logdomain2hdr(visualized_batch[2]).permute(1, 2, 0))
        #     #     output_ldr = self.tonemapper.process(output_vis)
        #     #     gt_ldr = self.tonemapper.process(gt_vis)
        #     #     visualized_batch[1] = torch.tensor(
        #     #         output_ldr).permute(2, 0, 1).cuda()
        #     #     visualized_batch[2] = torch.tensor(
        #     #         gt_ldr).permute(2, 0, 1).cuda()
        #     #
        #     #     hdr_outpath = Path(self.opt[IMG_DIRPATH]) / 'hdr_output'
        #     #     util.mkdir(hdr_outpath)
        #     #     cv2.imwrite(
        #     #         str(hdr_outpath / (osp.basename(batch[INPUT_FPATH][0]) + '.hdr')), output_vis)

        # save images
        self.save_img_batch(
            output_batch,
            self.opt[IMG_DIRPATH],
            osp.basename(batch[INPUT_FPATH][0]),
            save_num=save_num
        )
