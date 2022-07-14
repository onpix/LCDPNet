import os.path as osp

import kornia as kn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import utils.util as util

from globalenv import *
from .arch.nonlocal_block_embedded_gaussian import NONLocalBlock2D
from .basic_loss import L_TV, WeightedL1Loss, HistogramLoss, IntermediateHistogramLoss, LTVloss
# from .hdrunet import tanh_L1Loss
from .single_net_basemodel import SingleNetBaseModel


class tanh_L1Loss(nn.Module):
    def __init__(self):
        super(tanh_L1Loss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
        return loss


class LitModel(SingleNetBaseModel):
    def __init__(self, opt):
        super().__init__(opt, DeepWBNet(opt[RUNTIME]), [TRAIN, VALID])
        # self.pixel_loss = torch.nn.MSELoss()

        # [ 008-u1 ] use tanh L1 loss
        self.pixel_loss = tanh_L1Loss()

        # [ 019 ] use log L1 loss
        # self.pixel_loss = LogL2Loss()

        # [ 028 ] use weighted L1 loss.
        self.weighted_loss = WeightedL1Loss()
        self.tvloss = L_TV()
        self.ltv2 = LTVloss()
        self.cos = torch.nn.CosineSimilarity(1, 1e-8)
        self.histloss = HistogramLoss()
        # self.vggloss = VGGLoss(shift=2)
        # self.vggloss.train()
        self.inter_histloss = IntermediateHistogramLoss()

    def training_step(self, batch, batch_idx):
        input_batch, gt_batch, output_batch = super().training_step_forward(batch, batch_idx)
        # print('[*] Now running:', batch[INPUT].shape, batch[GT].shape, output_batch.shape, batch[INPUT_FPATH], batch[GT_FPATH])

        loss_lambda_map = {
            L1_LOSS: lambda: self.pixel_loss(output_batch, gt_batch),
            COS_LOSS: lambda: (1 - self.cos(output_batch, gt_batch).mean()) * 0.5,
            COS_LOSS + '2': lambda: 1 - F.sigmoid(self.cos(output_batch, gt_batch).mean()),
            LTV_LOSS: lambda: self.tvloss(output_batch),
            'tvloss1': lambda: self.tvloss(self.net.res[ILLU_MAP]),
            'tvloss2': lambda: self.tvloss(self.net.res[INVERSE_ILLU_MAP]),

            'tvloss1_new': lambda: self.ltv2(input_batch, self.net.res[ILLU_MAP], 1),
            'tvloss2_new': lambda: self.ltv2(1 - input_batch, self.net.res[INVERSE_ILLU_MAP], 1),
            'illumap_loss': lambda: F.mse_loss(self.net.res[ILLU_MAP], 1 - self.net.res[INVERSE_ILLU_MAP]),
            WEIGHTED_LOSS: lambda: self.weighted_loss(input_batch.detach(), output_batch, gt_batch),
            SSIM_LOSS: lambda: kn.losses.ssim_loss(output_batch, gt_batch, window_size=5),
            PSNR_LOSS: lambda: kn.losses.psnr_loss(output_batch, gt_batch, max_val=1.0),
            HIST_LOSS: lambda: self.histloss(output_batch, gt_batch),
            INTER_HIST_LOSS: lambda: self.inter_histloss(
                input_batch, gt_batch, self.net.res[BRIGHTEN_INPUT], self.net.res[DARKEN_INPUT]),
            VGG_LOSS: lambda: self.vggloss(input_batch, gt_batch),
        }
        loss = self.calc_and_log_losses(loss_lambda_map)

        # logging images:
        self.log_training_iogt_img(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_ix):
        super().test_step(batch, batch_ix)

        # save intermidiate results
        for k, v in self.net.res.items():
            dirpath = Path(self.opt[IMG_DIRPATH]) / k
            fname = osp.basename(batch[INPUT_FPATH][0])
            if 'illu' in k:
                util.mkdir(dirpath)
                torchvision.utils.save_image(v[0].unsqueeze(1), dirpath / fname)
            elif k == 'guide_features':
                # v.shape: [bs, region_num, h, w]
                util.mkdir(dirpath)
                max_size = v[-1][-1].shape[-2:]
                final = []
                for level_guide in v:
                    gs = [F.interpolate(g, max_size) for g in level_guide]
                    final.extend(gs)
                # import ipdb
                # ipdb.set_trace()
                region_num = final[0].shape[1]
                final = torch.stack(final).argmax(axis=2).float() / region_num
                # ipdb.set_trace()
                torchvision.utils.save_image(final, dirpath / fname)
            else:
                self.save_img_batch(v, dirpath, fname)


class DeepWBNet(nn.Module):
    def build_illu_net(self):
        # if self.opt[BACKBONE] == 'unet':
        #     if self.opt[USE_ATTN_MAP]:
        #         return UNet(
        #             self.opt,
        #             in_channels=4,
        #             out_channels=1,
        #             wavelet=self.opt[USE_WAVELET],
        #             non_local=self.opt[NON_LOCAL]
        #         )
        #     else:
        #         return UNet(self.opt, out_channels=self.opt[ILLUMAP_CHANNEL], wavelet=self.opt[USE_WAVELET])

        from .bilateralupsamplenet import BilateralUpsampleNet
        return BilateralUpsampleNet(self.opt[BUNET])
        #
        # elif self.opt[BACKBONE] == 'ynet':
        #     from .arch.ynet import YNet
        #     return YNet()
        #
        # elif self.opt[BACKBONE] == 'hdrunet':
        #     from .hdrunet import HDRUNet
        #     return HDRUNet()
        #
        # elif self.opt[BACKBONE] == 'hist-unet':
        #     from model.arch.unet_based.hist_unet import HistUNet
        #     return HistUNet(**self.opt[HIST_UNET])
        #
        # else:
        #     raise NotImplementedError(f'[[ ERR ]] Unknown backbone arch: {self.opt[BACKBONE]}')

    def backbone_forward(self, net, x):
        if self.opt[BACKBONE] in ['unet', 'hdrunet', 'hist-unet']:
            return net(x)

        elif self.opt[BACKBONE] == 'ynet':
            return net.forward_2input(x, 1 - x)

        elif self.opt[BACKBONE] == BUNET:
            low_x = self.down_sampler(x)
            res = net(low_x, x)
            try:
                self.res.update({'guide_features': net.guide_features})
            except:
                ...
                # print('[yellow]No guide feature found in BilateralUpsampleNet[/yellow]')
            return res

    def __init__(self, opt=None):
        super(DeepWBNet, self).__init__()
        self.opt = opt
        self.down_sampler = lambda x: F.interpolate(x, size=(256, 256), mode='bicubic', align_corners=False)
        self.illu_net = self.build_illu_net()

        # [ 021 ] use 2 illu nets (do not share weights).
        if not opt[SHARE_WEIGHTS]:
            self.illu_net2 = self.build_illu_net()

        # self.guide_net = GuideNN(out_channel=3)
        if opt[HOW_TO_FUSE] in ['cnn-weights', 'cnn-direct', 'cnn-softmax3']:
            # self.out_net = UNet(in_channels=9, wavelet=opt[USE_WAVELET])

            # [ 008-u1 ] use a simple network
            nf = 32
            self.out_net = nn.Sequential(
                nn.Conv2d(9, nf, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, 3, 1, 1),
                nn.ReLU(inplace=True),
                NONLocalBlock2D(nf, sub_sample='bilinear', bn_layer=False),
                nn.Conv2d(nf, nf, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, 3, 1),
                NONLocalBlock2D(3, sub_sample='bilinear', bn_layer=False),
            )

        elif opt[HOW_TO_FUSE] in ['cnn-color']:
            # self.out_net = UNet(in_channels=3, wavelet=opt[USE_WAVELET])
            ...

        if not self.opt[BACKBONE_OUT_ILLU]:
            print('[[ WARN ]] Use output of backbone as brighten & darken directly.')
        self.res = {}

    def decomp(self, x1, illu_map):
        return x1 / (torch.where(illu_map < x1, x1, illu_map.float()) + 1e-7)

    def one_iter(self, x, attn_map, inverse_attn_map):
        # used only when USE_ATTN_MAP
        x1 = torch.cat((x, attn_map), 1)
        inverse_x1 = torch.cat((1 - x, inverse_attn_map), 1)

        illu_map = self.illu_net(x1, attn_map)
        inverse_illu_map = self.illu_net(inverse_x1)
        return illu_map, inverse_illu_map

    def forward(self, x):
        # ──────────────────────────────────────────────────────────
        # [ <008 ] use guideNN
        # x1 = self.guide_net(x).clamp(0, 1)

        # [ 008 ] use original input
        x1 = x
        inverse_x1 = 1 - x1

        if self.opt[USE_ATTN_MAP]:
            # [ 015 ] use attn map iteration to get illu map
            r, g, b = x[:, 0] + 1, x[:, 1] + 1, x[:, 2] + 1

            # init attn map as illumination channel of original input img:
            attn_map = (1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.).unsqueeze(1)
            inverse_attn_map = 1 - attn_map
            for _ in range(3):
                inverse_attn_map, attn_map = self.one_iter(x, attn_map, inverse_attn_map)
            illu_map, inverse_illu_map = inverse_attn_map, attn_map

        elif self.opt[BACKBONE] == 'ynet':
            # [ 024 ] one encoder, 2 decoders.
            illu_map, inverse_illu_map = self.backbone_forward(self.illu_net, x1)

        else:
            illu_map = self.backbone_forward(self.illu_net, x1)
            if self.opt[SHARE_WEIGHTS]:
                inverse_illu_map = self.backbone_forward(self.illu_net, inverse_x1)
            else:
                # [ 021 ] use 2 illu nets
                inverse_illu_map = self.backbone_forward(self.illu_net2, inverse_x1)
        # ──────────────────────────────────────────────────────────

        if self.opt[BACKBONE_OUT_ILLU]:
            brighten_x1 = self.decomp(x1, illu_map)
            inverse_x2 = self.decomp(inverse_x1, inverse_illu_map)
        else:
            brighten_x1 = illu_map
            inverse_x2 = inverse_illu_map
        darken_x1 = 1 - inverse_x2
        # ──────────────────────────────────────────────────────────

        self.res.update({
            ILLU_MAP: illu_map,
            INVERSE_ILLU_MAP: inverse_illu_map,
            BRIGHTEN_INPUT: brighten_x1,
            DARKEN_INPUT: darken_x1,
        })

        # fusion:
        # ──────────────────────────────────────────────────────────
        if self.opt[HOW_TO_FUSE] == 'cnn-weights':
            # [ 009 ] only fuse 2 output image
            # fused_x = torch.cat([brighten_x1, darken_x1], dim=1)

            fused_x = torch.cat([x, brighten_x1, darken_x1], dim=1)

            # [ 007 ] get weight-map from UNet, then get output from weight-map
            weight_map = self.out_net(fused_x)  # <- 3 channels, [ N, 3, H, W ]
            w1 = weight_map[:, 0, ...].unsqueeze(1)
            w2 = weight_map[:, 1, ...].unsqueeze(1)
            w3 = weight_map[:, 2, ...].unsqueeze(1)
            out = x * w1 + brighten_x1 * w2 + darken_x1 * w3

            # [ 009 ] only fuse 2 output image
            # out = brighten_x1 * w1 + darken_x1 * w2
            # ────────────────────────────────────────────────────────────

        elif self.opt[HOW_TO_FUSE] == 'cnn-softmax3':
            fused_x = torch.cat([x, brighten_x1, darken_x1], dim=1)
            weight_map = F.softmax(self.out_net(fused_x), dim=1)  # <- 3 channels, [ N, 3, H, W ]
            w1 = weight_map[:, 0, ...].unsqueeze(1)
            w2 = weight_map[:, 1, ...].unsqueeze(1)
            w3 = weight_map[:, 2, ...].unsqueeze(1)
            out = x * w1 + brighten_x1 * w2 + darken_x1 * w3

        # [ 006 ] get output directly from UNet
        elif self.opt[HOW_TO_FUSE] == 'cnn-direct':
            fused_x = torch.cat([x, brighten_x1, darken_x1], dim=1)
            out = self.out_net(fused_x)

        # [ 016 ] average 2 outputs.
        elif self.opt[HOW_TO_FUSE] == 'avg':
            out = 0.5 * brighten_x1 + 0.5 * darken_x1

        # [ 017 ] global color ajust
        elif self.opt[HOW_TO_FUSE] == 'cnn-color':
            out = 0.5 * brighten_x1 + 0.5 * darken_x1

        # elif self.opt[HOW_TO_FUSE] == 'cnn-residual':
        #     out = x +

        else:
            raise NotImplementedError(f'Unknown fusion method: {self.opt[HOW_TO_FUSE]}')
        return out
