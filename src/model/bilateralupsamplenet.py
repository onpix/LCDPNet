import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from globalenv import *
from .arch.drconv import DRConv2d
from model.arch.unet_based.hist_unet import HistUNet
from .basic_loss import LTVloss
from .single_net_basemodel import SingleNetBaseModel


class LitModel(SingleNetBaseModel):
    def __init__(self, opt):
        super().__init__(opt, BilateralUpsampleNet(opt[RUNTIME]), [TRAIN, VALID])
        low_res = opt[RUNTIME][LOW_RESOLUTION]

        self.down_sampler = lambda x: F.interpolate(x, size=(low_res, low_res), mode='bicubic', align_corners=False)
        self.use_illu = opt[RUNTIME][PREDICT_ILLUMINATION]

        self.mse = torch.nn.MSELoss()
        self.ltv = LTVloss()
        self.cos = torch.nn.CosineSimilarity(1, 1e-8)

        self.net.train()

    def training_step(self, batch, batch_idx):
        input_batch, gt_batch, output_batch = super().training_step_forward(batch, batch_idx)
        loss_lambda_map = {
            MSE: lambda: self.mse(output_batch, gt_batch),
            COS_LOSS: lambda: (1 - self.cos(output_batch, gt_batch).mean()) * 0.5,
            LTV_LOSS: lambda: self.ltv(input_batch, self.net.illu_map, 1) if self.use_illu else None,
        }

        # logging:
        loss = self.calc_and_log_losses(loss_lambda_map)
        self.log_training_iogt_img(batch, extra_img_dict={
            PREDICT_ILLUMINATION: self.net.illu_map,
            GUIDEMAP: self.net.guidemap
        })
        return loss

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_ix):
        super().test_step(batch, batch_ix)

    def forward(self, x):
        low_res_x = self.down_sampler(x)
        return self.net(low_res_x, x)


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        conv_type = OPT['conv_type']
        if conv_type == 'conv':
            self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        elif conv_type.startswith('drconv'):
            region_num = int(conv_type.replace('drconv', ''))
            self.conv = DRConv2d(int(inc), int(outc), kernel_size, region_num=region_num, padding=padding,
                                 stride=stride)
            print(f'[ WARN ] Using DRconv2d(n_region={region_num}) instead of Conv2d in BilateralUpsampleNet.')
        else:
            raise NotImplementedError()

        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class SliceNode(nn.Module):
    def __init__(self, opt):
        super(SliceNode, self).__init__()
        self.opt = opt

    def forward(self, bilateral_grid, guidemap):
        # bilateral_grid shape: Nx12x8x16x16
        device = bilateral_grid.get_device()
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)

        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        guidemap = guidemap * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)

        # guidemap shape: [N, 1 (D), H, W]
        # bilateral_grid shape: [N, 12 (c), 8 (d), 16 (h), 16 (w)], which is considered as a 3D space: [8, 16, 16]
        # guidemap_guide shape: [N, 1 (D), H, W, 3], which is considered as a 3D space: [1, H, W]
        # coeff shape: [N, 12 (c), 1 (D), H, W]

        # in F.grid_sample, gird is guidemap_guide, input is bilateral_grid
        # guidemap_guide[N, D, H, W] is a 3-vector <x, y, z>. but:
        #       x -> W, y -> H, z -> D  in bilater_grid
        # What does it really do:
        #   [ 1 ] For pixel in guidemap_guide[D, H, W], get <x,y,z>, and:
        #   [ 2 ] Normalize <x, y, z> from [-1, 1] to [0, w - 1], [0, h - 1], [0, d - 1], respectively.
        #   [ 3 ] Locate pixel in bilateral_grid at position [N, :, z, y, x].
        #   [ 4 ] Interplate using the neighbor values as the output affine matrix.

        # Force them have the same type for fp16 training :
        guidemap_guide = guidemap_guide.type_as(bilateral_grid)
        # bilateral_grid = bilateral_grid.type_as(guidemap_guide)
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, 'bilinear', align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):
        '''
        coeff shape: [bs, 12, h, w]
        input shape: [bs, 3, h, w]
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)


class ApplyCoeffsGamma(nn.Module):
    def __init__(self):
        super(ApplyCoeffsGamma, self).__init__()
        print('[ WARN ] Use alter methods indtead of affine matrix.')

    def forward(self, x_r, x):
        '''
        coeff shape: [bs, 12, h, w]
        apply zeroDCE curve.
        '''

        # [ 008 ] single iteration alpha map:
        # coeff channel num: 3
        # return x + x_r * (torch.pow(x, 2) - x)

        # [ 009 ] 8 iteratoins:
        # coeff channel num: 24
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image

        # [ 014 ] use illu map:
        # coeff channel num: 3
        # return x / (torch.where(x_r < x, x, x_r) + 1e-7)

        # [ 015 ] use HSV and only affine V channel:
        # coeff channel num: 3
        # V = torch.sum(x * x_r, dim=1, keepdim=True) + x_r
        # return torch.cat([x[:, 0:2, ...], V], dim=1)


class ApplyCoeffsRetinex(nn.Module):
    def __init__(self):
        super().__init__()
        print('[ WARN ] Use alter methods indtead of affine matrix.')

    def forward(self, x_r, x):
        '''
        coeff shape: [bs, 12, h, w]
        apply division of illumap.
        '''

        # [ 014 ] use illu map:
        # coeff channel num: 3
        return x / (torch.where(x_r < x, x, x_r) + 1e-7)


class GuideNet(nn.Module):
    def __init__(self, params=None, out_channel=1):
        super(GuideNet, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=True)
        self.conv2 = ConvBlock(16, out_channel, kernel_size=1, padding=0, activation=nn.Sigmoid)  # nn.Tanh

    def forward(self, x):
        return self.conv2(self.conv1(x))  # .squeeze(1)


class LowResNet(nn.Module):
    def __init__(self, coeff_dim=12, opt=None):
        super(LowResNet, self).__init__()
        self.params = opt
        self.coeff_dim = coeff_dim

        lb = opt[LUMA_BINS]
        cm = opt[CHANNEL_MULTIPLIER]
        sb = opt[SPATIAL_BIN]
        bn = opt[BATCH_NORM]
        nsize = opt[LOW_RESOLUTION]

        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize / sb))
        self.splat_features = nn.ModuleList()
        prev_ch = 3
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(ConvBlock(prev_ch, cm * (2 ** i) * lb, 3, stride=2, batch_norm=use_bn))
            prev_ch = splat_ch = cm * (2 ** i) * lb

        # global features
        n_layers_global = int(np.log2(sb / 4))
        # print(n_layers_global)
        self.global_features_conv = nn.ModuleList()
        self.global_features_fc = nn.ModuleList()
        for i in range(n_layers_global):
            self.global_features_conv.append(ConvBlock(prev_ch, cm * 8 * lb, 3, stride=2, batch_norm=bn))
            prev_ch = cm * 8 * lb

        n_total = n_layers_splat + n_layers_global
        prev_ch = prev_ch * (nsize / 2 ** n_total) ** 2
        self.global_features_fc.append(FC(prev_ch, 32 * cm * lb, batch_norm=bn))
        self.global_features_fc.append(FC(32 * cm * lb, 16 * cm * lb, batch_norm=bn))
        self.global_features_fc.append(FC(16 * cm * lb, 8 * cm * lb, activation=None, batch_norm=bn))

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(splat_ch, 8 * cm * lb, 3, batch_norm=bn))
        self.local_features.append(ConvBlock(8 * cm * lb, 8 * cm * lb, 3, activation=None, use_bias=False))

        # predicton
        self.conv_out = ConvBlock(8 * cm * lb, lb * coeff_dim, 1, padding=0, activation=None)

    def forward(self, lowres_input):
        params = self.params
        bs = lowres_input.shape[0]
        lb = params[LUMA_BINS]
        cm = params[CHANNEL_MULTIPLIER]
        sb = params[SPATIAL_BIN]

        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)
        splat_features = x

        for layer in self.global_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x

        x = splat_features
        for layer in self.local_features:
            x = layer(x)
        local_features = x

        # shape: bs x 64 x 16 x 16
        fusion_grid = local_features

        # shape: bs x 64 x 1 x 1
        fusion_global = global_features.view(bs, 8 * cm * lb, 1, 1)
        fusion = self.relu(fusion_grid + fusion_global)

        x = self.conv_out(fusion)

        # reshape channel dimension -> bilateral grid dimensions:
        # [bs, 96, 16, 16] -> [bs, 12, 8, 16, 16]
        y = torch.stack(torch.split(x, self.coeff_dim, 1), 2)
        return y


class LowResHistUNet(HistUNet):
    def __init__(self, coeff_dim=12, opt=None):
        super(LowResHistUNet, self).__init__(
            in_channels=3,
            out_channels=coeff_dim * opt[LUMA_BINS],
            bilinear=True,
            **opt[HIST_UNET]
        )
        self.coeff_dim = coeff_dim
        print('[[ WARN ]] Using HistUNet in BilateralUpsampleNet as backbone')

    def forward(self, x):
        y = super(LowResHistUNet, self).forward(x)
        y = torch.stack(torch.split(y, self.coeff_dim, 1), 2)
        return y


class BilateralUpsampleNet(nn.Module):
    def __init__(self, opt):
        super(BilateralUpsampleNet, self).__init__()
        self.opt = opt
        global OPT
        OPT = opt
        self.guide = GuideNet(params=opt)
        self.slice = SliceNode(opt)
        self.build_coeffs_network(opt)

    def build_coeffs_network(self, opt):
        # Choose backbone:
        if opt[BACKBONE] == 'ori':
            Backbone = LowResNet
        elif opt[BACKBONE] == 'hist-unet':
            Backbone = LowResHistUNet
        else:
            raise NotImplementedError()

        # How to apply coeffs:
        # ───────────────────────────────────────────────────────────────────
        if opt[COEFFS_TYPE] == MATRIX:
            self.coeffs = Backbone(opt=opt)
            self.apply_coeffs = ApplyCoeffs()

        elif opt[COEFFS_TYPE] == GAMMA:
            print('[[ WARN ]] HDRPointwiseNN use COEFFS_TYPE: GAMMA.')

            # [ 008 ] change affine matrix -> other methods (alpha map, illu map)
            self.coeffs = Backbone(opt=opt, coeff_dim=24)
            self.apply_coeffs = ApplyCoeffsGamma()

        elif opt[COEFFS_TYPE] == 'retinex':
            print('[[ WARN ]] HDRPointwiseNN use COEFFS_TYPE: retinex.')
            self.coeffs = Backbone(opt=opt, coeff_dim=3)
            self.apply_coeffs = ApplyCoeffsRetinex()

        else:
            raise NotImplementedError(f'[ ERR ] coeff type {opt[COEFFS_TYPE]} unkown.')
        # ─────────────────────────────────────────────────────────────────────────────

    def forward(self, lowres, fullres):
        bilateral_grid = self.coeffs(lowres)
        try:
            self.guide_features = self.coeffs.guide_features
        except:
            ...
        guide = self.guide(fullres)
        self.guidemap = guide

        slice_coeffs = self.slice(bilateral_grid, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)

        # use illu map:
        self.slice_coeffs = slice_coeffs
        # if self.opt[PREDICT_ILLUMINATION]:
        #
        #     power = self.opt[ILLU_MAP_POWER]
        #     if power:
        #         assert type(power + 0.1) == float
        #         out = out.pow(power)
        #
        #     out = out.clamp(fullres, torch.ones_like(out))
        #     # out = torch.where(out < fullres, fullres, out)
        #     self.illu_map = out
        #     out = fullres / (out + 1e-7)
        # else:
        self.illu_map = None

        if self.opt[PREDICT_ILLUMINATION]:
            return fullres / (out.clamp(fullres, torch.ones_like(out)) + 1e-7)
        else:
            return out
