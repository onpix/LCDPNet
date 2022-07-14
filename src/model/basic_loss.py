import kornia as kn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from .arch.hist import get_hist, get_hist_conv


class HistogramLoss(nn.Module):
    def __init__(self, n_bins=8, downscale=16):
        super().__init__()
        self.n_bins = n_bins
        self.hist_conv = get_hist_conv(n_bins, downscale)

        # pack tensor: transform gt_hist.shape [n_bins, bs, c, h, w] -> [bs*c, b_bins, h, w]
        # merge dim 1 (bs) and dim 2 (channel).
        self.pack_tensor = lambda x: x.reshape(self.n_bins, -1, *x.shape[-2:]).permute(1, 0, 2, 3)

    def forward(self, output, gt):
        gt_hist = get_hist(gt, self.n_bins)
        output_hist = get_hist(output, self.n_bins)

        shrink_hist_gt = self.hist_conv(self.pack_tensor(gt_hist))
        shrink_hist_output = self.hist_conv(self.pack_tensor(output_hist))

        return F.mse_loss(shrink_hist_gt, shrink_hist_output)


class IntermediateHistogramLoss(HistogramLoss):
    def __init__(self, n_bins=8, downscale=16):
        super().__init__(n_bins, downscale)
        self.exposure_threshold = 0.5

    def forward(self, img, gt, brighten, darken):
        """
        input brighten and darken img, get errors between:
        - brighten img & darken region in GT
        - darken img & brighten region in GT
        """
        bs, c, _, _ = gt.shape
        gt_hist = get_hist(gt, self.n_bins)
        shrink_hist_gt = self.hist_conv(self.pack_tensor(gt_hist))

        down_size = shrink_hist_gt.shape[-2:]
        shrink_hist_gt = shrink_hist_gt.reshape(bs, c, self.n_bins, *down_size)
        down_x = F.interpolate(img, size=down_size)

        # get mask from the input:
        over_ixs = down_x > self.exposure_threshold
        under_ixs = down_x <= self.exposure_threshold
        over_mask = down_x.clone()
        over_mask[under_ixs] = 0
        over_mask[over_ixs] = 1
        over_mask.unsqueeze_(2)
        under_mask = down_x.clone()
        under_mask[under_ixs] = 1
        under_mask[over_ixs] = 0
        under_mask.unsqueeze_(2)

        shrink_darken_hist = self.hist_conv(self.pack_tensor(get_hist(darken, self.n_bins))).reshape(bs, c, self.n_bins,
                                                                                                     *down_size)
        shrink_brighten_hist = self.hist_conv(self.pack_tensor(get_hist(brighten, self.n_bins))).reshape(bs, c,
                                                                                                         self.n_bins,
                                                                                                         *down_size)

        # [ 046 ] use ssim loss
        return 0.5 * kn.losses.ssim_loss((shrink_hist_gt * over_mask).view(-1, c, *down_size),
                                         (shrink_darken_hist * over_mask).view(-1, c, *down_size),
                                         window_size=5) + 0.5 * kn.losses.ssim_loss(
            (shrink_hist_gt * under_mask).view(-1, c, *down_size),
            (shrink_brighten_hist * under_mask).view(-1, c, *down_size), window_size=5)

        # [ 042 ] use l2 loss
        # return 0.5 * F.mse_loss(shrink_hist_gt * over_mask, shrink_darken_hist * over_mask) + 0.5 * F.mse_loss(shrink_hist_gt * under_mask, shrink_brighten_hist * under_mask)


class WeightedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, output, gt):
        bias = 0.1
        weights = (torch.abs(input - 0.5) + bias) / 0.5
        weights = weights.mean(axis=1).unsqueeze(1).repeat(1, 3, 1, 1)
        loss = torch.mean(torch.abs(output - gt) * weights.detach())
        return loss


class LTVloss(nn.Module):
    def __init__(self, alpha=1.2, beta=1.5, eps=1e-4):
        super(LTVloss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, origin, illumination, weight):
        '''
        origin:       one batch of input data. shape [batchsize, 3, h, w]
        illumination: one batch of predicted illumination data. if predicted_illumination 
                      is False, then use the output (predicted result) of the network.
        '''

        # # re-normalize origin to 0 ~ 1
        # origin = (input_ - input_.min().item()) / (input_.max().item() - input_.min().item())

        I = origin[:, 0:1, :, :] * 0.299 + origin[:, 1:2, :, :] * \
            0.587 + origin[:, 2:3, :, :] * 0.114
        L = torch.log(I + self.eps)
        dx = L[:, :, :-1, :-1] - L[:, :, :-1, 1:]
        dy = L[:, :, :-1, :-1] - L[:, :, 1:, :-1]

        dx = self.beta / (torch.pow(torch.abs(dx), self.alpha) + self.eps)
        dy = self.beta / (torch.pow(torch.abs(dy), self.alpha) + self.eps)

        x_loss = dx * \
                 ((illumination[:, :, :-1, :-1] - illumination[:, :, :-1, 1:]) ** 2)
        y_loss = dy * \
                 ((illumination[:, :, :-1, :-1] - illumination[:, :, 1:, :-1]) ** 2)
        tvloss = torch.mean(x_loss + y_loss) / 2.0

        return tvloss * weight


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
