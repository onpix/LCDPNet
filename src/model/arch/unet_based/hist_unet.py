# -*- coding: utf-8 -*-
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.arch.drconv import DRConv2d, HistDRConv2d
from model.arch.hist import get_hist, get_hist_conv, pack_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DRDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, **kargs):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            DRConv2d(in_channels, mid_channels, kernel_size=3, region_num=REGION_NUM_, padding=1, **kargs),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DRConv2d(mid_channels, out_channels, kernel_size=3, region_num=REGION_NUM_, padding=1, **kargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        assert len(DRCONV_POSITION_) == 2
        assert DRCONV_POSITION_[0] or DRCONV_POSITION_[1]
        if DRCONV_POSITION_[0] == 0:
            print('[ WARN ] Use Conv in DRDoubleConv[0] instead of DRconv.')
            self.double_conv[0] = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        if DRCONV_POSITION_[1] == 0:
            print('[ WARN ] Use Conv in DRDoubleConv[3] instead of DRconv.')
            self.double_conv[3] = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.double_conv(x)
        self.guide_features = []
        if DRCONV_POSITION_[0]:
            self.guide_features.append(self.double_conv[0].guide_feature)
        if DRCONV_POSITION_[1]:
            self.guide_features.append(self.double_conv[3].guide_feature)
        return res


class HistDRDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = HistDRConv2d(in_channels, mid_channels, kernel_size=3, region_num=REGION_NUM_, padding=1)
        self.inter1 = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = HistDRConv2d(mid_channels, out_channels, kernel_size=3, region_num=REGION_NUM_, padding=1)
        self.inter2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, histmap):
        y = self.conv1(x, histmap)
        y = self.inter1(y)
        y = self.conv2(y, histmap)
        return self.inter2(y)


class HistGuidedDRDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, **kargs):
        super().__init__()
        assert len(DRCONV_POSITION_) == 2
        assert DRCONV_POSITION_[0] or DRCONV_POSITION_[1]

        if not mid_channels:
            mid_channels = out_channels
        if DRCONV_POSITION_[0]:
            self.conv1 = DRConv2d(in_channels, mid_channels, kernel_size=3, region_num=REGION_NUM_, padding=1, **kargs)
        else:
            print('[ WARN ] Use Conv in HistGuidedDRDoubleConv[0] instead of DRconv.')
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        self.inter1 = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        if DRCONV_POSITION_[1]:
            self.conv2 = DRConv2d(mid_channels, out_channels, kernel_size=3, region_num=REGION_NUM_, padding=1, **kargs)
        else:
            print('[ WARN ] Use Conv in HistGuidedDRDoubleConv[0] instead of DRconv.')
            self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

        self.inter2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, histmap):
        if DRCONV_POSITION_[0]:
            y = self.conv1(x, histmap)
        else:
            y = self.conv1(x)
        y = self.inter1(y)

        if DRCONV_POSITION_[1]:
            y = self.conv2(y, histmap)
        else:
            y = self.conv2(y)

        # self.guide_features = [self.conv1.guide_feature, self.conv2.guide_feature]
        self.guide_features = []
        if DRCONV_POSITION_[0]:
            self.guide_features.append(self.conv1.guide_feature)
        if DRCONV_POSITION_[1]:
            self.guide_features.append(self.conv2.guide_feature)

        return self.inter2(y)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, **kargs):
        super().__init__()
        self.up = nn.Upsample(scale_factor=DOWN_RATIO_, mode='bilinear', align_corners=True)
        if CONV_TYPE_ == 'drconv':
            if HIST_AS_GUIDE_:
                self.conv = HistDRDoubleConv(in_channels, out_channels, in_channels // 2)
            elif GUIDE_FEATURE_FROM_HIST_:
                self.conv = HistGuidedDRDoubleConv(in_channels, out_channels, in_channels // 2, **kargs)
            else:
                self.conv = DRDoubleConv(in_channels, out_channels, in_channels // 2)
        # elif CONV_TYPE_ == 'dconv':
        #     self.conv = HistDyDoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2, histmap):
        """
        histmap: shape [bs, c * n_bins, h, w]
        """
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        if HIST_AS_GUIDE_ or GUIDE_FEATURE_FROM_HIST_ or CONV_TYPE_ == 'dconv':
            x = torch.cat([x2, x1], dim=1)
            res = self.conv(x, histmap)
        else:
            x = torch.cat([x2, x1, histmap], dim=1)
            res = self.conv(x)
        self.guide_features = self.conv.guide_features
        return res


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_hist=False):
        super().__init__()
        self.use_hist = use_hist
        if not use_hist:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(DOWN_RATIO_),
                DoubleConv(in_channels, out_channels)
            )
        else:
            if HIST_AS_GUIDE_:
                # self.maxpool_conv = nn.Sequential(
                #     nn.MaxPool2d(2),
                #     HistDRDoubleConv(in_channels, out_channels, in_channels // 2)
                # )
                raise NotImplementedError()
            elif GUIDE_FEATURE_FROM_HIST_:
                self.maxpool = nn.MaxPool2d(DOWN_RATIO_)
                self.conv = HistGuidedDRDoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.maxpool_conv = nn.Sequential(
                    nn.MaxPool2d(DOWN_RATIO_),
                    DRDoubleConv(in_channels, out_channels, in_channels // 2)
                )

    def forward(self, x, histmap=None):
        if GUIDE_FEATURE_FROM_HIST_ and self.use_hist:
            x = self.maxpool(x)
            return self.conv(x, histmap)
        elif self.use_hist:
            return self.maxpool_conv(torch.cat([x, histmap], axis=1))
        else:
            return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class HistUNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 bilinear=True,
                 n_bins=8,
                 hist_as_guide=False,
                 channel_nums=None,
                 hist_conv_trainable=False,
                 encoder_use_hist=False,
                 guide_feature_from_hist=False,
                 region_num=8,
                 use_gray_hist=False,
                 conv_type='drconv',
                 down_ratio=1,
                 drconv_position=[1, 1],
                 ):
        super().__init__()
        C_NUMS = [16, 32, 64, 128, 256]
        if channel_nums:
            C_NUMS = channel_nums
        self.maxpool = nn.MaxPool2d(2)
        self.n_bins = n_bins
        self.encoder_use_hist = encoder_use_hist
        self.use_gray_hist = use_gray_hist
        self.hist_conv_trainable = hist_conv_trainable

        global HIST_AS_GUIDE_, GUIDE_FEATURE_FROM_HIST_, REGION_NUM_, CONV_TYPE_, DOWN_RATIO_, DRCONV_POSITION_
        HIST_AS_GUIDE_ = hist_as_guide
        GUIDE_FEATURE_FROM_HIST_ = guide_feature_from_hist
        REGION_NUM_ = region_num
        CONV_TYPE_ = conv_type
        DOWN_RATIO_ = down_ratio
        DRCONV_POSITION_ = drconv_position

        if hist_conv_trainable:
            self.hist_conv1 = get_hist_conv(n_bins * in_channels, down_ratio, train=True)
            self.hist_conv2 = get_hist_conv(n_bins * in_channels, down_ratio, train=True)
            self.hist_conv3 = get_hist_conv(n_bins * in_channels, down_ratio, train=True)
        else:
            self.hist_conv = get_hist_conv(n_bins, down_ratio)

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(in_channels, C_NUMS[0])
        if hist_as_guide or guide_feature_from_hist or conv_type == 'dconv':
            extra_c_num = 0
        elif use_gray_hist:
            extra_c_num = n_bins
        else:
            extra_c_num = n_bins * in_channels

        if guide_feature_from_hist:
            kargs = {
                'guide_input_channel': n_bins if use_gray_hist else n_bins * in_channels
            }
        else:
            kargs = {}

        if encoder_use_hist:
            encoder_extra_c_num = extra_c_num
        else:
            encoder_extra_c_num = 0

        self.down1 = Down(C_NUMS[0] + encoder_extra_c_num, C_NUMS[1], use_hist=encoder_use_hist)
        self.down2 = Down(C_NUMS[1] + encoder_extra_c_num, C_NUMS[2], use_hist=encoder_use_hist)
        self.down3 = Down(C_NUMS[2] + encoder_extra_c_num, C_NUMS[3], use_hist=encoder_use_hist)
        self.down4 = Down(C_NUMS[3] + encoder_extra_c_num, C_NUMS[4] // factor, use_hist=encoder_use_hist)

        self.up1 = Up(C_NUMS[4] + extra_c_num, C_NUMS[3] // factor, bilinear, **kargs)
        self.up2 = Up(C_NUMS[3] + extra_c_num, C_NUMS[2] // factor, bilinear, **kargs)
        self.up3 = Up(C_NUMS[2] + extra_c_num, C_NUMS[1] // factor, bilinear, **kargs)
        self.up4 = Up(C_NUMS[1] + extra_c_num, C_NUMS[0], bilinear, **kargs)
        self.outc = OutConv(C_NUMS[0], out_channels)

    def forward(self, x):
        # ipdb.set_trace()
        # get histograms
        # (`get_hist` return shape: n_bins, bs, c, h, w).
        if HIST_AS_GUIDE_ or self.use_gray_hist:
            histmap = get_hist(x, self.n_bins, grayscale=True)
        else:
            histmap = get_hist(x, self.n_bins)

        bs = x.shape[0]
        histmap = pack_tensor(histmap, self.n_bins).detach()  # out: [bs * c, n_bins, h, w]
        if not self.hist_conv_trainable:
            hist_down2 = self.hist_conv(histmap)
            hist_down4 = self.hist_conv(hist_down2)
            hist_down8 = self.hist_conv(hist_down4)

            # [bs * c, b_bins, h, w] -> [bs, c*b_bins, h, w]
            for item in [histmap, hist_down2, hist_down4, hist_down8]:
                item.data = item.reshape(bs, -1, *item.shape[-2:])
        else:
            histmap = histmap.reshape(bs, -1, *histmap.shape[-2:])
            hist_down2 = self.hist_conv1(histmap)
            hist_down4 = self.hist_conv2(hist_down2)
            hist_down8 = self.hist_conv3(hist_down4)  # [bs, n_bins * c, h/n, w/n]

        # forward
        encoder_hists = [None, ] * 4
        if self.encoder_use_hist:
            encoder_hists = [histmap, hist_down2, hist_down4, hist_down8]

        x1 = self.inc(x)
        x2 = self.down1(x1, encoder_hists[0])  # x2: 16
        x3 = self.down2(x2, encoder_hists[1])  # x3: 24
        x4 = self.down3(x3, encoder_hists[2])  # x4: 32
        x5 = self.down4(x4, encoder_hists[3])  # x5: 32

        # always apply hist in decoder:
        # ipdb.set_trace()
        x = self.up1(x5, x4, hist_down8)  # [x5, x4]: 32 + 32
        x = self.up2(x, x3, hist_down4)  # [x4, x3]:
        x = self.up3(x, x2, hist_down2)
        x = self.up4(x, x1, histmap)

        self.guide_features = [layer.guide_features for layer in [
            self.up1,
            self.up2,
            self.up3,
            self.up4,
        ]]

        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    model = HistUNet()
    x = torch.rand(4, 3, 512, 512)
    model(x)
    ipdb.set_trace()
