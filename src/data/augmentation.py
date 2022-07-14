# import cv2
from collections.abc import Iterable

import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms

from globalenv import *


class Downsample:
    def __init__(self, downsample_factor=None):
        self.downsample_factor = downsample_factor

        if isinstance(self.downsample_factor, Iterable):
            # should be [h, w]
            assert len(downsample_factor) == 2

    def __call__(self, img):
        '''
        img: passed by the previous transforms. PIL iamge or np.ndarray
        '''
        origin_h = img.size[1]
        origin_w = img.size[0]
        if isinstance(self.downsample_factor, Iterable):
            # pass [h,w]
            if -1 in self.downsample_factor:
                # automatic calculate the output size:
                h_scale = origin_h / self.downsample_factor[0]
                w_scale = origin_w / self.downsample_factor[1]

                # choose the correct one
                scale = max(w_scale, h_scale)
                new_size = [
                    int(origin_h / scale),  # H
                    int(origin_w / scale)  # W
                ]
            else:
                new_size = self.downsample_factor  # [H, W]

        elif type(self.downsample_factor + 0.1) == float:
            # pass a number as scale factor
            # PIL.Image, cv2.resize and torchvision.transforms.Resize all accepts [W, H]
            new_size = [
                int(img.size[1] / self.downsample_factor),  # H
                int(img.size[0] / self.downsample_factor)  # W
            ]
        else:
            raise RuntimeError(f'ERR: Wrong config aug.downsample: {self.downsample_factor}')

        img = img.resize(new_size[::-1])  # reverse passed [h, w] to [w, h]
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'({self.downsample_factor})'


def get_value(d, k):
    if k in d and d[k]:
        return d[k]
    else:
        return False


def parseAugmentation(opt):
    '''
    return: pytorch composed transform
    '''
    aug_config = opt[AUGMENTATION]
    aug_list = [transforms.ToPILImage(), ]

    # the order is fixed:
    augmentaionFactory = {
        DOWNSAMPLE: Downsample(aug_config[DOWNSAMPLE])
        if get_value(aug_config, DOWNSAMPLE) else None,
        CROP: transforms.RandomCrop(aug_config[CROP])
        if get_value(aug_config, CROP) else None,
        HORIZON_FLIP: transforms.RandomHorizontalFlip(),
        VERTICAL_FLIP: transforms.RandomVerticalFlip(),
    }

    for k, v in augmentaionFactory.items():
        if get_value(aug_config, k):
            aug_list.append(v)

    aug_list.append(transforms.ToTensor())
    print('Dataset augmentation:')
    print(aug_list)
    return transforms.Compose(aug_list)


if __name__ == '__main__':
    pass
