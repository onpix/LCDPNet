# -*- coding: utf-8 -*-
import random
from collections.abc import Iterable
from glob import glob

import cv2
import ipdb
import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningDataModule

from globalenv import *
from .augmentation import parseAugmentation


def parse_item_list_txt(txtpath):
    '''
    Parse txt file containing all file paths. Each line one file.
    '''
    txt = Path(txtpath)
    basedir = txt.parent
    content = txt.open().read().splitlines()

    sample = content[0]
    if sample.split('/')[0] in str(basedir):
        raise NotImplementedError('Not implemented: file path in txt and basedir have comman str.')
    else:
        assert (basedir / sample).exists()
        return [str(basedir / x) for x in content]


def load_from_glob_list(globs):
    if type(globs) == str:
        if globs.endswith('.txt'):
            print(f'Parse txt file: {globs}')
            return parse_item_list_txt(globs)
        else:
            return sorted(glob(globs))

    elif isinstance(globs, Iterable):
        # if iamges are glob lists, sort EACH list ** individually **
        res = []
        for g in globs:
            assert not g.endswith('.txt'), 'TXT file should not in glob list.'
            res.extend(sorted(glob(g)))
        return res
    else:
        ipdb.set_trace()
        raise TypeError(
            f'ERR: `ds.GT` or `ds.input` has wrong type: expect `str` or `list` but get {type(globs)}')


def augment_one_img(img, seed, transform=None):
    img = img.astype(np.uint8)
    random.seed(seed)
    torch.manual_seed(seed)
    if transform:
        img = transform(img)
    return img


class ImagesDataset(torch.utils.data.Dataset):

    def __init__(self, opt, ds_type=TRAIN_DATA, transform=None, batchsize=None):
        """Initialisation for the Dataset object
        transform: PyTorch image transformations to apply to the images
        """
        self.transform = transform
        self.opt = opt

        gt_globs = opt[ds_type][GT]
        input_globs = opt[ds_type][INPUT]
        self.have_gt = True if gt_globs else False

        print(f'{ds_type} - GT Directory path: [yellow]{gt_globs}[/yellow]')
        print(f'{ds_type} - Input Directory path: [yellow]{input_globs}[/yellow]')

        # load input images:
        self.input_list = load_from_glob_list(input_globs)

        # load GT images:
        if self.have_gt:
            self.gt_list = load_from_glob_list(gt_globs)
            try:
                assert len(self.input_list) == len(self.gt_list)
            except:
                ipdb.set_trace()
                raise AssertionError(
                    f'In [{ds_type}]: len(input_images) ({len(self.input_list)}) != len(gt_images) ({len(self.gt_list)})! ')

        # import ipdb; ipdb.set_trace()
        print(
            f'{ds_type} Dataset length: {self.__len__()}, batch num: {self.__len__() // batchsize}')

        if self.__len__() == 0:
            print(f'Error occured! Your ds is: TYPE={ds_type}, config:')
            print(opt[ds_type])
            raise RuntimeError(f'[ Err ] Dataset input nums is 0!')

    def __len__(self):
        return (len(self.input_list))

    def debug_save_item(self, input, gt):
        # home = os.environ['HOME']
        util.saveTensorAsImg(input, 'i.png')
        util.saveTensorAsImg(gt, 'o.png')

    def __getitem__(self, idx):
        """Returns a pair of images with the given identifier. This is lazy loading
        of data into memory. Only those image pairs needed for the current batch
        are loaded.

        :param idx: image pair identifier
        :returns: dictionary containing input and output images and their identifier
        :rtype: dictionary

        """
        res_item = {INPUT_FPATH: self.input_list[idx]}

        # different seed for different item, but same for GT and INPUT in one item:
        # the "seed of seed" is fixed for reproducing
        # random.seed(GLOBAL_SEED)
        seed = random.randint(0, 100000)
        input_img = cv2.imread(self.input_list[idx])[:, :, [2, 1, 0]]
        if self.have_gt and self.gt_list[idx].endswith('.hdr'):
            input_img = torch.Tensor(input_img / 255).permute(2, 0, 1)
        else:
            input_img = augment_one_img(input_img, seed, transform=self.transform)
        res_item[INPUT] = input_img

        if self.have_gt:
            res_item[GT_FPATH] = self.gt_list[idx]

            if res_item[GT_FPATH].endswith('.hdr'):
                # gt may be HDR
                # do not augment HDR image.
                gt_img = cv2.imread(self.gt_list[idx], flags=cv2.IMREAD_ANYDEPTH)[:, :, [2, 1, 0]]
                gt_img = torch.Tensor(np.log10(gt_img + 1)).permute(2, 0, 1)
            else:
                gt_img = cv2.imread(self.gt_list[idx])[:, :, [2, 1, 0]]
                gt_img = augment_one_img(gt_img, seed, transform=self.transform)

            res_item[GT] = gt_img
            assert res_item[GT].shape == res_item[INPUT].shape

        return res_item


class DataModule(LightningDataModule):
    def __init__(self, opt, apply_test_transform=False, apply_valid_transform=False):
        super().__init__()
        self.opt = opt
        self.transform = parseAugmentation(opt)
        # self.transform = None
        if apply_test_transform:
            self.test_transform = self.transform
        else:
            self.test_transform = torchvision.transforms.ToTensor()

        if apply_valid_transform:
            self.valid_transform = self.transform
        else:
            self.valid_transform = torchvision.transforms.ToTensor()

        self.training_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        ...

    def setup(self, stage):
        opt = self.opt
        # from train.py
        if stage == "fit":
            # if opt[TRAIN_DATA]:
            assert opt[TRAIN_DATA][INPUT]
            self.training_dataset = ImagesDataset(opt, ds_type=TRAIN_DATA, transform=self.transform, batchsize=opt.batchsize)

            # valid data provided:
            if opt[VALID_DATA] and opt[VALID_DATA][INPUT]:
                self.valid_dataset = ImagesDataset(opt, ds_type=VALID_DATA, transform=self.valid_transform, batchsize=opt.valid_batchsize)

            # no valid data, splt from training data:
            elif opt[VALID_RATIO]:
                print(f'Split valid dataset from training data. Ratio: {opt[VALID_RATIO]}')
                valid_size = int(opt[VALID_RATIO] * len(self.training_dataset))
                train_size = len(self.training_dataset) - valid_size
                torch.manual_seed(233)
                self.training_dataset, self.valid_dataset = torch.utils.data.random_split(self.training_dataset, [
                    train_size, valid_size
                ])
                print(
                    f'Update - training data: {len(self.training_dataset)}; valid data: {len(self.valid_dataset)}')

        # testing phase
        # if stage == 'test':
        if opt[TEST_DATA] and opt[TEST_DATA][INPUT]:
            self.test_dataset = ImagesDataset(opt, ds_type=TEST_DATA, transform=self.test_transform, batchsize=1)

    def train_dataloader(self):
        if self.training_dataset:
            trainloader = torch.utils.data.DataLoader(
                self.training_dataset,
                batch_size=self.opt[BATCHSIZE],
                num_workers=self.opt[DATALOADER_N],
                shuffle=True,
                drop_last=True,
                pin_memory=True
            )
            return trainloader

    def val_dataloader(self):
        if self.valid_dataset:
            return torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=self.opt[VALID_BATCHSIZE],
                shuffle=False,
                num_workers=self.opt[DATALOADER_N]
            )

    def test_dataloader(self):
        if self.test_dataset:
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.opt[DATALOADER_N],
                pin_memory=True
            )

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        ...


if __name__ == '__main__':
    ...
