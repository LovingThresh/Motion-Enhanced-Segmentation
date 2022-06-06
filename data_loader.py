# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 19:32
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_loader.py
# @Software: PyCharm

import os

import cv2
import torch
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset

from utils.motion_process import *


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomCrop(224, 224)
])


def get_motion_blur_image(input: np.ndarray, motion_angle: int, motion_dis: int):

    PSF = straight_motion_psf((input.shape[0], input.shape[1]), motion_angle, motion_dis)
    R, G, B = cv2.split(input)
    blurred_image_R, blurred_image_G, blurred_image_B = \
        np.abs(make_blurred(R, PSF, 1e-3)), np.abs(make_blurred(G, PSF, 1e-3)), np.abs(make_blurred(B, PSF, 1e-3))
    blurred_image = cv2.merge([blurred_image_R, blurred_image_G, blurred_image_B])

    return np.uint8(blurred_image)


class Motion_Blur_Dataset(Dataset):
    def __init__(self, raw_image_path, raw_mask_path, re_size, data_txt, transformer=transform):
        self.raw_image_path = raw_image_path
        self.raw_mask_path = raw_mask_path
        self.re_size = re_size
        self.data_txt = data_txt
        self.transform = transformer
        with open(self.data_txt, 'r') as f:
            self.file_list = f.read().splitlines()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        self.raw_image = cv2.imread(os.path.join(self.raw_image_path, self.file_list[item].split(',')[0]))
        self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
        self.raw_image = cv2.resize(self.raw_image, self.re_size)
        self.blur_image = get_motion_blur_image(self.raw_image, 45, 10)
        # self.blur_image = get_motion_blur_image(self.blur_image, 180, 5)
        # utils.visualize.plot(self.raw_image)
        # utils.visualize.plot(self.blur_image)
        self.raw_mask = cv2.imread(os.path.join(self.raw_mask_path, self.file_list[item].split(',')[1]))
        self.raw_mask = torch.tensor(self.raw_mask[:, :, 0] > 127, dtype=torch.int8)

        if self.transform is None:
            pass
        else:
            self.transformed = self.transform(image=self.blur_image, mask=self.raw_image)
            self.blur_image, self.raw_image = self.transformed['image'], self.transformed['mask']
            self.blur_image, self.raw_image = \
                transforms.ToTensor()(self.blur_image), transforms.ToTensor()(self.raw_image)

        return self.blur_image, self.raw_image, self.raw_mask


class ItemPool:

    def __init__(self, pool_size=50):
        self.items = []
        self.pool_size = pool_size

    def __call__(self, in_items):
        out_items = []
        for item in in_items:
            item = torch.unsqueeze(item, dim=0)
            if len(self.items) < self.pool_size:
                self.items.append(item)
                out_items.append(item)
            else:
                if torch.rand(1) > 0.5:
                    idx = torch.randint(self.pool_size, [1])
                    out_item, self.items[idx] = self.items[idx], item
                    out_items.append(out_item)
                else:
                    out_items.append(item)

        return torch.cat(out_items, dim=0)

