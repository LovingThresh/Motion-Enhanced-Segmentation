# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 19:32
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_loader.py
# @Software: PyCharm

import os

import torch
import random
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from utils.motion_process import *

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomCrop(512, 512)
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
        self.raw_mask = (self.raw_mask[:, :, 0] > 127).astype(np.uint8)
        self.raw_mask = np.expand_dims(self.raw_mask, axis=-1)
        self.raw_mask = np.concatenate([np.ones_like(self.raw_mask, dtype=np.uint8) - self.raw_mask, self.raw_mask],
                                       axis=-1)

        if self.transform is None:
            pass
        else:
            self.transformed = self.transform(image=self.blur_image, masks=[self.raw_image, self.raw_mask])
            self.blur_image, [self.raw_image, self.raw_mask] = self.transformed['image'], self.transformed['masks']
            self.blur_image, self.raw_image = \
                transforms.ToTensor()(self.blur_image), transforms.ToTensor()(self.raw_image)
            self.blur_image, self.raw_image = \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(self.blur_image), \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(self.raw_image)
            self.raw_mask = torch.tensor(self.raw_mask, dtype=torch.float32)
            self.raw_mask = torch.transpose(self.raw_mask, 0, 2)
            self.raw_mask = torch.transpose(self.raw_mask, 1, 2)
        return self.blur_image, self.raw_image, self.raw_mask
        # return self.raw_image, self.blur_image, self.raw_mask


def get_Fog_blur_image(input: np.ndarray, brightness=0.5, concentration=0.08, center=None):
    img_fog = input / 255.
    (row, col, chs) = input.shape
    size = math.sqrt(max(row, col))
    brightness = brightness
    beta = concentration
    size = size
    if center is None:
        center = (row // 2, col // 2)
    for j in range(row):
        for s_l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (s_l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_fog[j][s_l][:] = img_fog[j][s_l][:] * td + brightness * (1 - td)

    return img_fog


class Fog_Blur_Dataset(Dataset):
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
        self.blur_image = get_Fog_blur_image(self.raw_image)
        self.raw_mask = cv2.imread(os.path.join(self.raw_mask_path, self.file_list[item].split(',')[1]))
        self.raw_mask = (self.raw_mask[:, :, 0] > 127).astype(np.uint8)
        self.raw_mask = np.expand_dims(self.raw_mask, axis=-1)
        self.raw_mask = np.concatenate([np.ones_like(self.raw_mask, dtype=np.uint8) - self.raw_mask, self.raw_mask],
                                       axis=-1)

        if self.transform is None:
            pass
        else:
            self.transformed = self.transform(image=self.blur_image, masks=[self.raw_image, self.raw_mask])
            self.blur_image, [self.raw_image, self.raw_mask] = self.transformed['image'], self.transformed['masks']
            self.blur_image, self.raw_image = \
                transforms.ToTensor()(self.blur_image), transforms.ToTensor()(self.raw_image)
            self.blur_image, self.raw_image = \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(self.blur_image), \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(self.raw_image)
            self.blur_image = self.blur_image.float()
            self.raw_mask = torch.tensor(self.raw_mask, dtype=torch.float32)
            self.raw_mask = torch.transpose(self.raw_mask, 0, 2)
            self.raw_mask = torch.transpose(self.raw_mask, 1, 2)
        return self.blur_image, self.raw_image, self.raw_mask
        # return self.raw_image, self.blur_image, self.raw_mask


class ItemPool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def __call__(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images

# class ItemPool:
#
#     def __init__(self, pool_size=50):
#         self.items = []
#         self.pool_size = pool_size
#
#     def __call__(self, in_items):
#         out_items = []
#         for item in in_items:
#             item = torch.unsqueeze(item, dim=0)
#             if len(self.items) < self.pool_size:
#                 self.items.append(item)
#                 out_items.append(item)
#             else:
#                 if torch.rand(1) > 0.5:
#                     idx = torch.randint(self.pool_size, [1])
#                     out_item, self.items[idx] = self.items[idx], item
#                     out_items.append(out_item)
#                 else:
#                     out_items.append(item)
#
#         return torch.cat(out_items, dim=0)

# Test The Code
