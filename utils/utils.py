# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 9:34
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : utils.py
# @Software: PyCharm
import os

import torch
import random
import numpy as np


def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_ = {}
    for k in list(keys)[start:end]:
        c = k
        dict_[c] = adict[k]
    return dict_


def dict_load(adict, cdict):
    # adict source / cdict target
    for i, j in zip(adict, cdict):
        cdict[j] = adict[i]
    return cdict


def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)


def parameters_is_training(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


from model import define_G
mode = 'segmentation'
model = define_G(3, 2, 64, 'resnet_9blocks', learn_residual=False, norm='instance', mode=mode)


def get_changed_model(model_path):
    model_torch_dict = torch.load(model_path)
    segmentation_model_torch_dict_source = dict_slice(model_torch_dict, 0, 70)
    segmentation_model_torch_dict_target = model.state_dict()
    segmentation_model_torch_dict = dict_load(segmentation_model_torch_dict_source, segmentation_model_torch_dict_target)

    torch.save(segmentation_model_torch_dict, model_path[:-3] + '_seg.pt')


get_changed_model(r'M:\MotionBlur-Segmentation\关键模型\512_Blur_orientation\Epoch_199_eval_37.90572070417733.pt')
blur_model_dict = torch.load(r'M:\MotionBlur-Segmentation\关键模型\512_Blur_orientation\Epoch_199_eval_37.90572070417733.pt')
seg_model_dict = torch.load(r'M:\MotionBlur-Segmentation\关键模型\512_Blur_orientation\Epoch_199_eval_37.90572070417733_seg.pt')
