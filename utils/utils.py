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
