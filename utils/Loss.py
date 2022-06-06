# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 13:57
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Loss.py
# @Software: PyCharm
import torch
import torch.nn as nn


# How to define a custom loss function
class custom_Loss(nn.Module):
    def __init__(self):
        super(custom_Loss, self).__init__()

    @staticmethod
    def forward(x, y):
        return torch.sum(torch.pow((x - y), 2), dim=0)


# To implement several loss function for Deblur
