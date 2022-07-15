# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 13:51
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : ConvolutionLayer.py
# @Software: PyCharm

# Convolution
# in_channels out_channels kernel_size stride padding dilation group
import math
import torch
import torch.nn.functional as F


def matrix_multiplication_for_conv2d(input, kernel, bias=0, stride=1, padding=0):

    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    input_h, input_w = input.shape
    kernel_h, kernel_w = kernel.shape

    output_h = (math.floor((input_h - kernel_h) / stride) + 1)
    output_w = (math.floor((input_w - kernel_w) / stride) + 1)
    output = torch.zeros((output_h, output_w))

    for i in range(0, input_h - kernel_h + 1, stride):  # 对高度维进行遍历
        for j in range(0, input_w - kernel_w + 1, stride):  # 对宽度进行遍历
            region = input[i : i + kernel_h, j : j + kernel_w]
            output[int(i / stride), int(j / stride)] = torch.sum(region * kernel) + bias

    return output

# group convolution and dilated convolution
