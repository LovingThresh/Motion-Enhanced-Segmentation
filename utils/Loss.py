# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 13:57
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Loss.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torchvision.models as models


# How to define a custom loss function
class custom_Loss(nn.Module):
    def __init__(self):
        super(custom_Loss, self).__init__()

    @staticmethod
    def forward(x, y):
        return torch.sum(torch.pow((x - y), 2), dim=0)


# To implement several loss function for Deblur
class ContentLoss:
    def __init__(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


def contentFunc():
    conv_3_3_layer = 14
    cnn = models.vgg19(pretrained=True).features
    cnn = cnn.cuda()
    model = nn.Sequential()
    model = model.cuda()
    for i, layer in enumerate(list(cnn)):
        model.add_module(str(i), layer)
        if i == conv_3_3_layer:
            break
    return model


class PerceptualLoss:

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


def perceptual_loss(input, target):
    P = PerceptualLoss(nn.MSELoss())
    return 100 * P.get_loss(input, target)
