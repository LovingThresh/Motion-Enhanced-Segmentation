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


def gan_loss(input, target):
    return -input.mean() if target else input.mean()


def iou(input, target):

    intersection = input * target
    union = (input + target) - intersection
    Iou = (torch.sum(intersection) + torch.tensor(1e-8)) / (torch.sum(union) + torch.tensor(1e-8))
    return Iou


def pr(input, target):

    tp = torch.sum(target * input)
    pp = torch.sum(input)

    return tp / (pp + 1e-8)


def re(input, target):

    tp = torch.sum(target * input)
    pp = torch.sum(target)

    return tp / (pp + 1e-8)


def f1(input, target):

    p = pr(input, target)
    r = re(input, target)

    return 2 * p * r / (p + r + 1e-8)


def Asymmetry_Binary_Loss(input, target, alpha=100):
    # 纯净状态下alpha为1
    # 想要损失函数更加关心裂缝的标签值1
    y_pred, y_true = input, target
    y_true_0, y_pred_0 = y_true[:, 0, :, :] , y_pred[:, 0, :, :]
    # y_true_0, y_pred_0 = y_true[:, :, :, 0] * 255, y_pred[:, :, :, 0] * 255
    y_true_1, y_pred_1 = y_true[:, 1, :, :] * alpha, y_pred[:, 1, :, :] * alpha
    mse = torch.nn.MSELoss()
    return mse(y_true_0, y_pred_0) + mse(y_true_1, y_pred_1)


def correlation(input, target):
    input_vector =  input.reshape((1, -1))
    target_vector = target.reshape((1, -1))
    return torch.corrcoef(torch.cat([input_vector, target_vector], dim=0))[0, 1]
