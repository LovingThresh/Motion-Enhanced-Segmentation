# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 19:30
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : motion_process.py
# @Software: PyCharm

# Inference : https://blog.csdn.net/googler_offer/article/details/88841048
# Inference : https://www.likecs.com/show-165731.html

import numpy.fft as fft
import numpy as np
import math
import cv2


def straight_motion_psf(image_size: tuple, motion_angle: int, motion_dis: int):
    """
    直线运动模糊
    :param image_size:
    :param motion_angle:
    :param motion_dis:
    :return:
    """
    psf = np.zeros(image_size)
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2

    sin_val = math.sin(motion_angle * math.pi / 180)
    cos_val = math.cos(motion_angle * math.pi / 180)

    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        psf[int(x_center - x_offset), int(y_center + y_offset)] = 1

    return psf / psf.sum()


def gaussian_blur_process(input: np.ndarray, degree=21):
    """
    高斯模糊
    :param input:
    :param degree:
    :return:
    """
    blurred = cv2.GaussianBlur(input, ksize=(degree, degree), sigmaX=0, sigmaY=0)
    return blurred


def make_blurred(input: np.ndarray, psf: np.ndarray, eps: float):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(psf) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


def wiener(input: np.ndarray, psf: np.ndarray, eps: float, K=0.01):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(psf) + eps
    # np.conj是计算共轭值
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


def rotate_blur_process(input: np.ndarray, Num: int = 10):
    output = input.copy()
    output = output.astype(np.float32)
    row, col, channel = output.shape
    xx = np.arange(col)
    yy = np.arange(row)

    x_mask = np.tile(xx, (row, 1))
    y_mask = np.tile(yy, (col, 1))
    y_mask = np.transpose(y_mask)

    center_y = (row - 1) / 2.0
    center_x = (col - 1) / 2.0

    RR = np.sqrt((x_mask - center_x) ** 2 + (y_mask - center_y) ** 2)

    angle = np.arctan2(y_mask - center_y, x_mask - center_x)

    arr = (np.arange(Num) + 1) / 100.0

    for i in range(row):
        for j in range(col):
            T_angle = angle[i, j] + arr

            new_x = RR[i, j] * np.cos(T_angle) + center_x
            new_y = RR[i, j] * np.sin(T_angle) + center_y

            int_x = new_x.astype(int)
            int_y = new_y.astype(int)

            int_x[int_x > col - 1] = col - 1
            int_x[int_x < 0] = 0
            int_y[int_y < 0] = 0
            int_y[int_y > row - 1] = row - 1

            output[i, j, 0] = input[int_y, int_x, 0].sum() / Num
            output[i, j, 1] = input[int_y, int_x, 1].sum() / Num
            output[i, j, 2] = input[int_y, int_x, 2].sum() / Num

    return output
