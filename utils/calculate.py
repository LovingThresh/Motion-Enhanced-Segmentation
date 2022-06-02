# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 15:31
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : calculate.py
# @Software: PyCharm

import math
import numpy as np


def calculate_loss(output: np.ndarray, target: np.ndarray):
    output = output.astype(np.int64)
    target = target.astype(np.int64)
    difference = np.abs(output - target)
    sum_difference = difference.sum()
    array_size = output.size
    return sum_difference / array_size
