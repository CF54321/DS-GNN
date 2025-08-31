# !/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
@Project ：project2
@File    ：utils.py
@Author  ：XYH
@Date    ：2022/8/25 17:36
"""

import torch
import numpy as np
from torch import Tensor


def set_seed(init_seed=1000):
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)  # 用于当前GPU生成随机数设置种子
    np.random.seed(init_seed)  # 用于numpy的随机数


# 定义混淆矩阵
def confusion_matrix(pred: Tensor, target: Tensor):
    TP = ((pred == 1) & (target == 1)).sum()
    TN = ((pred == 0) & (target == 0)).sum()
    # FN = ((pred == 1) & (target == 0)).sum()
    # FP = ((pred == 0) & (target == 1)).sum()
    FP = ((pred == 1) & (target == 0)).sum()
    FN = ((pred == 0) & (target == 1)).sum()
    return TP, TN, FP, FN


# 边的权重参数转化成权重矩阵
def weight_to_matrix(weights, dim):
    mesh = np.zeros((dim, dim))
    index = weights[0]
    value = weights[1]
    for i in range(index.shape[1]):
        mesh[index[0, i], index[1, i]] = value[i, 0]
    return mesh
