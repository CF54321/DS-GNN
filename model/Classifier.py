# !/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
@Project ：project2 
@File    ：Classifier.py
@Author  ：XYH
@Date    ：2022/8/26 20:11 
"""

from torch import nn
import torch


class Classifier(nn.Module):
    def __init__(self, in_channels, num_fc, num_class):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, num_fc),
            nn.ReLU(inplace=True),
            # LayerNorm(num_fc),
            nn.BatchNorm1d(num_fc),
            # nn.Dropout(p=0.2),
            nn.Linear(num_fc, num_class)
        )

    def forward(self, x):
        pre = self.classifier(x)
        pre = pre.squeeze(-1)
        # pre = torch.softmax(x, dim=1)
        return pre

