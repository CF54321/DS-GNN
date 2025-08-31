# !/home/user1/anaconda3/envs/project2/bin/python
# -*- coding: utf-8 -*-
"""
@Project ：project2
@File    ：Mode_Branch.py
@Author  ：XYH
@Date    ：2022/8/24 16:50
"""

import torch
from torch import nn
import torch
from torch_geometric.nn import BatchNorm, LayerNorm
from .TransformerConv.TransformerConv import TransformerConv
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool as gap


class CNNBranch(nn.Module):
    def __init__(self):
        super(CNNBranch).__init__()
        # TODO
        pass

    def forward(self, x):
        pass


class GNNBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, heads: int, p=0):
        super(GNNBranch, self).__init__()
        # 卷积层
        self.conv1 = TransformerConv(in_channels=in_channels, out_channels=hidden_size // heads, heads=heads, bias=True,
                                     dropout=p)
        # self.bn1 = BatchNorm(hidden_size)  # LayerBN
        self.ln1 = LayerNorm(hidden_size)
        self.relu1 = torch.nn.LeakyReLU(inplace=True)

        self.conv2 = TransformerConv(in_channels=hidden_size, out_channels=hidden_size // heads, heads=heads, bias=True,
                                     dropout=p)
        self.ln2 = LayerNorm(hidden_size)
        self.relu2 = torch.nn.LeakyReLU(inplace=True)
        

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 两层卷积，考虑到脑区数目需对应，不使用池化层
        x, weight1 = self.conv1(x=x, edge_index=edge_index)
        x = self.relu1(x)
        x = self.ln1(x, batch)
        # x = self.bn1(x)

        x, weight2 = self.conv2(x=x, edge_index=edge_index)
        x = self.relu2(x)
        x = self.ln2(x, batch)
        
    
        feature = gap(x, batch)
        return feature, weight1, weight2

class GNNBranch1(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, heads: int, p=0):
        super(GNNBranch1, self).__init__()
        # 卷积层
        self.conv = TransformerConv(in_channels=in_channels, out_channels=hidden_size // heads, heads=heads, bias=True,
                                    dropout=p)
        self.bn = BatchNorm(hidden_size)  # LayerBN
        self.ln = LayerNorm(hidden_size)
        self.relu = torch.nn.LeakyReLU(inplace=True)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # 卷积，考虑到脑区数目需对应，不使用池化层
        x, weight = self.conv(x=x, edge_index=edge_index)
        # x = self.bn(x)
        x = self.ln(x, batch)
        x = self.relu(x)
        return x, edge_index, batch, weight
