# !/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
@Project ：project2
@File    ：Feature_Fusion_Backbone.py
@Author  ：XYH
@Date    ：2022/8/24 16:52
"""
import numpy as np
from scipy.sparse import coo_matrix
from torch import nn
import torch
from torch_geometric.nn import BatchNorm, LayerNorm, TransformerConv
from torch_geometric.nn import global_add_pool as gap
from torch import sparse_coo_tensor


class FeatureFusionByConcat(nn.Module):
    def __init__(self):
        super(FeatureFusionByConcat, self).__init__()

    def forward(self, *features):
        return torch.cat(features, dim=1)


class FeatureFusionByGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, heads: int, p=0):
        super(FeatureFusionByGNN, self).__init__()
        self.conv = TransformerConv(in_channels=in_channels, out_channels=hidden_size // heads, heads=heads, bias=True,
                                    dropout=p)
        self.bn = BatchNorm(hidden_size)  # LayerBN
        self.ln = LayerNorm(hidden_size)
        self.relu = torch.nn.LeakyReLU(inplace=True)

    # def edge_fusion(self, edge1, edge2):
    #     #TODO: 合并边特征
    #     indice = torch.cat([edge1, edge2], dim=1)
    #     data = torch.tensor(np.ones(len(indice)))
    #     # coo_tensor = sparse_coo_tensor(, size=(108, 108))

    def forward(self, edge_index1, edge_index2, batch, *features):
        edge_index = torch.unique(torch.cat([edge_index1, edge_index2], dim=1), dim=1)
        x = torch.cat(features, dim=1)
        x = self.conv(x, edge_index)
        x = self.ln(x, batch)
        # x = self.bn(x)
        x = self.relu(x)

        features = gap(x, batch)
        return features
