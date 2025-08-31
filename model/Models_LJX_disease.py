# !/home/user1/anaconda3/envs/project2/bin/python
# -*- coding: utf-8 -*-
"""
@Project ：project2 
@File    ：Models.py
@Author  ：XYH
@Date    ：2022/8/26 20:35 
"""

import torch
from torch.nn import functional as F
from torch_geometric import nn
from torch.nn.parameter import Parameter
from .TransformerConv.TransformerConv import TransformerConv
from model.Feature_Fusion_Backbone import FeatureFusionByConcat, FeatureFusionByGNN


class GNNAE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, heads: int):
        super().__init__()
        
        self.bn_1 = torch.nn.BatchNorm1d(in_channels)
        self.gcn_1 = TransformerConv(in_channels, hidden_size // heads, heads, bias=True)
        
        self.gcn_5 = TransformerConv(hidden_size, hidden_size // heads, heads, bias=True)
        self.bn_5 = torch.nn.BatchNorm1d(hidden_size)
        
        self.gcn_2 = TransformerConv(hidden_size, hidden_size // heads, heads, bias=True)
        self.bn_2 = torch.nn.BatchNorm1d(hidden_size)
        
        self.gcn_3 = TransformerConv(hidden_size, hidden_size // heads, heads, bias=True)
        self.bn_3 = torch.nn.BatchNorm1d(hidden_size)
        
        self.gcn_6 = TransformerConv(hidden_size, hidden_size // heads, heads, bias=True)
        self.bn_6 = torch.nn.BatchNorm1d(hidden_size)
        
        self.bn_4 = torch.nn.BatchNorm1d(hidden_size)
        self.gcn_4 = TransformerConv(hidden_size, in_channels // heads, heads, bias=True)

        self.dropout = torch.nn.Dropout(0.3)
        
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x_origion = x
        
        x= self.bn_1(x)
        x, weight1 = self.gcn_1(x, edge_index)
        x = self.dropout(x)
        x = F.leaky_relu(x, True)
        
        x = self.bn_5(x)
        x, weight5 = self.gcn_5(x, edge_index)
        x = self.dropout(x)
        x = F.leaky_relu(x,True)
        
        x = self.bn_2(x)
        x, weight2 = self.gcn_2(x, edge_index)
        x = self.dropout(x)
        z = F.leaky_relu(x,True)
        
        #三层
        x = self.bn_3(z)
        x, weight3 = self.gcn_3(x, edge_index)
        x = self.dropout(x)
        x = F.leaky_relu(x, True)
        
        x = self.bn_6(x)
        x, weight6 = self.gcn_6(x, edge_index)
        x = self.dropout(x)
        x = F.leaky_relu(x, True)
        
        x = self.bn_4(x)
        x, weight4 = self.gcn_4(x, edge_index)
        x_pred = F.leaky_relu(x, True)
        
        return x_origion, x_pred, z

class GNNAEP(torch.nn.Module):
    def __init__(self, in_channels=None, hidden_size=None, heads=None):
        super(GNNAEP, self).__init__()
        # AE结构学习MRI特征
        # self.pre_gnnaesmri = GNNAE(in_channels=in_channels[0], hidden_size=hidden_size[0], heads=heads[0])
        self.pre_gnnaefmri = GNNAE(in_channels=in_channels, hidden_size=hidden_size, heads=heads)
        
        # for param in self.pre_gnnaefmri.gcn_1.parameters():
        #     param.requires_grad = False
        # for param in self.pre_gnnaefmri.gcn_5.parameters():
        #     param.requires_grad = False
            
        # for param in self.pre_gnnaefmri.gcn_6.parameters():
        #     param.requires_grad = False
        # for param in self.pre_gnnaefmri.gcn_4.parameters():
        #     param.requires_grad = False
        
        # # 提取归一化表型数据特征 模态融合
        # self.feature_fusion = FeatureFusionByConcat()
        # self.non_imaging_MLP = MLP_pred(in_channel=in_channels[2], num_fc=32, num_pred=16)
        # self.addnonimage = FeatureFusionByConcat()
        # MLP 单模态 116*hiddensize 多模态 116*2*hiddensize 表型+16
        # self.pre_MLP = MLP_pred(in_channel=3712, num_fc=256, num_pred=2)#单3712  双7424 non-image7440  节点concat-10624
        # put = 116 * hidden_size
        self.mlp = torch.nn.Linear(116 * hidden_size, 512)
        # self.mlp = torch.nn.Linear(3712, 512)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn = torch.nn.BatchNorm1d(512)
        self.pre = torch.nn.Linear(512, 2)

    def forward(self, args):  # *args
        # pre_ggae 先通过AE结构学习特征
        # with torch.no_grad():
        # x_origion1, x_pred1, z1 = self.pre_gnnaesmri(args[0])
        x_origion2, x_pred2, z = self.pre_gnnaefmri(args)
        # z = self.feature_fusion(z1,z2)
        z = F.normalize(z, p=2, dim=1)
        column_dim = z.shape[1]
        h = 116 * column_dim
        z = z.view(-1, h)
        # z = z.view(-1, 3712) # 7424
        # z3 = self.non_imaging_MLP(args[2])
        # z = self.addnonimage(z,z3)
        # x_pred_MLP = self.pre_MLP(z)
        middle_x = self.mlp(z)
        x = self.relu(middle_x)
        x = self.bn(x)
        x_pred_MLP = self.pre(x)

        return x_origion2, x_pred2, z, x_pred_MLP # x_origion1, x_pred1,
    

class MLP_pred(torch.nn.Module):
    def __init__(self, in_channel: int, num_fc: int, num_pred: int):
        super(MLP_pred, self).__init__()
        self.mlp_pred_pred= torch.nn.Sequential(
            torch.nn.Linear(in_channel, num_fc),
            torch.nn.ReLU(inplace=True),
            # LayerNorm(num_fc),
            torch.nn.BatchNorm1d(num_fc),
            # nn.Dropout(p=0.2),
            torch.nn.Linear(num_fc, num_pred)
        )
    def forward(self, x):
        pre = self.mlp_pred_pred(x)
        pre = torch.softmax(pre, dim=1)
        return pre
    
if __name__ == '__main__':
    model = GNNAE(in_channel=116, hidden_size=32)