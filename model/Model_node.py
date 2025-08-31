
import os
import os.path as osp
from abc import ABC
import sys
# sys.path.append('/home/atlas/CF/multiclassification')  
sys.path.append('/home/toor/CF/multiclassification') 

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from tools.utils import set_seed
import torch
from torch import sparse_coo_tensor

import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv

import random
# import data.population_graph_dataset as gd
from torch_sparse import SparseTensor

from .TransformerConv.TransformerConv import TransformerConv
from scipy.spatial import distance

from torch.nn.parameter import Parameter
import math
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import ChebConv

import data.population_graph_dataset as gd

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_train_test_masks(labels, idx_train, idx_val, idx_test):
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    return train_mask, val_mask, test_mask # y_train, y_val, y_test, 

def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=1) # , step=100, verbose=1

    labels = np.array(labels)
    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)

    # print("Number of labeled samples %d" % len(train_ind))
    # print("Number of features selected %d" % x_data.shape[1])

    return x_data

class MLP_pred(torch.nn.Module):
    def __init__(self, in_channel: int, num_fc: int, num_pred: int):
        super(MLP_pred, self).__init__()
        self.mlp_pred_pred= torch.nn.Sequential(
            torch.nn.Linear(in_channel, num_fc),
            torch.nn.ReLU(inplace=True),
            # LayerNorm(num_fc),
            torch.nn.BatchNorm1d(num_fc),
            nn.Dropout(p=0.2),
            # torch.nn.Linear(num_fc, num_fc),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.BatchNorm1d(num_fc),
            # nn.Dropout(p=0.2),
            torch.nn.Linear(num_fc, num_pred)

        )
    def forward(self, x):
        pre = self.mlp_pred_pred(x)
        # pre = torch.softmax(pre, dim=1)
        
        return pre
    
class MLP(torch.nn.Module):
    def __init__(self, in_channel: int, num_fc: int):
        super(MLP, self).__init__()
        self.mlp_pred_pred= torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channel),
            torch.nn.Linear(in_channel, num_fc)
        )
    def forward(self, x):
        pre = self.mlp_pred_pred(x)
        return pre

class GCNTransformer(nn.Module):
    def __init__(self, in_channel: int, hidden_size: int, head: int, nclass, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        
        # self.gcn_1 = TransformerConv(in_channel, hidden_size//head, head, bias=True)
        self.gcn_1 = ChebConv(in_channel, hidden_size, 3)
        # self.gcn_1 = SAGEConv(in_channel, hidden_size, normalize=False, bias=True)
        # self.gcn_1 = GATConv(in_channel, hidden_size // head, head)
        self.bn_1 = torch.nn.BatchNorm1d(hidden_size)
        
        # self.gcn_2 = TransformerConv(hidden_size, hidden_size//head, head, bias=True)
        self.gcn_2 = ChebConv(hidden_size, hidden_size, 3)
        # self.gcn_2 = SAGEConv(hidden_size, hidden_size, normalize=False, bias=True)
        # self.gcn_2 = GATConv(hidden_size, hidden_size // head, head)
        self.bn_2 = torch.nn.BatchNorm1d(hidden_size)

        # self.bn_3 = torch.nn.BatchNorm1d(hidden_size)
        # self.gcn_3 = TransformerConv(hidden_size, hidden_size//head, head, bias=True)

        self.dropout = dropout
        self.pre_MLP = MLP_pred(in_channel=hidden_size, num_fc=int(hidden_size/2), num_pred=nclass)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.gcn_1(x, edge_index, edge_attr) # , weight1
        x = self.bn_1(x)
        x = F.leaky_relu(x, True)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn_2(x, edge_index, edge_attr)  # , weight2
        x = self.bn_2(x)
        x = F.leaky_relu(x, True)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x, weight3 = self.gcn_3(x, edge_index, edge_attr)
        # x = self.bn_3(x)
        # x = F.leaky_relu(x, True)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.pre_MLP(x))

        return x
    
class GCNTransforme_MLP(nn.Module):
    def __init__(self, in_channel: int, hidden_size: int, head: int, nclass, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        
        # self.gcn_1 = TransformerConv(in_channel, hidden_size//head, head, bias=True)
        self.gcn_1 = ChebConv(in_channel, hidden_size, 3)
        # self.gcn_1 = SAGEConv(in_channel, hidden_size, normalize=False, bias=True)
        # self.gcn_1 = GATConv(in_channel, hidden_size // head, head)
        self.bn_1 = torch.nn.BatchNorm1d(hidden_size)
        
        # self.gcn_2 = TransformerConv(hidden_size, hidden_size//head, head, bias=True)
        self.gcn_2 = ChebConv(hidden_size, hidden_size, 3)
        # self.gcn_2 = SAGEConv(hidden_size, hidden_size, normalize=False, bias=True)
        # self.gcn_2 = GATConv(hidden_size, hidden_size // head, head)
        self.bn_2 = torch.nn.BatchNorm1d(hidden_size)

        # self.bn_3 = torch.nn.BatchNorm1d(hidden_size)
        # self.gcn_3 = TransformerConv(hidden_size, hidden_size//head, head, bias=True)

        self.dropout = dropout
        self.pre_MLP = MLP_pred(in_channel=hidden_size, num_fc=int(hidden_size/2), num_pred=nclass)
        self.mlp = MLP(in_channel=3, num_fc=8)

    def forward(self, data, no_image_feature):

        w = self.mlp(no_image_feature)
        W = gd.get_similarity_tensor(w)
        W = W.astype(np.float32)
        W = torch.tensor(W)
        W = W.to('cuda:0')
        W = (W + torch.ones_like(W)) / 2
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        X = data.x
        # C = data.S1 * data.S2
        # E1 = torch.tensor(1)
        # E0 = torch.tensor(0)
        # E1 = E1.to('cuda:0')
        # E0 = E0.to('cuda:0')
        # C = torch.where(C > 0.2, E1, E0)
        # A = W * C
        A = W * data.S2
        edge_index = torch.transpose(A.nonzero(), 0, 1)
        edge_attr = A[A.nonzero()[:, 0], A.nonzero()[:, 1]].unsqueeze(1)

        x = self.gcn_1(X, edge_index, edge_attr) # , weight1
        x = self.bn_1(x)
        x = F.leaky_relu(x, True)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn_2(x, edge_index, edge_attr)  # , weight2
        x = self.bn_2(x)
        x = F.leaky_relu(x, True)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x, weight3 = self.gcn_3(x, edge_index, edge_attr)
        # x = self.bn_3(x)
        # x = F.leaky_relu(x, True)
        x = F.dropout(x, self.dropout, training=self.training)

        S2 = gd.get_similarity_tensor(x)
        S2 = S2.astype(np.float32)
        S2 = torch.tensor(S2)
        S2 = S2.to('cuda:0')
        # C0 = data.S1 * S2
        # C0 = torch.where(C0 > 0.2, E1, E0)
        # A0 = W * C0
        A0 = W * S2
        edge_index0 = torch.transpose(A0.nonzero(), 0, 1)
        edge_attr0 = A0[A0.nonzero()[:, 0], A0.nonzero()[:, 1]].unsqueeze(1)
        x0 = self.gcn_1(X, edge_index0, edge_attr0) # , weight1
        x0 = self.bn_1(x0)
        x0 = F.leaky_relu(x0, True)
        x0 = F.dropout(x0, self.dropout, training=self.training)
        x0 = self.gcn_2(x0, edge_index0, edge_attr0)  # , weight2
        x0 = self.bn_2(x0)
        x0 = F.leaky_relu(x0, True)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x, weight3 = self.gcn_3(x, edge_index, edge_attr)
        # x = self.bn_3(x)
        # x = F.leaky_relu(x, True)
        x0 = F.dropout(x0, self.dropout, training=self.training)

        x0 = F.relu(self.pre_MLP(x0))

        return x0

class GCNTransformer_Pre(nn.Module):
    def __init__(self, in_channel: int, hidden_size: int, head: int, nclass, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        
        # self.gcn_1 = TransformerConv(in_channel, hidden_size//head, head, bias=True)
        self.gcn_1 = ChebConv(in_channel, hidden_size, 3)
        # self.gcn_1 = SAGEConv(in_channel, hidden_size, normalize=False, bias=True)
        # self.gcn_1 = GATConv(in_channel, hidden_size // head, head)
        self.bn_1 = torch.nn.BatchNorm1d(hidden_size)
        
        # self.gcn_2 = TransformerConv(hidden_size, hidden_size//head, head, bias=True)
        self.gcn_2 = ChebConv(hidden_size, hidden_size, 3)
        # self.gcn_2 = SAGEConv(hidden_size, hidden_size, normalize=False, bias=True)
        # self.gcn_2 = GATConv(hidden_size, hidden_size // head, head)
        self.bn_2 = torch.nn.BatchNorm1d(hidden_size)

        # self.bn_3 = torch.nn.BatchNorm1d(hidden_size)
        # self.gcn_3 = TransformerConv(hidden_size, hidden_size//head, head, bias=True)

        self.dropout = dropout
        self.pre_MLP = MLP_pred(in_channel=hidden_size, num_fc=int(hidden_size/2), num_pred=nclass)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.gcn_1(x, edge_index, edge_attr) # , weight1
        x = self.bn_1(x)
        x = F.leaky_relu(x, True)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn_2(x, edge_index, edge_attr)  # , weight2
        x = self.bn_2(x)
        x = F.leaky_relu(x, True)
        x0 = x
        # x = F.dropout(x, self.dropout, training=self.training)
        # x, weight3 = self.gcn_3(x, edge_index, edge_attr)
        # x = self.bn_3(x)
        # x = F.leaky_relu(x, True)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.pre_MLP(x))

        return x0


class GCNTransformer_ZJ(nn.Module):
    def __init__(self, in_channel: int, hidden_size: int, head: int, nclass, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.gcn_1 = TransformerConv(in_channel, hidden_size//head, head, bias=True)
        # self.gcn_1 = ChebConv(in_channel, hidden_size, 3)
        # self.gcn_1 = SAGEConv(in_channel, hidden_size, normalize=False, bias=True)
        # self.gcn_1 = GATConv(in_channel, hidden_size // head, head)
        self.bn_1 = torch.nn.BatchNorm1d(hidden_size)
        
        self.gcn_2 = TransformerConv(hidden_size, hidden_size//head, head, bias=True)
        # self.gcn_2 = ChebConv(hidden_size, hidden_size, 3)
        # self.gcn_2 = SAGEConv(hidden_size, hidden_size, normalize=False, bias=True)
        # self.gcn_2 = GATConv(hidden_size, hidden_size // head, head)
        self.bn_2 = torch.nn.BatchNorm1d(hidden_size)

        # self.bn_3 = torch.nn.BatchNorm1d(hidden_size)
        # self.gcn_3 = TransformerConv(hidden_size, hidden_size//head, head, bias=True)

        self.dropout = torch.nn.Dropout(0.5)
        self.pre_MLP = MLP_pred(in_channel=hidden_size, num_fc=int(hidden_size/2), num_pred=nclass)
        
    # @torch.no_grad()
    def forward(self, data):
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # x = self.gcn_1(x, edge_index, edge_attr) # , weight1
        x, weight1 = self.gcn_1(x, edge_index)
        x = self.bn_1(x)
        x = F.leaky_relu(x, True)
        x = self.dropout(x)
        # x = self.gcn_2(x, edge_index, edge_attr)  # , weight2
        x, weight2 = self.gcn_2(x, edge_index)
        x = self.bn_2(x)
        x = F.leaky_relu(x, True)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x, weight3 = self.gcn_3(x, edge_index, edge_attr)
        # x = self.bn_3(x)
        # x = F.leaky_relu(x, True)
        x = self.dropout(x)
        x0 = x
        x = F.relu(self.pre_MLP(x))

        return x, x0
    