# !/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
@Project ：project2
@File    ：get_label.py
@Author  ：XYH
@Date    ：2022/8/24 15:03
"""

import os
import numpy as np
import pandas as pd
ID = [int(i[0:-4]) for i in os.listdir('E:/Data/asd_new/Sorted_pearson_mat_5')]
all_ID = pd.read_csv('E:/Data/asd/Phenotypic_V1_0b_preprocessed1.csv', usecols=['SUB_ID']).values.reshape(-1)
all_labels = pd.read_csv('E:/Data/asd/Phenotypic_V1_0b_preprocessed1.csv', usecols=['DX_GROUP']).values.reshape(-1) - 1
meta_info_old = pd.read_csv('E:/Data/asd/Phenotypic_V1_0b_preprocessed1.csv')
labels_new = []
ID_new = []
ID_labels = pd.DataFrame()
meta_info = pd.DataFrame()
for (i, index) in enumerate(all_ID):
    if index in ID:
        labels_new.append(all_labels[i])
        meta_info = pd.concat([meta_info, meta_info_old.iloc[[i]]])
ID_labels['SUB_ID'] = ID
ID_labels['labels'] = labels_new
ID_labels.to_csv('./ID_labels.csv', index=False)
meta_info.to_csv('./meta_info.csv', index=False)




