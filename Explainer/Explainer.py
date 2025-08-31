#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Explainer: population-graph feature masking to estimate node (ROI) contributions.
- Logic preserved from your original script.
- All paths and constants are now configurable via CLI.
"""

import os
import sys
import copy
import csv
import numpy as np
import torch
import pandas as pd

from scipy.spatial import distance
from scipy.stats import entropy, pearsonr  # kept imports as in your original

# Optional: append your project root so local modules resolve
def maybe_append_project_root(project_root: str | None):
    if project_root and project_root not in sys.path:
        sys.path.append(project_root)

import argparse


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Explainer (masking-based) for population-graph features.")
    # project layout
    p.add_argument("--project-root", type=str, default=None,
                   help="Optional path to append to sys.path so that tools/model/data imports work.")
    # pretrained model directories
    p.add_argument("--pre-model-dir", type=str, required=True,
                   help="Directory containing GNNAEP pretrained weights (sorted by name; pick by fold index).")
    p.add_argument("--node-model-dir", type=str, required=True,
                   help="Directory containing population GCN weights (sorted by name; pick by fold index).")
    # data sources/paths
    p.add_argument("--phenotype-csv", type=str, required=True,
                   help="CSV with columns SUB_ID, labels_age, labels_site, labels_sex, labels_class.")
    p.add_argument("--dataset-root", type=str, required=True,
                   help="GNNDataset root directory (processed individual graphs).")
    # output
    p.add_argument("--out-csv", type=str, required=True,
                   help="Output CSV to append explainer results.")
    # run config
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=822)
    p.add_argument("--num-folds", type=int, default=10,
                   help="Number of stratified folds to iterate (will loop 0..num-folds-1).")
    p.add_argument("--fold-start", type=int, default=0,
                   help="Start fold index (inclusive).")
    p.add_argument("--fold-end", type=int, default=None,
                   help="End fold index (exclusive). If None, uses num-folds.")
    # cohort sizes / model hyperparams (kept defaults matching your script)
    p.add_argument("--sample-count", type=int, default=233, help="Use first N subjects.")
    p.add_argument("--n_class0", type=int, default=143, help="Count of class 0 (used for split index logic).")
    p.add_argument("--n_class1", type=int, default=90, help="Count of class 1.")
    p.add_argument("--in-channels", type=int, default=116, help="GNNAEP input channels.")
    p.add_argument("--hidden-size", type=int, default=32, help="GNNAEP hidden size.")
    p.add_argument("--heads", type=int, default=1, help="GNNAEP heads.")
    p.add_argument("--num-features", type=int, default=3712, help="Population GCN feature dimension.")
    p.add_argument("--gcn-hidden", type=int, default=256, help="Population GCN hidden size.")
    p.add_argument("--gcn-heads", type=int, default=2, help="Population GCN attention heads.")
    p.add_argument("--gcn-classes", type=int, default=2, help="Population GCN number of classes.")
    p.add_argument("--gcn-dropout", type=float, default=0.2, help="Population GCN dropout.")
    return p.parse_args()


args = parse_args()
maybe_append_project_root(args.project_root)

# -----------------------------
# Project imports (unchanged)
# -----------------------------
from torch_geometric.data import Data, DataLoader, Dataset
from data.GNN_dataset import GNNDataset
from model.Models_LJX_disease import GNNAEP
import model.Model_node as Gcn
from tools.utils import set_seed
import data.population_graph_dataset as gd


# -----------------------------
# Globals used by evaluate() (kept as in your original)
# -----------------------------
model = None
labels_class = None
criclass = None


# -----------------------------
# Functions (logic preserved)
# -----------------------------
def evaluate(data, val_mask, val_set, v_count_less_448, v_count_greater_448):
    """Validation forward pass (unchanged)."""
    with torch.no_grad():
        model.eval()
        out, val_feature = model(data)
        val_out = out[val_mask]
        val_y = labels_class[val_mask]
    return val_feature, val_y, val_out


def mask_feature(population_graph_feature, feature_idx):
    """Mask a 32-wide feature block (unchanged helper; unused in your current loop)."""
    masked_population_graph_feature = population_graph_feature
    masked_population_graph_feature[:, feature_idx: feature_idx + 32] = 0
    return masked_population_graph_feature


# -----------------------------
# Main
# -----------------------------
def main():
    global model, labels_class, criclass

    set_seed(init_seed=args.seed)
    dev = torch.device(args.device)

    # prepare output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    # load fold weight filenames (sorted, pick by fold index)
    pre_model_files = sorted([f for f in os.listdir(args.pre_model_dir) if f.endswith((".pth", ".pt"))])
    node_model_files = sorted([f for f in os.listdir(args.node_model_dir) if f.endswith((".pth", ".pt"))])
    if len(pre_model_files) == 0:
        raise FileNotFoundError(f"No weight files found in --pre-model-dir: {args.pre_model_dir}")
    if len(node_model_files) == 0:
        raise FileNotFoundError(f"No weight files found in --node-model-dir: {args.node_model_dir}")

    # phenotype
    pheno = pd.read_csv(args.phenotype_csv)
    label_ID = pheno['SUB_ID'].values.reshape(-1)[:args.sample_count]
    labels_age = pheno['labels_age'].values[:args.sample_count]
    labels_site = pheno['labels_site'].values[:args.sample_count]
    labels_sex = pheno['labels_sex'].values[:args.sample_count]
    labels_class_csv = pheno['labels_class'].values[:args.sample_count]
    phenotype_matrix, sex_matrix, age_matrix, site_matrix, num_nodes, labels_class_csv = \
        gd.get_phenotype_matrix(labels_age, labels_site, labels_sex, labels_class_csv)

    # NOTE: preserve your behavior of using a fixed label list, not CSV labels
    label_list = [0] * args.n_class0 + [1] * args.n_class1
    labels_class_t = torch.tensor(label_list)  # used for splitting and masks

    # dataset
    dataset = GNNDataset(root=args.dataset_root)
    dataset = dataset[:args.sample_count]
    node_indices = np.arange(len(dataset))

    # folds
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    folds = []
    for train_index, test_index in skf.split(node_indices, labels_class_t):
        folds.append((train_index, test_index))

    fold_start = args.fold_start
    fold_end = args.fold_end if args.fold_end is not None else args.num_folds
    fold_range = range(fold_start, fold_end)

    # write header if file does not exist
    if not os.path.exists(args.out_csv):
        with open(args.out_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            # columns: fold, subject_index, then 116 weights
            writer.writerow(["fold", "subject_index"] + [f"w_{j}" for j in range(args.in_channels)])

    # iterate folds
    for num_cross in fold_range:
        # load pretrained GNNAEP (pick file by fold index)
        pre_model_path = os.path.join(args.pre_model_dir, pre_model_files[min(num_cross, len(pre_model_files)-1)])
        pre_model = GNNAEP(in_channels=args.in_channels, hidden_size=args.hidden_size, heads=args.heads)
        pre_model.load_state_dict(torch.load(pre_model_path, map_location=dev))
        pre_model = pre_model.to(dev)

        # load population GCN (pick file by fold index)
        node_model_path = os.path.join(args.node_model_dir, node_model_files[min(num_cross, len(node_model_files)-1)])
        model_local = Gcn.GCNTransformer_ZJ(args.num_features, args.gcn_hidden, args.gcn_heads,
                                            args.gcn_classes, args.gcn_dropout)
        model_local.load_state_dict(torch.load(node_model_path, map_location=dev))
        model_local = model_local.to(dev)

        # expose globals for evaluate()
        global model
        model = model_local
        global labels_class
        labels_class = labels_class_t.to(dev)
        global criclass
        criclass = torch.nn.CrossEntropyLoss().to(dev)

        # build all-loader
        all_subset = torch.utils.data.Subset(dataset, node_indices)
        all_loader = DataLoader(all_subset, len(all_subset), shuffle=False)

        # get population features from pre_model
        with torch.no_grad():
            for data2 in all_loader:
                data2 = data2.to(dev)
                x_origion, x_pred, x_middle, x_pred_MLP = pre_model(data2)
                population_graph_feature = x_middle

        population_graph_feature = population_graph_feature.cpu().detach().numpy()

        # similarity & adjacency
        similarity_matrix = gd.get_similarity(population_graph_feature)
        adj_matrix = (similarity_matrix * phenotype_matrix).astype(np.float32)
        adj_tensor = torch.tensor(adj_matrix)
        edge_index = torch.transpose(adj_tensor.nonzero(), 0, 1)
        edge_attr = adj_tensor[adj_tensor.nonzero()[:, 0], adj_tensor.nonzero()[:, 1]].unsqueeze(1)

        # build population graph Data
        nodes_feature = torch.tensor(population_graph_feature)
        data_pg = Data(x=nodes_feature, y=labels_class, edge_index=edge_index, edge_attr=edge_attr)
        data_pg = data_pg.to(dev)

        # masks
        train_set, test_set = folds[num_cross]
        train_set_t = torch.from_numpy(train_set).to(dev)
        val_set_t = torch.from_numpy(test_set).to(dev)

        train_mask, val_mask, test_mask = Gcn.get_train_test_masks(labels_class, train_set, test_set, test_set)
        train_mask = torch.from_numpy(train_mask).to(dev)
        val_mask = torch.from_numpy(val_mask).to(dev)

        # counts relative to split index (keep your logic comparing against 143)
        split_idx = args.n_class0
        v_count_greater_448 = len([x for x in val_set_t.cpu().numpy() if x >= split_idx])
        v_count_less_448 = len([x for x in val_set_t.cpu().numpy() if x < split_idx])

        # evaluate baseline on validation set
        val_feature_m, val_y, val_out = evaluate(data_pg, val_mask, val_set_t, v_count_less_448, v_count_greater_448)
        val_out_label = torch.argmax(val_out, dim=1)

        # main masking loop (logic preserved)
        weight = torch.zeros([len(val_set_t), args.in_channels], device="cpu", dtype=torch.float32)

        # mask each node (j)
        for j in range(args.in_channels):
            with torch.no_grad():
                for data2 in all_loader:
                    data10 = data2.to(dev)
                    masked_data_list = copy.deepcopy(data10)
                    # set node j to zero for every subject
                    for l in range(len(masked_data_list)):
                        masked_data_list[l].x[j] = 0
                    x_origion, x_pred, x_middle, x_pred_MLP = pre_model(masked_data_list)
                    masked_population_graph_feature = x_middle

            masked_population_graph_feature = masked_population_graph_feature.cpu().detach().numpy()

            # rebuild graph with masked features
            masked_similarity_matrix = gd.get_similarity(masked_population_graph_feature)
            masked_adj_matrix = (masked_similarity_matrix * phenotype_matrix).astype(np.float32)
            masked_adj_tensor = torch.tensor(masked_adj_matrix)
            masked_edge_index = torch.transpose(masked_adj_tensor.nonzero(), 0, 1)
            masked_edge_attr = masked_adj_tensor[masked_adj_tensor.nonzero()[:, 0],
                                                 masked_adj_tensor.nonzero()[:, 1]].unsqueeze(1)

            masked_nodes_feature = torch.tensor(masked_population_graph_feature)
            masked_data_pg = Data(x=masked_nodes_feature, y=labels_class,
                                  edge_index=masked_edge_index, edge_attr=masked_edge_attr).to(dev)

            masked_val_feature_m, val_y_m, val_out_m = evaluate(
                masked_data_pg, val_mask, val_set_t, v_count_less_448, v_count_greater_448
            )

            # cosine distance between baseline and masked embeddings for each val subject
            base_feat_cpu = val_feature_m.cpu()
            masked_feat_cpu = masked_val_feature_m.cpu()

            for i_idx in range(len(val_set_t)):
                subj_idx = val_set_t[i_idx]
                weight[i_idx, j] = distance.cosine(
                    base_feat_cpu[subj_idx, :], masked_feat_cpu[subj_idx, :]
                )

        # row-wise L2 normalization (preserved behavior)
        weight_np = weight.numpy()
        for i_idx in range(weight_np.shape[0]):
            norm = np.linalg.norm(weight_np[i_idx])
            if norm > 0:
                weight_np[i_idx] = weight_np[i_idx] / norm

        # write rows where prediction equals label
        val_set_cpu = val_set_t.cpu().numpy()
        val_y_cpu = val_y.cpu().numpy()
        val_out_label_cpu = val_out_label
