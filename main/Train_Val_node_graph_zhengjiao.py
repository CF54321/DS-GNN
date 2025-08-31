# -*- coding: utf-8 -*-
"""
Cleaned population-graph training script:
- No hard-coded absolute paths; everything is configurable via CLI.
- Keeps your function/model logic intact.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

import sys
import time
import argparse
import inspect
import numpy as np
import torch
import torch.nn as nn

from torch.optim import lr_scheduler
from torch_geometric.data import Data, DataLoader
from tensorboardX import SummaryWriter

# -----------------------
# CLI (we parse first so we can set sys.path if needed)
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Population-graph training (paths made configurable).")

    # Optional: append your project root to sys.path so local imports work
    p.add_argument("--project-root", type=str, default=None,
                   help="Optional path to append to sys.path (so tools/model/data imports work).")

    # I/O directories and files (replace hard-coded paths)
    p.add_argument("--dataset-root", type=str, required=True,
                   help="GNNDataset root for individual graphs (was args.root).")
    p.add_argument("--phenotype-csv", type=str, required=True,
                   help="CSV with SUB_ID/AGE/SITE_ID/SEX/label (was phenotype_path).")
    p.add_argument("--pre-model-dir", type=str, required=True,
                   help="Directory containing pre-trained GNNAEP weight files (sorted; pick by --num-cross).")

    p.add_argument("--log-dir", type=str, default="./logs",
                   help="Directory to save log files.")
    p.add_argument("--tb-dir", type=str, default="./tensorboard",
                   help="Directory for TensorBoard runs.")
    p.add_argument("--model-dir", type=str, default="./checkpoints",
                   help="Directory to save model checkpoints.")
    p.add_argument("--metrics-csv", type=str, default="./ACC.csv",
                   help="Where to append per-fold metrics (acc/SEN/SPE/F1/AUC).")

    # Run config
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=856)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-folds", type=int, default=10)
    p.add_argument("--num-cross", type=int, required=True,
                   help="Fold index (0..num-folds-1).")
    p.add_argument("--tag", type=str, default="SCHZ_COBRE",
                   help="Run tag used in filenames/logs.")

    # Optimizer/scheduler (same semantics as your original)
    p.add_argument("--lr", type=float, default=0.0007)
    p.add_argument("--weight-decay", type=float, default=0.0007)
    p.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    p.add_argument("--scheduler", type=int, default=3,
                   help="0=None, 1=OneCycle, 2=StepLR, 3=Cosine (two phases)")

    # Pre-model (GNNAEP) hyperparams — unchanged defaults
    p.add_argument("--pre-in-channels", type=int, default=116)
    p.add_argument("--pre-hidden-size", type=int, default=32)
    p.add_argument("--pre-heads", type=int, default=1)

    # Population GCN hyperparams — unchanged defaults
    p.add_argument("--num-features", type=int, default=3712)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    # Loss/graph params — unchanged defaults
    p.add_argument("--para", type=float, default=0.0,
                   help="Orthogonality loss coefficient (your args.para).")
    p.add_argument("--h", type=float, default=0.0,
                   help="Adjacency threshold (your args.h).")

    # Phenotype slice / sample count (you had [:146])
    p.add_argument("--sample-count", type=int, default=146,
                   help="Use the first N subjects from phenotype/dataset (default 146).")

    return p.parse_args()

args = parse_args()
if args.project_root:
    if args.project_root not in sys.path:
        sys.path.append(args.project_root)

# -----------------------
# Local project imports (unchanged)
# -----------------------
from init_weights import init_weights
from data.GNN_dataset0 import GNNDataset
from model.Models_LJX_disease import GNNAEP
from tools.utils import confusion_matrix, set_seed, weight_to_matrix
from tools.logger import Logger as log
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
import csv
from sklearn.metrics import roc_auc_score
import data.population_graph_dataset as gd
import model.Model_node as Gcn  # expects GCNTransformer_ZJ, get_train_test_masks

# -----------------------
# Optional focal loss (kept as in your original; not used in this script)
# -----------------------
class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, preds, labels):
        eps = 1e-7
        loss_1 = -self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        loss_0 = -(1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        return torch.mean(loss_0 + loss_1)

# -----------------------
# Training function (logic preserved; paths/configs from args)
# -----------------------
def train(args):
    nowtag = f"{args.tag}-{args.time}"

    # logging setup
    os.makedirs(args.log_dir, exist_ok=True)
    log_name = f"{nowtag}.log"
    log.initialize(os.path.join(args.log_dir, log_name))
    log.i("python " + " ".join(sys.argv))
    log.i("Arguments:")
    for k, v in vars(args).items():
        log.i(f"\t{k} = {v}")

    set_seed(args.seed)
    dev = torch.device(args.device)
    log.i(f"Device: {args.device}")

    # tensorboard
    tb_root = os.path.join(args.tb_dir, f"log-{nowtag}")
    os.makedirs(tb_root, exist_ok=True)
    writer = SummaryWriter(tb_root)

    # pre-trained individual-graph model (GNNAEP)
    pre_model = GNNAEP(in_channels=args.pre_in_channels, hidden_size=args.pre_hidden_size, heads=args.pre_heads)
    pre_model = pre_model.to(dev)

    # choose a weight file by sorted order and --num-cross
    weight_files = sorted([f for f in os.listdir(args.pre_model_dir) if f.endswith(".pth") or f.endswith(".pt")])
    if len(weight_files) == 0:
        raise FileNotFoundError(f"No weight files found in {args.pre_model_dir}")
    idx = max(0, min(args.num_cross, len(weight_files) - 1))
    pre_weight_path = os.path.join(args.pre_model_dir, weight_files[idx])
    log.i(f"Loading pre_model weights: {pre_weight_path}")
    pre_model.load_state_dict(torch.load(pre_weight_path, map_location=dev))

    # population-graph model
    model = Gcn.GCNTransformer_ZJ(args.num_features, args.hidden, args.heads, 2, args.dropout)
    model = model.to(dev)
    log.i("pre_model:")
    log.i(f"\n{pre_model}")
    log.i(f"\n{inspect.getsource(pre_model.__init__)}")
    log.i(f"\n{inspect.getsource(pre_model.forward)}")
    log.i("population model:")
    log.i(f"\n{model}")
    log.i(f"\n{inspect.getsource(model.__init__)}")
    log.i(f"\n{inspect.getsource(model.forward)}")

    # phenotype & labels
    pheno = pd.read_csv(args.phenotype_csv)
    label_ID = pheno["SUB_ID"].values.reshape(-1)[:args.sample_count]
    labels_age = pheno["AGE"].values[:args.sample_count]
    labels_site = pheno["SITE_ID"].values[:args.sample_count]
    labels_sex = pheno["SEX"].values[:args.sample_count]
    labels_class = pheno["label"].values[:args.sample_count]
    # your original code zeroed SITE_ID
    labels_site = np.zeros_like(labels_site)
    label_list = labels_class

    phenotype_matrix, sex_matrix, age_matrix, site_matrix, num_nodes = gd.get_phenotype_matrix(
        labels_age, labels_site, labels_sex
    )

    # individual-graph dataset
    dataset = GNNDataset(root=args.dataset_root)
    dataset = dataset[:args.sample_count]
    node_indices = np.arange(len(dataset))  # 0..N-1

    # build folds
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    folds = [(tr, te) for (tr, te) in skf.split(node_indices, label_list)]
    train_idx, test_idx = folds[args.num_cross]
    all_subset = Subset(dataset, node_indices)
    all_loader = DataLoader(all_subset, len(all_subset), shuffle=False)

    # masks/tensors
    labels_class_t = torch.tensor(labels_class)
    train_mask, val_mask, test_mask = Gcn.get_train_test_masks(labels_class_t, train_idx, test_idx, test_idx)
    train_mask = torch.from_numpy(train_mask).to(dev)
    val_mask = torch.from_numpy(val_mask).to(dev)
    train_set = torch.from_numpy(train_idx).to(dev)
    val_set = torch.from_numpy(test_idx).to(dev)

    log.i("dataset:")
    log.i(f"\tFound {len(train_idx)} training samples")
    log.i(f"\tFound {len(test_idx)} validation samples")
    log.i(f"\nFold indices (val): {test_idx}")

    # optimizer/scheduler (kept semantics)
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 0:
        scheduler = None
    elif args.scheduler == 1:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, steps_per_epoch=max(1, len(train_idx)//args.batch_size), epochs=args.epochs)
    elif args.scheduler == 2:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    elif args.scheduler == 3:
        scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
        scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        scheduler = [scheduler1, scheduler2]
    else:
        scheduler = None

    criclass = nn.CrossEntropyLoss().to(dev)

    log.i("optimizer:")
    log.i(f"\n{optimizer}")
    log.i("scheduler:")
    log.i(f"\tscheduler={args.scheduler}")
    log.i("criterionclass:")
    log.i(f"\t{criclass}")

    # checkpoints
    model_out_dir = os.path.join(args.model_dir, nowtag)
    os.makedirs(model_out_dir, exist_ok=True)

    # training loop (logic preserved)
    max_acc0 = 0.3
    val_SEN0 = 0.0
    val_SPE0 = 0.0
    best_f1 = 0.0
    best_auc = 0.0

    for epoch in range(args.epochs):
        log.i(f"epoch: {epoch}")
        pre_model = pre_model.to(dev).eval()  # you used no grad for pre_model; keep eval
        model = model.to(dev).train()

        optimizer.zero_grad()

        # build population graph features from pre_model
        with torch.no_grad():
            for data2 in all_loader:
                data2 = data2.to(dev)
                x_origion, x_pred, x_middle, x_pred_MLP = pre_model(data2)
                population_graph_feature = x_middle  # (N, F)

        pop_feat = population_graph_feature.detach().cpu().numpy()
        sim = gd.get_similarity(pop_feat)
        adj = sim * phenotype_matrix
        adj[adj < args.h] = 0
        adj = adj.astype(np.float32)
        adj_t = torch.tensor(adj)
        edge_index = torch.transpose(adj_t.nonzero(), 0, 1)
        edge_attr = adj_t[adj_t.nonzero()[:, 0], adj_t.nonzero()[:, 1]].unsqueeze(1)

        # build Data for the population graph
        labels_class_t = torch.tensor(labels_class)
        nodes_feature = torch.tensor(pop_feat)
        data_pg = Data(x=nodes_feature, y=labels_class_t, edge_index=edge_index, edge_attr=edge_attr)
        data_pg = data_pg.to(dev)
        labels_flat = torch.squeeze(labels_class_t.to(dev))

        # mini-batches over indices (kept as your original loop)
        train_indices = np.arange(len(train_idx))
        train_loader_idx = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(train_indices)),
            batch_size=args.batch_size, shuffle=True, drop_last=True
        )

        total_loss = 0.0
        batchnum = 0

        for (batch_index,) in train_loader_idx:
            out, feature_m = model(data_pg)
            train_out = out[train_mask]
            train_y = labels_flat[train_mask]

            # orthogonality penalty (unchanged)
            indices_0 = torch.where(train_y == 0)[0]
            indices_1 = torch.where(train_y == 1)[0]
            negative_features = feature_m[indices_0]
            positive_features = feature_m[indices_1]
            inner_product = torch.mm(positive_features, negative_features.t())
            inner_product_square = torch.pow(inner_product, 2)
            sum_of_elements = torch.sum(inner_product_square)

            # classification loss (you used sigmoid + CE on logits; kept as-is)
            loss0 = criclass(torch.sigmoid(train_out[batch_index]), train_y[batch_index])
            loss1 = args.para * (sum_of_elements / len(data_pg.y))
            loss = 10 * (loss0 + loss1)

            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batchnum += 1

            # scheduler stepping (kept as your condition)
            if scheduler == 3:
                if epoch < 600:
                    scheduler[0].step()
                else:
                    scheduler[1].step()

            # train metrics (unchanged)
            TP, TN, FP, FN = 0, 0, 0, 0
            TPb, TNb, FPb, FNb = confusion_matrix(torch.argmax(train_out, dim=1), train_y)
            TP += TPb; TN += TNb; FP += FPb; FN += FNb

        train_acc = (TP + TN) / (TP + TN + FP + FN)
        train_SEN = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        train_SPE = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        train_F1 = 2 * TP / (2 * TP + FN + FP) if (2 * TP + FN + FP) > 0 else 0.0
        train_loss = total_loss / max(1, batchnum)

        # validation (unchanged)
        with torch.no_grad():
            pre_model.eval()
            model.eval()
            out, val_feature = model(data_pg)
            val_out = out[val_mask]
            val_y = labels_flat[val_mask]

            indices_0 = torch.where(val_y == 0)[0]
            indices_1 = torch.where(val_y == 1)[0]
            negative_features = val_feature[indices_0]
            positive_features = val_feature[indices_1]
            inner_product = torch.mm(positive_features, negative_features.t())
            inner_product_square = torch.pow(inner_product, 2)
            v_sum_of_elements = torch.sum(inner_product_square)

            loss2 = criclass(torch.sigmoid(val_out), val_y)
            loss3 = args.para * (v_sum_of_elements / len(val_set))
            val_loss = 10 * (loss2 + loss3)

            TP, TN, FP, FN = 0, 0, 0, 0
            TPb, TNb, FPb, FNb = confusion_matrix(torch.argmax(val_out, dim=1), val_y)
            TP += TPb; TN += TNb; FP += FPb; FN += FNb
            val_acc = (TP + TN) / (TP + TN + FP + FN)
            val_SEN = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            val_SPE = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            val_F1 = 2 * TP / (2 * TP + FN + FP) if (2 * TP + FN + FP) > 0 else 0.0

            # AUC (unchanged approach)
            probs = torch.softmax(val_out, dim=1)
            pos_scores = probs[:, 1]
            y_true = val_y.cpu().numpy()
            y_score = pos_scores.cpu().numpy()
            val_auc = roc_auc_score(y_true, y_score)

        writer.add_scalars("loss", {"Train": train_loss, "Val": float(val_loss)}, epoch)
        writer.add_scalars("Orthogonality_loss", {"Train": float(args.para * loss1), "Val": float(args.para * loss3)}, epoch)

        log.i(f"\tTrain loss:{train_loss:.6f}\tacc:{float(train_acc):.4f}\tSEN:{float(train_SEN):.4f}\tSPE:{float(train_SPE):.4f}\tF1:{float(train_F1):.4f}")
        log.i(f"\tVal   loss:{float(val_loss):.6f}\tacc:{float(val_acc):.4f}\tSEN:{float(val_SEN):.4f}\tSPE:{float(val_SPE):.4f}\tF1:{float(val_F1):.4f}\tAUC:{float(val_auc):.4f}")

        # save best by acc (kept)
        if val_acc > max_acc0:
            max_acc0 = float(val_acc)
            val_SEN0 = float(val_SEN)
            val_SPE0 = float(val_SPE)
            best_f1 = float(val_F1)
            best_auc = float(val_auc)
            ckpt_path = os.path.join(model_out_dir, f"latest{args.num_cross}.pth")
            torch.save(model.state_dict(), ckpt_path)
            log.i(f"Model saved: {ckpt_path}")

        torch.cuda.empty_cache()

    writer.close()
    return max_acc0, val_SEN0, val_SPE0, best_f1, best_auc


def main():
    # prepare metrics CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_csv)), exist_ok=True)
    if not os.path.exists(args.metrics_csv):
        with open(args.metrics_csv, mode="w", newline="") as f:
            csv.writer(f).writerow(["fold", "acc", "SEN", "SPE", "F1", "AUC", "time"])

    # single run for the requested fold
    args.time = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    acc, sen, spe, f1, auc = train(args)
    with open(args.metrics_csv, mode="a", newline="") as f:
        csv.writer(f).writerow([args.num_cross, acc, sen, spe, f1, auc, args.time])


if __name__ == "__main__":
    main()
