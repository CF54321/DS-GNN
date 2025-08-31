# -*- coding: utf-8 -*-
"""
Cleaned training script with configurable paths (no logic changes).
"""

import os
import sys
import time
import argparse
import inspect
import math
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data
from torch_geometric.data import DataLoader

# Optional: add your project root to sys.path for imports like tools/model/data
# (Keeps your original package layout working without hard-coded absolute paths)
def maybe_append_project_root(project_root: str | None):
    if project_root and project_root not in sys.path:
        sys.path.append(project_root)

# -----------------------
# Parse CLI first (so we can set sys.path), then import project modules
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="DS-GNN training (paths made configurable).")

    # Replaces hard-coded paths
    parser.add_argument("--project-root", type=str, default=None,
                        help="Optional path to append to sys.path so that tools/model/data imports work.")
    parser.add_argument("--root", type=str, required=True,
                        help="Dataset root for GNNDataset (same as your original GNNDataset(root=...)).")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory to save log files.")
    parser.add_argument("--tb-dir", type=str, default="./tensorboard",
                        help="Directory for TensorBoard event files.")
    parser.add_argument("--model-dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--metrics-csv", type=str, default="./ACC.csv",
                        help="Path to write aggregated metrics CSV.")

    # CV & training setup
    parser.add_argument("--outer-loops", type=int, default=6, help="Number of outer repeats (range(outer_loops)).")
    parser.add_argument("--num-folds", type=int, default=10, help="Number of stratified folds.")
    parser.add_argument("--num-cross", type=int, default=None,
                        help="If set, run only this fold index (0..num-folds-1). If None, iterate all.")
    parser.add_argument("--seed", type=int, default=820, help="Base random seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g., cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epochs. If None, uses (50 + 50 * outer_index) to match your original loop.")

    # Model hyperparameters (kept as your defaults/types)
    parser.add_argument("--model-path", type=str, default=None, help="Optional pre-trained model path.")
    parser.add_argument("--in-channels", type=int, default=116, help="Model input channels.")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden size.")
    parser.add_argument("--heads", type=int, default=1, help="Number of attention heads.")

    # Optimizer/scheduler (kept identical to your choices/IDs)
    parser.add_argument("--lr", type=float, default=0.0007, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0007, help="Weight decay.")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="Optimizer type.")
    parser.add_argument("--scheduler", type=int, default=3,
                        help="LR scheduler ID (kept same semantics as your original script).")

    # Optional run tag
    parser.add_argument("--tag", type=str, default="ASD", help="Run tag used in log/model filenames.")

    return parser.parse_args()

args = parse_args()
maybe_append_project_root(args.project_root)

# Now import your project modules (unchanged)
from init_weights import init_weights
from data.GNN_dataset import GNNDataset
from model.Models_LJX_disease import GNNAEP
from tools.utils import confusion_matrix, set_seed, weight_to_matrix
from tensorboardX import SummaryWriter
from tools.logger import Logger as log

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
import pandas as pd
import csv

# -----------------------
# Trainer & Val (unchanged logic)
# -----------------------
class Trainer(object):
    def __init__(self, model: torch.nn.Module, optimizer, scheduler, criterionpre, criterionclass, prediff, device: torch.device):
        super().__init__()
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterionpre = criterionpre.to(device)
        self.criterionclass = criterionclass.to(device)
        self.prediff = prediff.to(device)
        self.device = device

    def train(self, data_loader, e):
        self.model.train()
        loss_all = 0
        premaes_all = 0
        premaef_all = 0
        regularization_loss = 0
        predalls = torch.tensor([])
        rawalls = torch.tensor([])
        predallf = torch.tensor([])
        rawallf = torch.tensor([])
        TP, TN, FP, FN = 0, 0, 0, 0
        batchnum = 0
        for data1 in data_loader:
            y = data1.y.to(self.device)
            data1 = data1.to(self.device)

            x_origionf, predf, feature, pred_MLP = self.model(data1)

            TPb, TNb, FPb, FNb = confusion_matrix(torch.argmax(pred_MLP, dim=1), y)
            TP += TPb
            TN += TNb
            FP += FPb
            FN += FNb

            loss_pref = self.criterionpre(predf.float(), x_origionf.float())
            loss_premlp = self.criterionclass(pred_MLP, y)
            for param in self.model.parameters():
                regularization_loss += torch.sum(torch.abs(param))

            loss = loss_pref + loss_premlp + 0.0004 * regularization_loss.item()

            premaef = self.prediff(predf.float(), x_origionf.float())
            self.optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            self.optimizer.step()
            loss_all += loss.item()

            premaef_all += premaef.item()

            predallf = torch.cat((predallf.to(self.device), predf.float().to(self.device)), 0)
            rawallf = torch.cat((rawallf.to(self.device), x_origionf.float().to(self.device)), 0)
            batchnum += 1
            # NOTE: preserved as-is (original checks equality against 3)
            if self.scheduler == 3:
                if e < 600:
                    self.scheduler[0].step()
                else:
                    self.scheduler[1].step()

        pccmaes = []
        pccmaef = []

        acc = (TP + TN) / (TP + TN + FP + FN)
        SEN = TP / (TP + FN)
        SPE = TN / (TN + FP)
        F1_score = 2 * TP / (2 * TP + FN + FP)

        return TP, TN, FP, FN, loss_all / batchnum, premaes_all / batchnum, pccmaes, premaef_all / batchnum, pccmaef, acc, SEN, SPE, F1_score

class Val(object):
    def __init__(self, model: torch.nn.Module, device: torch.device, criterionpre, criterionclass, prediff):
        super().__init__()
        self.model = model.to(device)
        self.criterionpre = criterionpre.to(device)
        self.criterionclass = criterionclass.to(device)
        self.prediff = prediff.to(device)
        self.device = device

    def evaluate(self, data_loader):
        self.model.eval()
        loss_all = 0
        premaes_all = 0
        premaef_all = 0
        regularization_loss = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        predalls = torch.tensor([])
        predallf = torch.tensor([])
        rawalls = torch.tensor([])
        rawallf = torch.tensor([])
        with torch.no_grad():
            for data1 in data_loader:
                y = data1.y.to(self.device)
                data1 = data1.to(self.device)

                x_origionf, predf, feature, pred_MLP = self.model(data1)

                TPb, TNb, FPb, FNb = confusion_matrix(torch.argmax(pred_MLP, dim=1), y)
                TP += TPb
                TN += TNb
                FP += FPb
                FN += FNb

                loss_pref = self.criterionpre(predf.float(), x_origionf.float())
                loss_premlp = self.criterionclass(pred_MLP, y)
                for param in self.model.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
                loss = loss_pref + loss_premlp + 0.0001 * regularization_loss.item()

                premaef = self.prediff(predf.float(), x_origionf.float())
                loss_all += loss.item()

                premaef_all += premaef.item()

                predallf = torch.cat((predallf.to(self.device), predf.float().to(self.device)), 0)
                rawallf = torch.cat((rawallf.to(self.device), x_origionf.float().to(self.device)), 0)

            pccmaes = []
            pccmaef = []

            acc = (TP + TN) / (TP + TN + FP + FN)
            SEN = TP / (TP + FN)
            SPE = TN / (TN + FP)
            F1_score = 2 * TP / (2 * TP + FN + FP)

        return TP, TN, FP, FN, loss_all, premaes_all, pccmaes, premaef_all, pccmaef, acc, SEN, SPE, F1_score

# -----------------------
# Training wrapper (paths now configurable)
# -----------------------
def train(args, run_time_str: str, fold_index: int):
    # set seed/device
    set_seed(args.seed)
    dev = torch.device(args.device)

    # logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_name = f"{args.tag}-{run_time_str}.log"
    log.initialize(os.path.join(args.log_dir, log_name))
    log.i("python " + " ".join(sys.argv))
    log.i("Arguments:")
    for k, v in vars(args).items():
        log.i(f"\t{k} = {v}")
    log.i(f"Device: {args.device}")

    # model
    m = GNNAEP(in_channels=args.in_channels, hidden_size=args.hidden_size, heads=args.heads)
    init_weights(m)
    if args.model_path and os.path.isfile(args.model_path):
        m.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        log.i(f"Loaded pre-trained weights from {args.model_path}")

    log.i("Model:")
    log.i(f"\n{m}")
    log.i(f"\n{inspect.getsource(m.__init__)}")
    log.i(f"\n{inspect.getsource(m.forward)}")

    # data
    dataset = GNNDataset(root=args.root)

    # NOTE: your original script hard-coded label_list lengths; preserved here.
    label_list = [0] * 526 + [1] * 495
    node_indices = np.arange(len(dataset))

    # CV folds
    num_folds = args.num_folds
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=args.seed)
    folds = []
    for train_index, test_index in skf.split(node_indices, label_list):
        folds.append((train_index, test_index))

    test_set = folds[fold_index][1]
    train_set = folds[fold_index][0]

    training_set = Subset(dataset, train_set)
    validation_set = Subset(dataset, test_set)

    train_loader = DataLoader(training_set, args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(validation_set, len(validation_set), shuffle=False)

    log.i("Dataset:")
    log.i(f"\tFound {len(training_set)} training samples")
    log.i(f"\tFound {len(validation_set)} validation samples")
    log.i(f"\nFold indices (val): {test_set}")

    # optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler (kept original ID semantics)
    if args.scheduler == 0:
        scheduler = None
    elif args.scheduler == 1:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.00001, steps_per_epoch=len(train_loader),
                                            epochs=args.epochs)
    elif args.scheduler == 2:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    elif args.scheduler == 3:
        scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.000001)
        scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.000001)
        scheduler = [scheduler1, scheduler2]
    else:
        scheduler = None

    # losses (unchanged)
    cripre = torch.nn.MSELoss()
    criclass = torch.nn.CrossEntropyLoss()
    pmae = nn.L1Loss()

    log.i("optimizer:")
    log.i(f"\n{optimizer}")
    log.i("scheduler:")
    log.i(f"\tscheduler={args.scheduler}")
    log.i("criterionpre:")
    log.i(f"\t{cripre}")
    log.i("criterionclass:")
    log.i(f"\t{criclass}")
    log.i("MAE:")
    log.i(f"\t{pmae}")

    # tensorboard
    tb_dir = os.path.join(args.tb_dir, f"{args.tag}-{run_time_str}")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    # checkpoints
    model_dir = os.path.join(args.model_dir, args.tag, run_time_str)
    os.makedirs(model_dir, exist_ok=True)

    # train loop (unchanged core)
    trainer = Trainer(model=m, optimizer=optimizer, scheduler=scheduler,
                      criterionpre=cripre, criterionclass=criclass, prediff=pmae, device=dev)
    val = Val(model=m, criterionpre=cripre, criterionclass=criclass, prediff=pmae, device=dev)
    log.i("Start training:")

    max_acc0 = 0.3
    m_val_SEN0 = 0
    m_val_SPE0 = 0

    for epoch in range(args.epochs):
        log.i(f"epoch: {epoch}")
        train_TP, train_TN, train_FP, train_FN, train_loss, train_MAEs, train_pccs, train_MAEf, train_pccf, train_acc0, train_SEN0, train_SPE0, train_F1 = \
            trainer.train(train_loader, e=epoch)
        val_TP, val_TN, val_FP, val_FN, val_loss, val_MAEs, val_pccs, val_MAEf, val_pccf, val_acc0, val_SEN0, val_SPE0, val_F1 = \
            val.evaluate(val_loader)

        writer.add_scalars('loss', {'Train Loss': train_loss, 'Val loss': val_loss}, epoch)
        writer.add_scalars('premaes', {'Train MAE': train_MAEs, 'Val MAE': val_MAEs}, epoch)
        writer.add_scalars('premaef', {'Train MAE': train_MAEf, 'Val MAE': val_MAEf}, epoch)

        log.i("\tTrain TP:{}\tTN:{}\tFP:{}\tFN:{}\tLoss:{:.6f}\tMAEs:{}\tpccs:{}\tMAEf:{}\tpccf:{}\tAcc:{:.4f}\tSEN:{:.4f}\tSPE:{:.4f}\tF1:{:.4f}".format(
            train_TP, train_TN, train_FP, train_FN, train_loss, train_MAEs, train_pccs, train_MAEf, train_pccf, train_acc0, train_SEN0, train_SPE0, train_F1))
        log.i("\tVal   TP:{}\tTN:{}\tFP:{}\tFN:{}\tLoss:{:.6f}\tMAEs:{}\tpccs:{}\tMAEf:{}\tpccf:{}\tAcc:{:.4f}\tSEN:{:.4f}\tSPE:{:.4f}\tF1:{:.4f}".format(
            val_TP, val_TN, val_FP, val_FN, val_loss, val_MAEs, val_pccs, val_MAEf, val_pccf, val_acc0, val_SEN0, val_SPE0, val_F1))

        if val_acc0 > max_acc0:
            ckpt_path = os.path.join(model_dir, f"{args.tag}-{args.num_cross if args.num_cross is not None else fold_index}.pth")
            torch.save(m.state_dict(), ckpt_path)
            log.i(f"Model saved: {ckpt_path}")
            max_acc0 = val_acc0
            m_val_SEN0 = val_SEN0
            m_val_SPE0 = val_SPE0

    writer.close()
    return max_acc0, m_val_SEN0, m_val_SPE0


def main():
    # Outer repeats
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_csv)), exist_ok=True)
    # Write header once
    if not os.path.exists(args.metrics_csv):
        with open(args.metrics_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seed_or_fold', 'acc', 'SEN', 'SPE', 'time'])

    for outer_idx in range(args.outer_loops):
        # Each outer loop writes its fold results + an average row
        with open(args.metrics_csv, mode='a', newline='') as f:
            writer = csv.writer(f)

            acc_sum = 0.0
            sen_sum = 0.0
            spe_sum = 0.0

            # Either a single fold or all folds
            fold_range = [args.num_cross] if args.num_cross is not None else list(range(args.num_folds))
            for fold in fold_range:
                nowtime = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))

                # epochs: use CLI override or your original pattern (50 + 50 * outer_idx)
                epochs = args.epochs if args.epochs is not None else (50 + 50 * outer_idx)

                # pack args into a simple namespace clone with possible per-fold overrides
                run_args = argparse.Namespace(**vars(args))
                run_args.time = nowtime
                run_args.epochs = epochs
                run_args.num_cross = fold

                max_acc0, m_val_SEN0, m_val_SPE0 = train(run_args, nowtime, fold)
                writer.writerow([fold, max_acc0, m_val_SEN0, m_val_SPE0, nowtime])

                acc_sum += max_acc0
                sen_sum += m_val_SEN0
                spe_sum += m_val_SPE0

            # average over folds (if multiple)
            n = len(fold_range)
            writer.writerow([args.seed, acc_sum / n, sen_sum / n, spe_sum / n, "avg_over_folds"])

if __name__ == "__main__":
    main()
