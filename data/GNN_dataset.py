# -*- coding: utf-8 -*-

import os
import os.path as osp
from abc import ABC
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, DataLoader


def edge_judge(img, threshold):
    """
    Build edges by absolute-value thresholding on an FC matrix.
    Returns (edge_index [2,E], edge_attr [E,1]) tensors.
    """
    edge_index_0 = []
    edge_attr_0 = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.abs(img[i, j]) > threshold:
                edge_index_0.append([i, j])
                edge_attr_0.append(img[i, j])
    edge_index_0 = np.array(edge_index_0)
    edge_attr_0 = np.array(edge_attr_0).reshape(-1, 1)
    return torch.tensor(edge_index_0, dtype=torch.long), torch.tensor(edge_attr_0, dtype=torch.float32)


class GNNDataset(Dataset, ABC):
    """
    Cleaned version of the original dataset:
    - No hard-coded paths; pass paths via __init__.
    - Saves per-subject tensors as fMRI_data{i}.pt (unchanged behavior).
    - Threshold remains configurable (default 0.5).
    """

    def __init__(
        self,
        data_root: str,
        label_path: str,
        processed_dir: str,
        threshold: float = 0.2,
        id_col: str = "SUB_ID",
        label_col: str = "labels_class",
        transform=None,
        pre_transform=None,
    ):
        self.data_root = osp.abspath(data_root)
        self.label_path = osp.abspath(label_path)
        self._threshold = float(threshold)
        self.id_col = id_col
        self.label_col = label_col

        # read once to know dataset length
        df = pd.read_csv(self.label_path, usecols=[self.id_col, self.label_col])
        self._N = len(df)

        super().__init__(processed_dir, transform, pre_transform)  # root=processed_dir

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        # keep the original naming scheme
        return [f"fMRI_data{i}.pt" for i in range(self._N)]

    def download(self):
        pass

    def process(self):
        # keep original behavior: skip if processed files already exist
        existing = [f for f in os.listdir(self.processed_dir) if f.endswith(".pt")]
        if len(existing) != 0:
            print("Processed files already exist. Delete the output directory to rebuild.")
            return

        data_root = self.data_root
        label_path = self.label_path

        data_path_list = sorted(os.listdir(data_root))
        label_ids = pd.read_csv(label_path, usecols=[self.id_col]).values.reshape(-1)
        label_list = pd.read_csv(label_path, usecols=[self.label_col]).values

        for index in range(len(label_list)):
            # keep the original check: same order match between IDs and filenames
            if str(label_ids[index]) not in data_path_list[index]:
                raise Exception("FC files do not match labels (order-based check failed).")

            fc_path = osp.join(data_root, data_path_list[index])
            node_feature = np.loadtxt(fc_path)
            edge_attr = np.loadtxt(fc_path)

            node_feature = torch.tensor(node_feature, dtype=torch.float32)

            edge_index, edge_attr = edge_judge(edge_attr, self._threshold)

            y = torch.tensor(label_list[index], dtype=torch.long)

            fMRI_data = Data(
                x=node_feature,
                edge_index=edge_index.t().contiguous(),
                edge_attr=edge_attr,
                y=y,
            )

            torch.save(
                fMRI_data,
                osp.join(self.processed_dir, f"fMRI_data{index}.pt"),
            )

        print("Saved processed dataset to:", self.processed_dir)

    def len(self):
        df = pd.read_csv(self.label_path, usecols=[self.label_col])
        return len(df)

    def get(self, idx):
        return torch.load(osp.join(self.processed_dir, f"fMRI_data{idx}.pt"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build and sanity-check the GNNDataset.")
    parser.add_argument("--data-root", required=True, help="Directory with per-subject FC files.")
    parser.add_argument("--label-path", required=True, help="CSV with at least SUB_ID and labels_class columns.")
    parser.add_argument("--out-dir", required=True, help="Output directory to store processed .pt files.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Absolute threshold for edges (default 0.5).")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    ds = GNNDataset(
        data_root=args.data_root,
        label_path=args.label_path,
        processed_dir=args.out_dir,
        threshold=args.threshold,
    )
    dl = DataLoader(ds, num_workers=0, batch_size=args.batch_size, shuffle=False)

    print("Number of samples:", len(dl.dataset))
    for data in dl:
        dev = torch.device("cpu")
        data = data.to(dev)
        _ = torch.unique(data.edge_index, dim=1)
        break
    print("Sanity check passed.")
