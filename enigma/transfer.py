# -*- coding: utf-8 -*-
"""
Map AAL ROI weights to DK+ASEG parcels by voxel overlap.
- Resamples AAL labels to DK+ASEG grid (nearest) to ensure same voxel space
- Builds AAL × DK+ASEG voxel-overlap matrix
- Produces DK+ASEG weights as overlap-weighted averages of AAL weights
Outputs:
  1) <OUT_PREFIX>_weights.csv
  2) <OUT_PREFIX>_overlap_heatmap.png
"""

import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt
import os


def parse_args():
    p = argparse.ArgumentParser(description="Map AAL weights to DK+ASEG parcels via voxel overlap.")
    p.add_argument("--aal-label-nii", required=True, help="Path to AAL label NIfTI (integer labels).")
    p.add_argument("--dkaseg-label-nii", required=True, help="Path to DK+ASEG label NIfTI (integer labels).")
    p.add_argument("--aal-weights-csv", required=True,
                   help="CSV with AAL weights; must contain a 'label' column with integer AAL IDs. "
                        "All other columns are treated as diseases/weights.")
    p.add_argument("--out-prefix", required=True, help="Output prefix for CSV and PNG.")
    p.add_argument("--skip-resample", action="store_true",
                   help="If set, do NOT resample AAL to DK+ASEG grid (assume same space).")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Load images ----
    aal_img = nib.load(args.aal_label_nii)
    dkaseg_img = nib.load(args.dkaseg_label_nii)

    # Resample AAL to DK+ASEG grid unless told otherwise (nearest to preserve integers)
    if args.skip_resample:
        aal_aligned = aal_img
    else:
        aal_aligned = resample_to_img(aal_img, dkaseg_img, interpolation="nearest")

    aal_data = aal_aligned.get_fdata().astype(int)
    dkaseg_data = dkaseg_img.get_fdata().astype(int)

    # Positive labels only
    aal_labels = np.unique(aal_data)
    aal_labels = aal_labels[aal_labels > 0]
    dkaseg_labels = np.unique(dkaseg_data)
    dkaseg_labels = dkaseg_labels[dkaseg_labels > 0]

    if aal_labels.size == 0 or dkaseg_labels.size == 0:
        raise ValueError("No positive labels found in AAL and/or DK+ASEG volumes.")

    # ---- Read AAL weights ----
    wdf = pd.read_csv(args.aal_weights_csv)
    if "label" not in wdf.columns:
        raise ValueError("AAL weights CSV must contain a 'label' column with integer AAL IDs.")
    disease_cols = [c for c in wdf.columns if c not in ["roi", "label"]]
    if len(disease_cols) == 0:
        raise ValueError("No disease/weight columns found; CSV should have columns besides 'label' (e.g., weight_SZ).")

    # Map {aal_label -> {disease -> weight}}
    aal_weight_map = {
        int(row["label"]): {d: float(row[d]) for d in disease_cols}
        for _, row in wdf.iterrows()
    }

    # Fast indexers
    aal_label_to_col = {lab: j for j, lab in enumerate(aal_labels)}

    # ---- Build voxel-overlap matrix (DK rows × AAL cols) ----
    overlap = np.zeros((len(dkaseg_labels), len(aal_labels)), dtype=np.int64)
    for i, dk_id in enumerate(dkaseg_labels):
        dk_mask = (dkaseg_data == dk_id)
        # Intersect with AAL labels present under this DK mask
        aal_vals, counts = np.unique(aal_data[dk_mask], return_counts=True)
        for v, cnt in zip(aal_vals, counts):
            if v <= 0:
                continue
            j = aal_label_to_col.get(int(v), None)
            if j is not None:
                overlap[i, j] = cnt

    # ---- Compute DK+ASEG weights as overlap-weighted averages of AAL weights ----
    dkaseg_weights = {d: np.zeros(len(dkaseg_labels), dtype=float) for d in disease_cols}
    voxel_sum = overlap.sum(axis=1)  # total overlapping voxels per DK parcel

    # Track missing weights (AAL labels with no weight provided)
    missing = {d: 0 for d in disease_cols}

    for i, dk_id in enumerate(dkaseg_labels):
        denom = voxel_sum[i]
        if denom == 0:
            for d in disease_cols:
                dkaseg_weights[d][i] = np.nan
            continue

        for d in disease_cols:
            num = 0.0
            for j, aal_id in enumerate(aal_labels):
                cnt = overlap[i, j]
                if cnt <= 0:
                    continue
                w = aal_weight_map.get(int(aal_id), {}).get(d, np.nan)
                if np.isfinite(w):
                    num += w * cnt
                else:
                    missing[d] += 1
            dkaseg_weights[d][i] = num / denom

    # ---- Save DK+ASEG weights table ----
    dkaseg_df = pd.DataFrame({"label": dkaseg_labels})
    for d in disease_cols:
        dkaseg_df[f"weight_{d}"] = dkaseg_weights[d]
    out_csv = f"{args.out_prefix}_weights.csv"
    dkaseg_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # ---- QC: overlap heatmap ----
    plt.figure(figsize=(8, 6))
    plt.imshow(np.log1p(overlap), aspect="auto")
    plt.colorbar(label="log(1 + voxel overlap)")
    plt.xlabel("AAL labels")
    plt.ylabel("DK+ASEG labels")
    plt.title("AAL × DK+ASEG Voxel Overlap")
    plt.tight_layout()
    out_png = f"{args.out_prefix}_overlap_heatmap.png"
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)

    # Optional summary
    if any(missing.values()):
        print("Note: some AAL labels had missing weights in CSV (count per column):", missing)


if __name__ == "__main__":
    main()
