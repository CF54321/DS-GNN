# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import hypergeom
import matplotlib.pyplot as plt

from enigmatoolbox.datasets import load_summary_stats
from enigmatoolbox.permutation_testing import spin_test, shuf_test

from nilearn.image import load_img, resample_to_img
import nibabel as nib


# ---------- 0) optional: edge matrix -> node importance ----------
def edge_to_node_importance(W, mode='l1'):
    """
    W: (P,P) connectivity matrix (FC/SC).
    mode='l1': sum |w_ij|; mode='l2': sqrt(sum w_ij^2)
    """
    W = np.asarray(W)
    W = np.triu(W, 1) + np.triu(W, 1).T
    if mode == 'l1':
        return np.sum(np.abs(W), axis=1)
    elif mode == 'l2':
        return np.sqrt(np.sum(W**2, axis=1))
    else:
        raise ValueError("mode ∈ {'l1','l2'}")


# ---------- 1) label remap by voxel overlap (source -> aparc) ----------
def remap_values_by_overlap(src_labels_nii, src_values, dst_labels_nii):
    """
    Map values from source label NIfTI to target label NIfTI by voxel overlap.
    src_labels_nii: source label NIfTI (int labels)
    src_values: values aligned to source label IDs (dict/Series or array)
    dst_labels_nii: target label NIfTI (e.g., aparc in MNI)
    Returns DataFrame with columns: ['dst_label','value']
    """
    src_img = load_img(src_labels_nii)
    dst_img = load_img(dst_labels_nii)
    src_resamp = resample_to_img(src_img, dst_img, interpolation='nearest')

    src_data = np.asarray(src_resamp.get_fdata(), dtype=int)
    dst_data = np.asarray(dst_img.get_fdata(), dtype=int)

    src_labels = np.unique(src_data); src_labels = src_labels[src_labels > 0]
    dst_labels = np.unique(dst_data); dst_labels = dst_labels[dst_labels > 0]

    if isinstance(src_values, pd.Series):
        val_map = {int(k): float(v) for k, v in src_values.items()}
    elif isinstance(src_values, dict):
        val_map = {int(k): float(v) for k, v in src_values.items()}
    else:
        val_map = {int(k): float(v) for k, v in zip(src_labels, src_values)}

    dst_vals = []
    for lab in dst_labels:
        mask = (dst_data == lab)
        if not np.any(mask):
            dst_vals.append(np.nan); continue
        sub_src = src_data[mask]
        sub_src = sub_src[sub_src > 0]
        if sub_src.size == 0:
            dst_vals.append(np.nan); continue
        uniq, counts = np.unique(sub_src, return_counts=True)
        num = 0.0; den = 0.0
        for u, c in zip(uniq, counts):
            if u in val_map:
                num += val_map[u] * c
                den += c
        dst_vals.append(num / den if den > 0 else np.nan)

    return pd.DataFrame({'dst_label': dst_labels, 'value': dst_vals})


# ---------- 2) load ENIGMA summary stats (kept as your original) ----------
def load_enigma_scz():
    use_abs = True
    ss = load_summary_stats('depression')  # as in your code; alternatives: 'schizophrenia', 'asd'

    # Cortex (MDD adult)
    CT = ss['CortThick_case_vs_controls_adult']
    SA = ss['CortSurf_case_vs_controls_adult']
    SV = ss['SubVol_case_vs_controls']

    def get_names(df):
        for key in ['Structure', 'structure', 'ROI', 'roi', 'region', 'Region']:
            if key in df.columns:
                return df[key].astype(str).tolist()
        return [str(x) for x in df.index]

    def dvec(series):
        x = series.to_numpy()
        return np.abs(x) if use_abs else x

    return {
        'CT': CT,
        'CT_names': get_names(CT),
        'CT_d': dvec(CT['d_icv']),
        'CT_fdr': CT.get('fdr_p', pd.Series([np.nan]*len(CT))),
        'SV': SV,
        'SV_names': get_names(SV),
        'SV_d': dvec(SV['d_icv']),
        'SV_fdr': SV.get('fdr_p', pd.Series([np.nan]*len(SV))),
    }


# ---------- 3) correlations & permutation utilities (kept) ----------
def basic_corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {'n': int(m.sum()), 'pearson_r': pr, 'pearson_p': pp, 'spearman_r': sr, 'spearman_p': sp}

def topk_enrichment(imp, enigma_fdr_p=None, enigma_thr='fdr<0.05', topk=0.2, use_abs=True, n_perm=10000, seed=0):
    """
    Enrichment of model Top-k vs ENIGMA significant set (default FDR<0.05).
    If no FDR, pass enigma_thr as fraction on |d|.
    """
    rng = np.random.default_rng(seed)
    v = np.abs(imp) if use_abs else imp.copy()
    P = len(v)
    k = int(topk*P) if 0 < topk < 1 else int(topk)
    A = np.zeros(P, dtype=bool)
    A[np.argsort(v)[::-1][:k]] = True

    if isinstance(enigma_thr, str) and enigma_thr.lower() == 'fdr<0.05' and enigma_fdr_p is not None:
        B = (np.asarray(enigma_fdr_p) < 0.05)
    else:
        enigma_d = np.asarray(enigma_fdr_p)
        thr = np.nanpercentile(np.abs(enigma_d), 80)
        B = (np.abs(enigma_d) >= thr)

    overlap = int(np.sum(A & B)); K = int(np.sum(B))
    p_hyper = hypergeom.sf(overlap-1, P, K, k)

    perm = []
    for _ in range(n_perm):
        idx = rng.choice(P, size=k, replace=False)
        perm.append(int(np.sum(B[idx])))
    p_perm = (np.sum(np.array(perm) >= overlap) + 1) / (n_perm + 1)

    a = overlap; b = k - overlap; c = K - overlap; d = P - k - K + overlap
    OR = (a+0.5)*(d+0.5)/((b+0.5)*(c+0.5))
    return {'P': P, 'k': k, 'K': K, 'overlap': overlap, 'OR': float(OR), 'p_hyper': float(p_hyper), 'p_perm': float(p_perm)}

def spin_test_subset(x_all, y_all, mask, label,
                    surface_name='fsa5', parcellation_name='aparc',
                    n_rot=10000, corr_type='spearman'):
    """
    Spin test within a subset (mask) using enigmatoolbox.spin_test.
    x_all, y_all: length-68 vectors; mask: boolean array of same length
    """
    xm = x_all[mask]; ym = y_all[mask]
    if corr_type == 'spearman':
        obs_r = stats.spearmanr(xm, ym).correlation
    else:
        obs_r = stats.pearsonr(xm, ym)[0]

    x_nan = x_all.astype(float).copy()
    y_nan = y_all.astype(float).copy()
    x_nan[~mask] = np.nan
    y_nan[~mask] = np.nan

    p_spin, null_dist = spin_test(
        x_nan, y_nan,
        surface_name=surface_name,
        parcellation_name=parcellation_name,
        type=corr_type,
        n_rot=n_rot,
        null_dist=True
    )
    print(f"[{label}] subset size={int(mask.sum())}, obs {corr_type} r={obs_r:.3f}, spin p={p_spin:.4g}")
    if isinstance(null_dist, (list, np.ndarray)):
        null_arr = np.asarray(null_dist, dtype=float)
        print(f"[{label}] null mean={np.nanmean(null_arr):.3f}, std={np.nanstd(null_arr):.3f}")
    return {'obs_r': obs_r, 'p_spin': p_spin, 'null_dist': null_dist}

def enrich_in_subset(mask, label, topk=0.2, frac=0.2, n_perm=5000):
    x = x_all[mask]; y = y_all[mask]  # uses globals as in your code
    res_e = topk_enrichment(x, enigma_fdr_p=y, enigma_thr=frac,
                            topk=topk, use_abs=False, n_perm=n_perm)
    print(f"[{label}] enrichment: P={res_e['P']}, k={res_e['k']}, K={res_e['K']}, "
          f"overlap={res_e['overlap']}, OR={res_e['OR']:.2f}, "
          f"p_hyper={res_e['p_hyper']:.3f}, p_perm={res_e['p_perm']:.3f}")
    return res_e

def auc_enrichment(x, y, frac=0.2, n_perm=5000, seed=0, min_K=4):
    """
    x: |model importance| ; y: |ENIGMA d|
    frac: top fraction of y as positives
    Returns AUC and permutation p-value.
    """
    x = np.asarray(x); y = np.asarray(y)
    P = len(y); K = int(np.floor(frac * P))
    if K < min_K or K > P - min_K:
        return {'K': K, 'AUC': np.nan, 'U': np.nan, 'p_mw': np.nan, 'p_perm': np.nan}

    idx = np.argsort(y)[::-1]
    pos = np.zeros(P, dtype=bool); pos[idx[:K]] = True
    x_pos, x_neg = x[pos], x[~pos]

    U, p_mw = stats.mannwhitneyu(x_pos, x_neg, alternative='greater')
    auc = U / (len(x_pos) * len(x_neg))

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        mask = np.zeros(P, dtype=bool)
        mask[rng.choice(P, size=K, replace=False)] = True
        U_perm, _ = stats.mannwhitneyu(x[mask], x[~mask], alternative='greater')
        if U_perm >= U:
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    return {'K': int(K), 'AUC': float(auc), 'U': float(U), 'p_mw': float(p_mw), 'p_perm': float(p_perm)}

def auc_sweep(x, y, fracs=(0.10, 0.15, 0.20, 0.25, 0.30), n_perm=5000, min_K=4):
    rows = []
    for f in fracs:
        res = auc_enrichment(x, y, frac=f, n_perm=n_perm, min_K=min_K)
        rows.append({'frac': f, **res})
    return pd.DataFrame(rows)


# ---------- 4) main (paths/filenames are now CLI args; logic unchanged) ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ENIGMA concordance (stratified) with spin test and enrichment.")
    parser.add_argument("--cortex-csv", required=True,
                        help="CSV with columns: parcel, importance (model importance in cortex).")
    parser.add_argument("--subcortical-csv", required=True,
                        help="CSV with columns: parcel, importance (model importance in subcortex).")
    parser.add_argument("--scatter-out", required=True,
                        help="Output PNG path for stratified scatter (cortex).")
    parser.add_argument("--scatter-low-out", required=True,
                        help="Output PNG path for LOW subset scatter.")
    parser.add_argument("--scatter-high-out", required=True,
                        help="Output PNG path for HIGH subset scatter.")
    parser.add_argument("--title", default="MDD",
                        help="Title prefix for plots (e.g., MDD / SZ).")
    parser.add_argument("--n-rot", type=int, default=10000, help="Number of spins for spin_test.")
    parser.add_argument("--enrich-fracs", type=str, default="0.10,0.15,0.20,0.25,0.30",
                        help="Comma-separated fractions for AUC sweep (on |d| top-frac).")
    args = parser.parse_args()

    enigma = load_enigma_scz()  # as-is per your code

    # --- load your cortex & subcortex importance CSVs ---
    ct_raw = pd.read_csv(args.cortex_csv)
    ct_map = dict(zip(ct_raw['parcel'], ct_raw['importance']))

    sv_raw = pd.read_csv(args.subcortical_csv)
    sv_raw.columns = [c.strip() for c in sv_raw.columns]
    sv_raw['parcel'] = sv_raw['parcel'].astype(str).str.strip()
    sv_raw['importance'] = pd.to_numeric(sv_raw['importance'], errors='coerce')

    # --- cortex alignment (aparc order) ---
    ct_df = pd.DataFrame({
        'parcel': enigma['CT_names'],
        'enigma_d': enigma['CT_d'],
        'enigma_fdr': np.asarray(enigma['CT_fdr'])
    })
    ct_df['model_imp'] = ct_df['parcel'].map(ct_map)

    missing_ct = ct_df.loc[ct_df['model_imp'].isna(), 'parcel'].tolist()
    if missing_ct:
        print('Warning - missing cortex parcels in your CSV:', missing_ct)

    # whole-cortex correlation
    res = basic_corr(ct_df['model_imp'].to_numpy(), ct_df['enigma_d'].to_numpy())
    print("CT: basic correlations:", res)

    p_spin, null_dist = spin_test(ct_df['model_imp'].to_numpy(),
                                  ct_df['enigma_d'].to_numpy(),
                                  surface_name='fsa5', parcellation_name='aparc',
                                  type='spearman', n_rot=args.n_rot, null_dist=True)
    print(f"CT: spin-test Spearman p={p_spin:.4g}")

    # top-20% enrichment example (|importance| vs |d|)
    enr = topk_enrichment(
        np.abs(ct_df['model_imp'].to_numpy()),
        enigma_fdr_p=np.abs(ct_df['enigma_d'].to_numpy()),
        enigma_thr=0.2,     # top 20% of |d|
        topk=0.2,           # model top 20%
        use_abs=False,
        n_perm=5000
    )
    print("CT: enrichment:", enr)

    # --- stratification masks on aparc names (kept as in your code) ---
    import matplotlib.patches as mpatches

    def base_name(parcel):
        return parcel.split('_', 1)[1] if '_' in str(parcel) else str(parcel)

    LOW_UNIMODAL = {
        'pericalcarine','cuneus','lingual','lateraloccipital',
        'precentral','postcentral','paracentral',
        'transversetemporal',
        'bankssts',
    }
    HIGH_ASSOC = {
        'medialorbitofrontal','posteriorcingulate','precuneus','rostralanteriorcingulate','isthmuscingulate',
        'rostralmiddlefrontal','caudalmiddlefrontal','superiorfrontal','lateralorbitofrontal',
        'parsopercularis','parstriangularis','parsorbitalis','frontalpole',
        'inferiorparietal','supramarginal','superiorparietal','middletemporal','superiortemporal','temporalpole',
        'parahippocampal','entorhinal','fusiform','insula',
    }

    names_ct = ct_df['parcel'].astype(str).tolist()
    low_mask   = np.array([base_name(p) in LOW_UNIMODAL  for p in names_ct])
    high_mask  = np.array([base_name(p) in HIGH_ASSOC    for p in names_ct])
    other_mask = ~(low_mask | high_mask)

    print('LOW(unimodal)=', int(low_mask.sum()),
          'HIGH(association)=', int(high_mask.sum()),
          'OTHER=', int(other_mask.sum()))

    # subset correlations on |values|
    x_all = np.abs(ct_df['model_imp'].to_numpy())
    y_all = np.abs(ct_df['enigma_d'].to_numpy())

    def corr_report(mask, label):
        res_sub = basic_corr(x_all[mask], y_all[mask])
        print(f"[{label}] n={res_sub['n']}, "
              f"pearson r={res_sub['pearson_r']:.3f} (p={res_sub['pearson_p']:.3f}), "
              f"spearman r={res_sub['spearman_r']:.3f} (p={res_sub['spearman_p']:.3f})")
        return res_sub

    corr_report(low_mask,  "LOW (unimodal)")
    corr_report(high_mask, "HIGH (association)")

    # spin within subsets (|values|)
    res_low  = spin_test_subset(x_all, y_all, low_mask,  label='LOW (unimodal)',  n_rot=args.n_rot)
    res_high = spin_test_subset(x_all, y_all, high_mask, label='HIGH (association)', n_rot=args.n_rot)

    # AUC sweep in HIGH subset
    print("\n=== AUC sweep (HIGH subset) ===")
    fracs = tuple(float(s) for s in args.enrich_fracs.split(","))
    df_high = auc_sweep(x_all[high_mask], y_all[high_mask], fracs=fracs, n_perm=10000, min_K=4)
    print(df_high)

    # stratified scatter (cortex)
    colors = np.where(high_mask, 'red', np.where(low_mask, 'blue', 'lightgray'))
    plt.figure(figsize=(5,4))
    plt.scatter(x_all, y_all, s=24, alpha=0.9, c=colors, edgecolors='white', linewidths=0.5)
    plt.xlabel('|Model importance| (cortex)')
    plt.ylabel('|ENIGMA CT d|')
    plt.title(f'{args.title}: Cortex |importance| vs |d| (stratified)')
    legend_handles = [
        mpatches.Patch(color='red', label='High (association)'),
        mpatches.Patch(color='blue', label='Low (unimodal)'),
        mpatches.Patch(color='lightgray', label='Other'),
    ]
    plt.legend(handles=legend_handles, frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(args.scatter_out, dpi=200)
    print("Saved:", args.scatter_out)

    # per-subset scatter saves
    def save_subset(mask, color, label, fname):
        plt.figure(figsize=(5,4))
        plt.scatter(x_all[mask], y_all[mask], s=28, alpha=0.95, c=color, edgecolors='white', linewidths=0.6)
        plt.xlabel('|Model importance| (cortex)'); plt.ylabel('|ENIGMA CT d|')
        plt.title(f'{args.title}: {label}')
        plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()
        print("Saved:", fname)

    save_subset(low_mask,  'blue', 'LOW (unimodal)',  args.scatter_low_out)
    save_subset(high_mask, 'red',  'HIGH (association)', args.scatter_high_out)

    # --- subcortex (kept as in your code) ---
    sv_df = pd.DataFrame({
        'parcel': enigma['SV_names'],
        'enigma_d': enigma['SV_d'],
        'enigma_fdr': np.asarray(enigma['SV_fdr']),
    })

    sv_map = dict(zip(sv_raw['parcel'], sv_raw['importance']))
    aliases_enigma_to_canon = {
        'Laccumb': 'LAccumbens', 'Lamyg': 'LAmygdala', 'Lcaud': 'LCaudate',
        'Lhippo': 'LHippo', 'Lpal': 'LPallidum', 'Lput': 'LPutamen', 'Lthal': 'LThal',
        'Raccumb': 'RAccumbens', 'Ramyg': 'RAmygdala', 'Rcaud': 'RCaudate',
        'Rhippo': 'RHippo', 'Rpal': 'RPallidum', 'Rput': 'RPutamen', 'Rthal': 'RThal',
        'Llatvent': 'LLatVent', 'Rlatvent': 'RLatVent',
    }
    sv_map_expanded = sv_map.copy()
    for enigma_key, canon_key in aliases_enigma_to_canon.items():
        val = sv_map.get(canon_key)
        if val is not None:
            sv_map_expanded[enigma_key] = val
            sv_map_expanded[enigma_key.capitalize()] = val
            sv_map_expanded[enigma_key.upper()] = val

    sv_df['model_imp'] = sv_df['parcel'].map(sv_map_expanded)
    sv_no_vent = sv_df[~sv_df['parcel'].isin(['LLatVent', 'RLatVent'])].reset_index(drop=True)

    missing_sv = sv_no_vent.loc[sv_no_vent['model_imp'].isna(), 'parcel'].tolist()
    if missing_sv:
        print('Warning - missing subcortical parcels in your CSV:', missing_sv)

    res_sv = basic_corr(sv_no_vent['model_imp'].to_numpy(), sv_no_vent['enigma_d'].to_numpy())
    print("SV: basic correlations:", res_sv)

    p_shuf, null_sv = shuf_test(sv_no_vent['model_imp'].to_numpy(),
                                sv_no_vent['enigma_d'].to_numpy(),
                                n_rot=10000, type='spearman', null_dist=True)
    print(f"SV: shuf-test Spearman p={p_shuf:.4g}")
