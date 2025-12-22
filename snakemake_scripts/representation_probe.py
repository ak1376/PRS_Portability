#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Stats helpers (no scipy dependency)
# -----------------------------------------------------------------------------
def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).mean()) * np.sqrt((y * y).mean()))
    if denom <= 0:
        return float("nan")
    return float((x * y).mean() / denom)

def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    # dense ranks
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = (np.sqrt((rx * rx).mean()) * np.sqrt((ry * ry).mean()))
    if denom <= 0:
        return float("nan")
    return float((rx * ry).mean() / denom)


# -----------------------------------------------------------------------------
# Embedding windowing
# -----------------------------------------------------------------------------
def load_embeddings(path: Path) -> np.ndarray:
    Z = np.load(path)
    if Z.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported emb ndim={Z.ndim} shape={Z.shape}. Expected (N,d) or (N,L,d)."
        )
    return Z

def pooled_embeddings_for_window(emb: np.ndarray, s: int, e: int) -> np.ndarray:
    """
    emb:
      - (N,d): already pooled, returned as-is
      - (N,L,d): pool across L in [s:e] (mean)
    returns: (N,d)
    """
    if emb.ndim == 2:
        return emb.astype(np.float32, copy=False)

    # (N,L,d)
    if not (0 <= s < e <= emb.shape[1]):
        raise ValueError(f"window out of bounds for embeddings: s={s} e={e} L={emb.shape[1]}")
    Z = emb[:, s:e, :].mean(axis=1).astype(np.float32)
    return Z

def cosine_dist_pairs(Z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """
    Z: (N,d) float
    pairs: (P,2) int
    returns: (P,) cosine distance in [0,2]
    """
    Zi = Z[pairs[:, 0]]
    Zj = Z[pairs[:, 1]]
    Zi = Zi / (np.linalg.norm(Zi, axis=1, keepdims=True) + 1e-12)
    Zj = Zj / (np.linalg.norm(Zj, axis=1, keepdims=True) + 1e-12)
    cos_sim = np.sum(Zi * Zj, axis=1)
    return (1.0 - cos_sim).astype(np.float32)


# -----------------------------------------------------------------------------
# Pair sampling + window selection
# -----------------------------------------------------------------------------
def choose_windows(L_total: int, window_len: int, stride: int) -> List[Tuple[int, int]]:
    if window_len <= 0:
        raise ValueError("window_len must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if window_len >= L_total:
        return [(0, L_total)]

    starts = list(range(0, L_total - window_len + 1, stride))
    windows = [(s, s + window_len) for s in starts]
    if not windows:
        windows = [(max(0, L_total - window_len), L_total)]
    return windows

def sample_pairs(N: int, n_pairs: int, seed: int) -> np.ndarray:
    """
    Random pairs (i,j) with i<j, allowing repeats across sampling is OK,
    but we avoid i==j.
    """
    rng = np.random.default_rng(seed)
    i = rng.integers(0, N, size=n_pairs, dtype=np.int64)
    j = rng.integers(0, N, size=n_pairs, dtype=np.int64)

    bad = (i == j)
    while bad.any():
        j[bad] = rng.integers(0, N, size=int(bad.sum()), dtype=np.int64)
        bad = (i == j)

    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    return np.stack([lo, hi], axis=1).astype(np.int64)


# -----------------------------------------------------------------------------
# Hamming + IBS tract stats
# -----------------------------------------------------------------------------
def hamming_frac_pairs(Hwin: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """
    Hwin: (N,Lw) int/uint in {0,1}
    returns: (P,) fraction mismatched
    """
    H = (Hwin > 0.5).astype(np.uint8, copy=False)
    Pi = pairs[:, 0]
    Pj = pairs[:, 1]
    mism = (H[Pi] != H[Pj])
    return (mism.mean(axis=1)).astype(np.float32)

def mismatch_count_pairs(Hwin: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """
    Exact mismatch count (integer in [0, Lw]) for each pair.
    """
    H = (Hwin > 0.5).astype(np.uint8, copy=False)
    Pi = pairs[:, 0]
    Pj = pairs[:, 1]
    mism = (H[Pi] != H[Pj])
    return mism.sum(axis=1).astype(np.int32)

def maf_weights(Hwin01: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Weights per SNP using MAF estimated from the current window (on the sampled hap set).
    """
    H = (Hwin01 > 0.5).astype(np.float32, copy=False)
    p = H.mean(axis=0)
    return (1.0 / np.sqrt(p * (1.0 - p) + eps)).astype(np.float32)

def weighted_hamming_pairs(Hwin01: np.ndarray, pairs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted fraction mismatched for each pair.
    """
    H = (Hwin01 > 0.5).astype(np.uint8, copy=False)
    Pi = pairs[:, 0]
    Pj = pairs[:, 1]
    mism = (H[Pi] != H[Pj]).astype(np.float32)  # (P,Lw)
    num = (mism * w[None, :]).sum(axis=1)
    den = float(w.sum()) + 1e-12
    return (num / den).astype(np.float32)

def compute_ibs_stats_for_pairs(Hwin: np.ndarray, pairs: np.ndarray, min_tract: int) -> pd.DataFrame:
    """
    Hwin: (N,Lw) in {0,1}
    pairs: (P,2)
    """
    H = (Hwin > 0.5).astype(np.uint8, copy=False)
    P = pairs.shape[0]

    mean_tract = np.zeros(P, dtype=np.float32)
    max_tract = np.zeros(P, dtype=np.float32)
    total_match = np.zeros(P, dtype=np.float32)
    frac_match = np.zeros(P, dtype=np.float32)
    n_tracts = np.zeros(P, dtype=np.int32)

    Lw = H.shape[1]
    for k, (i, j) in enumerate(pairs):
        match = (H[i] == H[j])

        x = match.astype(np.int8)
        d = np.diff(np.concatenate(([0], x, [0])))
        starts = np.where(d == 1)[0]
        ends   = np.where(d == -1)[0]
        lens = (ends - starts).astype(np.int64)

        if lens.size > 0 and int(min_tract) > 1:
            lens = lens[lens >= int(min_tract)]

        if lens.size == 0:
            tm = float(match.sum())
            total_match[k] = tm
            frac_match[k] = tm / float(Lw)
            mean_tract[k] = 0.0
            max_tract[k] = 0.0
            n_tracts[k] = 0
        else:
            tm = float(lens.sum())
            total_match[k] = tm
            frac_match[k] = tm / float(Lw)
            mean_tract[k] = float(lens.mean())
            max_tract[k] = float(lens.max())
            n_tracts[k] = int(lens.size)

    df = pd.DataFrame({
        "hap_i": pairs[:, 0].astype(np.int64),
        "hap_j": pairs[:, 1].astype(np.int64),
        "mean_tract": mean_tract,
        "max_tract": max_tract,
        "total_match": total_match,
        "frac_match": frac_match,
        "n_tracts": n_tracts,
    })
    return df


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
TRACK_COLS = ["hamming", "frac_match", "total_match", "mean_tract", "max_tract"]

def _binned_median_curve(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    q: int,
    min_per_bin: int,
):
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < max(50, min_per_bin):
        return np.array([]), np.array([]), np.array([]), np.array([])

    tmp = pd.DataFrame({xcol: x, ycol: y})
    if np.nanmin(x) == np.nanmax(x):
        return np.array([]), np.array([]), np.array([]), np.array([])

    bins = pd.qcut(tmp[xcol], q=int(q), duplicates="drop")
    g = tmp.groupby(bins, observed=True)

    rows = []
    for _, d in g:
        if len(d) < int(min_per_bin):
            continue
        rows.append((
            float(d[xcol].median()),
            float(d[ycol].median()),
            float(d[ycol].quantile(0.25)),
            float(d[ycol].quantile(0.75)),
        ))
    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([])
    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

def plot_corr_summary_bar(corr_df: pd.DataFrame, out_png: Path):
    labels, meds, q1s, q3s = [], [], [], []
    for col in TRACK_COLS:
        rcol = f"{col}_spearman_r"
        if rcol not in corr_df.columns:
            continue
        vals = corr_df[rcol].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        labels.append(col)
        meds.append(float(np.median(vals)))
        q1s.append(float(np.quantile(vals, 0.25)))
        q3s.append(float(np.quantile(vals, 0.75)))

    if not labels:
        return

    x = np.arange(len(labels))
    meds = np.array(meds, dtype=float)
    yerr = np.vstack([meds - np.array(q1s), np.array(q3s) - meds])

    plt.figure(figsize=(9.0, 4.2))
    plt.bar(x, meds, yerr=yerr, capsize=4)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Spearman r across windows (median ± IQR)")
    plt.title("Window-aligned association: distance/IBS statistic vs embedding cosine_dist")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_binned_curve_across_windows(
    window_csvs: List[Path],
    out_png: Path,
    *,
    xcol: str,
    q: int,
    min_per_bin: int,
):
    dfs = []
    for p in window_csvs:
        df = pd.read_csv(p)
        if xcol not in df.columns or "cosine_dist" not in df.columns:
            continue
        dfs.append(df[[xcol, "cosine_dist"]].dropna())
    if not dfs:
        return

    allx = pd.concat([d[[xcol]] for d in dfs], ignore_index=True)[xcol].to_numpy()
    allx = allx[np.isfinite(allx)]
    if allx.size == 0:
        return

    grid = np.quantile(allx, np.linspace(0.05, 0.95, 60))
    grid = np.unique(grid)
    if grid.size < 6:
        return

    curves = []
    for p in window_csvs:
        df = pd.read_csv(p)
        x_mid, y_med, _, _ = _binned_median_curve(df, xcol, "cosine_dist", q=q, min_per_bin=min_per_bin)
        if x_mid.size < 6:
            continue
        y_interp = np.interp(grid, x_mid, y_med, left=np.nan, right=np.nan)
        curves.append(y_interp)

    if len(curves) < 3:
        return

    C = np.vstack(curves)  # (W, G)
    ok = np.mean(np.isfinite(C), axis=0) >= 0.5
    grid2 = grid[ok]
    C2 = C[:, ok]
    if grid2.size < 6:
        return

    med = np.nanmedian(C2, axis=0)
    lo = np.nanquantile(C2, 0.25, axis=0)
    hi = np.nanquantile(C2, 0.75, axis=0)

    plt.figure(figsize=(7.6, 4.9))
    plt.plot(grid2, med, linewidth=2.2)
    plt.fill_between(grid2, lo, hi, alpha=0.2)
    plt.xlabel(xcol)
    plt.ylabel("cosine_dist (lower = more similar)")
    plt.title(f"Across windows: binned median cosine_dist vs {xcol}\nmedian across windows ± IQR")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_scatter(df: pd.DataFrame, out_png: Path, *, xcol: str, ycol: str, title: str):
    plt.figure(figsize=(7, 6))
    plt.scatter(df[xcol].to_numpy(), df[ycol].to_numpy(), s=8, alpha=0.2)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_binned_trend(df: pd.DataFrame, out_png: Path, *, xcol: str, ycol: str, bins: np.ndarray, title: str):
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 100:
        return

    mids = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(len(mids), np.nan, dtype=float)
    se   = np.full(len(mids), np.nan, dtype=float)
    nbin = np.zeros(len(mids), dtype=int)

    for k in range(len(mids)):
        lo, hi = bins[k], bins[k+1]
        mk = (x >= lo) & (x < hi) if k < len(mids)-1 else (x >= lo) & (x <= hi)
        vals = y[mk]
        nbin[k] = int(vals.size)
        if vals.size == 0:
            continue
        mean[k] = float(vals.mean())
        se[k]   = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0

    plt.figure(figsize=(7, 6))
    plt.errorbar(mids, mean, yerr=se, marker="o", linestyle="-", capsize=3)
    plt.xlabel(f"{xcol} (bin midpoint)")
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def stratify_cosine_by_mismatch_count(
    df: pd.DataFrame,
    out_png: Path,
    out_csv: Path,
    *,
    mcount_col: str,
    ycol: str,
    top_k: int,
    min_pairs: int,
):
    """
    Picks the top-K most frequent mismatch-count strata (among those with >= min_pairs),
    and makes a boxplot of ycol in each stratum.
    Also writes a summary CSV (q25/q50/q75, n).
    """
    mc = df[mcount_col].to_numpy()
    y = df[ycol].to_numpy()
    m = np.isfinite(mc) & np.isfinite(y)
    mc = mc[m].astype(np.int64, copy=False)
    y = y[m].astype(np.float64, copy=False)

    vals, counts = np.unique(mc, return_counts=True)
    keep = counts >= int(min_pairs)
    vals = vals[keep]
    counts = counts[keep]
    if vals.size == 0:
        # still write an empty CSV so Snakemake has something deterministic if desired
        pd.DataFrame(columns=["mcount", "n_pairs", "y_q25", "y_q50", "y_q75", "y_mean", "y_std"]).to_csv(out_csv, index=False)
        return

    order = np.argsort(-counts)
    vals = vals[order][: int(top_k)]

    rows = []
    data = []
    labels = []
    for mc_val in vals:
        sel = (mc == mc_val)
        yy = y[sel]
        data.append(yy)
        labels.append(f"m={int(mc_val)}\n(n={int(sel.sum())})")
        rows.append({
            "mcount": int(mc_val),
            "n_pairs": int(sel.sum()),
            "y_q25": float(np.quantile(yy, 0.25)),
            "y_q50": float(np.quantile(yy, 0.50)),
            "y_q75": float(np.quantile(yy, 0.75)),
            "y_mean": float(yy.mean()),
            "y_std": float(yy.std(ddof=1)) if yy.size > 1 else 0.0,
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    plt.figure(figsize=(10.5, 4.2))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels)+1), labels, rotation=0)
    plt.xlabel("Exact mismatch count stratum (top-K frequent)")
    plt.ylabel(ycol)
    plt.title(f"{ycol} stratified by exact mismatch count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def stratify_cosine_by_weighted_bins(
    df: pd.DataFrame,
    out_png: Path,
    out_csv: Path,
    *,
    xcol: str,
    ycol: str,
    n_bins: int,
    min_pairs: int,
):
    """
    For weighted hamming, stratify by n_bins bins in [0,1].
    """
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m].astype(np.float64, copy=False)
    y = y[m].astype(np.float64, copy=False)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    labs = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        labs.append(f"[{lo:.2f},{hi:.2f})" if i < len(bins)-2 else f"[{lo:.2f},{hi:.2f}]")

    si = np.digitize(x, bins, right=False) - 1
    si = np.clip(si, 0, len(labs)-1)

    rows = []
    data = []
    labels = []
    for k, lab in enumerate(labs):
        sel = (si == k)
        if int(sel.sum()) < int(min_pairs):
            continue
        yy = y[sel]
        data.append(yy)
        labels.append(f"{lab}\n(n={int(sel.sum())})")
        rows.append({
            "bin": lab,
            "n_pairs": int(sel.sum()),
            "x_mean": float(x[sel].mean()),
            "y_q25": float(np.quantile(yy, 0.25)),
            "y_q50": float(np.quantile(yy, 0.50)),
            "y_q75": float(np.quantile(yy, 0.75)),
            "y_mean": float(yy.mean()),
            "y_std": float(yy.std(ddof=1)) if yy.size > 1 else 0.0,
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if len(data) == 0:
        return

    plt.figure(figsize=(10.5, 4.2))
    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(labels)+1), labels, rotation=0)
    plt.xlabel(f"{xcol} bins")
    plt.ylabel(ycol)
    plt.title(f"{ycol} stratified by {xcol} bins")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hap", required=True, help="hap1.npy (N x L_total), values in {0,1}")
    ap.add_argument("--meta", required=True, help="meta.pkl aligned to hap (rows correspond to hap rows)")
    ap.add_argument("--emb", required=True, help="hap1_embeddings.npy: (N,d) or (N,L_total,d)")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--window_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--n_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_tract", type=int, default=1)

    ap.add_argument("--q", type=int, default=20)
    ap.add_argument("--min_per_bin", type=int, default=200)
    ap.add_argument("--rep_choose", choices=["best_frac", "random", "first"], default="best_frac")

    # NEW: hamming modes + stratification controls
    ap.add_argument("--use_weighted_hamming", action="store_true",
                    help="Use MAF-weighted hamming (computed per window).")
    ap.add_argument("--top_k_mcount", type=int, default=3,
                    help="For plain hamming: keep top-K most frequent mismatch-count strata (rep window).")
    ap.add_argument("--min_pairs_per_mcount", type=int, default=200,
                    help="Only include mismatch-count strata with at least this many pairs (rep window).")
    ap.add_argument("--strata_bins", type=int, default=3,
                    help="For weighted hamming: stratify using this many bins in [0,1] (rep window).")

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    H = np.load(args.hap)
    if H.ndim != 2:
        raise ValueError(f"Expected hap 2D (N,L), got {H.shape}")
    H = (H > 0.5).astype(np.uint8, copy=False)
    N, L_total = H.shape

    meta = pd.read_pickle(args.meta)
    if len(meta) != N:
        raise ValueError(f"meta rows ({len(meta)}) != hap rows (N={N}). Must be aligned.")

    emb = load_embeddings(Path(args.emb))
    if emb.shape[0] != N:
        raise ValueError(f"emb N ({emb.shape[0]}) != hap N ({N})")

    windows = choose_windows(L_total, int(args.window_len), int(args.stride))

    pairs = sample_pairs(N, int(args.n_pairs), seed=int(args.seed))
    np.save(outdir / "pairs.npy", pairs)

    run_info = {
        "hap": str(args.hap),
        "meta": str(args.meta),
        "emb": str(args.emb),
        "N": int(N),
        "L_total": int(L_total),
        "window_len": int(args.window_len),
        "stride": int(args.stride),
        "n_pairs": int(args.n_pairs),
        "seed": int(args.seed),
        "min_tract": int(args.min_tract),
        "emb_shape": list(emb.shape),
        "windows": [{"start": int(s), "end": int(e)} for (s, e) in windows],
        "use_weighted_hamming": bool(args.use_weighted_hamming),
        "top_k_mcount": int(args.top_k_mcount),
        "min_pairs_per_mcount": int(args.min_pairs_per_mcount),
        "strata_bins": int(args.strata_bins),
    }
    (outdir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    corr_rows: List[Dict[str, Any]] = []
    window_csvs: List[Path] = []

    for wi, (s, e) in enumerate(windows, start=1):
        print(f"[window {wi}/{len(windows)}] s={s} e={e} (Lw={e-s})")

        Hwin = H[:, s:e]
        Zwin = pooled_embeddings_for_window(emb, s, e)

        df = compute_ibs_stats_for_pairs(Hwin, pairs, min_tract=int(args.min_tract))

        # ---- hamming mode ----
        if args.use_weighted_hamming:
            w = maf_weights(Hwin)
            df["hamming"] = weighted_hamming_pairs(Hwin, pairs, w)
            # mismatch count not meaningful for weighted; keep it anyway for inspection
            df["mismatch_count"] = mismatch_count_pairs(Hwin, pairs)
        else:
            df["hamming"] = hamming_frac_pairs(Hwin, pairs)
            df["mismatch_count"] = mismatch_count_pairs(Hwin, pairs)

        df["cosine_dist"] = cosine_dist_pairs(Zwin, pairs)

        # carry meta columns if present (no pop splitting, but useful for later)
        for col in ["population", "pop", "individual_id", "ind", "hap_id"]:
            if col in meta.columns:
                mi = meta.iloc[df["hap_i"].to_numpy()][col].to_numpy()
                mj = meta.iloc[df["hap_j"].to_numpy()][col].to_numpy()
                df[f"{col}_i"] = mi
                df[f"{col}_j"] = mj

        out_csv = outdir / f"pairs_window_s{s}_e{e}.csv"
        df.to_csv(out_csv, index=False)
        window_csvs.append(out_csv)

        row: Dict[str, Any] = {"start": int(s), "end": int(e)}

        y = df["cosine_dist"].to_numpy()
        for xcol in TRACK_COLS:
            x = df[xcol].to_numpy()
            row[f"{xcol}_spearman_r"] = spearmanr(x, y)
            row[f"{xcol}_pearson_r"] = pearsonr(x, y)

        corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows)
    corr_path = outdir / "window_correlation_summary.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"[wrote] {corr_path}")

    # fig1: median/IQR Spearman across windows
    fig1 = outdir / "corr_summary_median_iqr.png"
    plot_corr_summary_bar(corr_df, fig1)
    print(f"[wrote] {fig1}")

    # fig2: across-windows curve for frac_match
    fig2 = outdir / "binned_curve_across_windows_frac_match.png"
    plot_binned_curve_across_windows(
        window_csvs, fig2, xcol="frac_match", q=int(args.q), min_per_bin=int(args.min_per_bin)
    )
    print(f"[wrote] {fig2}")

    # fig3: across-windows curve for max_tract
    fig3 = outdir / "binned_curve_across_windows_max_tract.png"
    plot_binned_curve_across_windows(
        window_csvs, fig3, xcol="max_tract", q=int(args.q), min_per_bin=int(args.min_per_bin)
    )
    print(f"[wrote] {fig3}")

    # NEW: across-windows curve for hamming
    figH = outdir / "binned_curve_across_windows_hamming.png"
    plot_binned_curve_across_windows(
        window_csvs, figH, xcol="hamming", q=int(args.q), min_per_bin=int(args.min_per_bin)
    )
    print(f"[wrote] {figH}")

    # pick representative window
    rep_s = rep_e = None
    if args.rep_choose in ("best_frac", "first", "random") and len(corr_df) > 0:
        if args.rep_choose == "first":
            pick_idx = 0
        elif args.rep_choose == "random":
            rng = np.random.default_rng(int(args.seed))
            pick_idx = int(rng.integers(0, len(corr_df)))
        else:
            # "best_frac" = most negative association between frac_match and cosine_dist
            col = "frac_match_spearman_r"
            pick_idx = int(np.nanargmin(corr_df[col].to_numpy())) if col in corr_df.columns else 0

        rep_s = int(corr_df.loc[pick_idx, "start"])
        rep_e = int(corr_df.loc[pick_idx, "end"])
        (outdir / "representative_window.txt").write_text(f"s={rep_s}\ne={rep_e}\n")
        print(f"[wrote] {outdir / 'representative_window.txt'}")

    # -------------------------------------------------------------------------
    # NEW: representative-window hamming plots + 2–3 strata stratification
    # -------------------------------------------------------------------------
    if rep_s is not None and rep_e is not None:
        rep_csv = outdir / f"pairs_window_s{rep_s}_e{rep_e}.csv"
        if rep_csv.exists():
            df = pd.read_csv(rep_csv)

            # scatter
            fig_sc = outdir / "scatter_hamming_vs_cosine_repwindow.png"
            plot_scatter(
                df, fig_sc,
                xcol="hamming", ycol="cosine_dist",
                title=f"Representative window: hamming vs cosine_dist (s={rep_s} e={rep_e})"
            )
            print(f"[wrote] {fig_sc}")

            # binned trend
            bins = np.linspace(0.0, 1.0, 21)
            fig_bt = outdir / "binned_hamming_vs_cosine_repwindow.png"
            plot_binned_trend(
                df, fig_bt,
                xcol="hamming", ycol="cosine_dist",
                bins=bins,
                title=f"Representative window: binned cosine_dist vs hamming (s={rep_s} e={rep_e})"
            )
            print(f"[wrote] {fig_bt}")

            # stratify (2–3 groups)
            if args.use_weighted_hamming:
                out_png = outdir / "stratified_cosine_by_weighted_hamming_repwindow.png"
                out_csv = outdir / "stratified_by_weighted_hamming_summary_repwindow.csv"
                stratify_cosine_by_weighted_bins(
                    df, out_png, out_csv,
                    xcol="hamming", ycol="cosine_dist",
                    n_bins=int(args.strata_bins),
                    min_pairs=int(args.min_pairs_per_mcount),
                )
                print(f"[wrote] {out_png}")
                print(f"[wrote] {out_csv}")
            else:
                out_png = outdir / "stratified_cosine_by_mcount_repwindow.png"
                out_csv = outdir / "stratified_by_mcount_summary_repwindow.csv"
                stratify_cosine_by_mismatch_count(
                    df, out_png, out_csv,
                    mcount_col="mismatch_count",
                    ycol="cosine_dist",
                    top_k=int(args.top_k_mcount),
                    min_pairs=int(args.min_pairs_per_mcount),
                )
                print(f"[wrote] {out_png}")
                print(f"[wrote] {out_csv}")

    print(f"[done] outputs in: {outdir}")


if __name__ == "__main__":
    main()
