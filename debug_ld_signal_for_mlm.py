#!/usr/bin/env python3
"""
debug_ld_signal_for_mlm.py

Quick sanity checks for whether your hap windows should contain LD / contextual
signal (i.e., whether MLM should be able to beat ~log(2)=0.693).

Checks:
  1) hap value support (should be {0,1})
  2) adjacent SNP bp-distance quantiles (are "contiguous SNP windows" contiguous in bp?)
  3) neighbor r^2 across haplotypes (is there short-range correlation at all?)
  4) optional: window-level summaries for random windows (bp span + mean neighbor r^2)

Usage:
  python debug_ld_signal_for_mlm.py \
      --hap /sietch_colab/akapoor/PRS_Portability/experiments/out_of_africa/processed_data/hap1.npy  \
      --positions /sietch_colab/akapoor/PRS_Portability/experiments/out_of_africa/processed_data/variant_positions_bp.npy \
      --window-len 512 \
      --n-windows 50 \
      --seed 42 \
      --max-sites 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def _quantiles(x: np.ndarray, qs=(0, 0.5, 0.9, 0.99, 0.999)) -> str:
    x = np.asarray(x)
    qv = np.quantile(x, qs)
    return ", ".join([f"q{int(q*1000)/10:g}={v:.6g}" for q, v in zip(qs, qv)])


def neighbor_r2(hap: np.ndarray) -> np.ndarray:
    """
    hap: (N, L) with values 0/1 (float or int)
    Returns r^2 for pairs (i, i+1) across individuals: shape (L-1,)
    """
    X = hap.astype(np.float32)
    x = X[:, :-1]
    y = X[:, 1:]

    x0 = x - x.mean(axis=0, keepdims=True)
    y0 = y - y.mean(axis=0, keepdims=True)

    num = (x0 * y0).sum(axis=0)
    den = np.sqrt((x0 * x0).sum(axis=0) * (y0 * y0).sum(axis=0) + 1e-12)
    r = num / den
    return r * r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hap", type=str, required=True, help="hap1.npy (N, L)")
    ap.add_argument("--positions", type=str, required=True, help="variant_positions_bp.npy (L,)")
    ap.add_argument("--max-sites", type=int, default=5000, help="max number of sites to use for global r^2 check")
    ap.add_argument("--window-len", type=int, default=512, help="SNP-count window length (for window summaries)")
    ap.add_argument("--n-windows", type=int, default=50, help="number of random windows to summarize")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hap_path = Path(args.hap)
    pos_path = Path(args.positions)
    if not hap_path.exists():
        raise FileNotFoundError(hap_path)
    if not pos_path.exists():
        raise FileNotFoundError(pos_path)

    H = np.load(hap_path)
    pos = np.load(pos_path)

    if H.ndim != 2:
        raise ValueError(f"hap must be 2D (N,L). got {H.shape}")
    if pos.ndim != 1:
        raise ValueError(f"positions must be 1D (L,). got {pos.shape}")
    if H.shape[1] != pos.shape[0]:
        raise ValueError(f"hap L={H.shape[1]} but positions L={pos.shape[0]}")

    N, L = H.shape
    print("=" * 80)
    print("[1] Basic shapes")
    print(f"hap: {hap_path}  shape=(N={N}, L={L}) dtype={H.dtype}")
    print(f"pos: {pos_path}  shape=(L={pos.shape[0]},) dtype={pos.dtype}")

    print("=" * 80)
    print("[2] Value support (should be {0,1} for your current vocab_size=3, mask_id=2)")
    u, c = np.unique(H, return_counts=True)
    pairs = list(zip(u.tolist(), c.tolist()))
    print("unique values and counts:", pairs)
    if not set(u.tolist()).issubset({0, 1}):
        print("WARNING: hap contains values outside {0,1}. If you have a 3rd state,")
        print("         DO NOT use mask_id equal to that state. Use a fresh mask_id and increase vocab_size.")

    print("=" * 80)
    print("[3] Adjacent SNP bp distances (are index-contiguous SNPs physically local?)")
    d_bp = np.diff(pos.astype(np.float64))
    print("adjacent bp distance:", _quantiles(d_bp))
    print(f"min={d_bp.min():.6g} median={np.median(d_bp):.6g} max={d_bp.max():.6g}")
    if np.median(d_bp) > 5_000:
        print("WARNING: median adjacent bp gap is large. Your 'contiguous SNP windows' may span huge bp distances.")
        print("         If so, LD may be ~0 and MLM will sit near log(2).")

    print("=" * 80)
    print("[4] Neighbor r^2 (global quick check on first chunk of sites)")
    end = min(L, int(args.max_sites))
    if end < 2:
        raise ValueError("Not enough sites for r^2 check.")
    r2 = neighbor_r2(H[:, :end])
    print(f"using sites [0:{end}) -> {end-1} neighbor pairs")
    print("neighbor r^2:", _quantiles(r2, qs=(0, 0.5, 0.9, 0.99)))
    print(f"mean={float(r2.mean()):.6g}, median={float(np.median(r2)):.6g}")
    if float(np.quantile(r2, 0.99)) < 1e-3:
        print("WARNING: even the 99th percentile neighbor r^2 is tiny. Very little short-range correlation present.")

    print("=" * 80)
    print("[5] Random window summaries (bp span + mean neighbor r^2 within window)")
    rng = np.random.default_rng(int(args.seed))
    w = int(args.window_len)
    if w <= 1 or w > L:
        print(f"skipping window summaries (window_len={w} invalid for L={L})")
        return

    max_start = L - w
    spans = []
    means = []
    for _ in range(int(args.n_windows)):
        s = int(rng.integers(0, max_start + 1))
        e = s + w
        span_bp = float(pos[e - 1] - pos[s])
        win_r2 = neighbor_r2(H[:, s:e]).mean()
        spans.append(span_bp)
        means.append(float(win_r2))

    spans = np.asarray(spans, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)

    print(f"sampled {len(spans)} windows of {w} SNPs")
    print("window bp span:", _quantiles(spans, qs=(0, 0.5, 0.9, 0.99)))
    print("window mean neighbor r^2:", _quantiles(means, qs=(0, 0.5, 0.9, 0.99)))
    print(f"median span={float(np.median(spans)):.6g} bp, median mean r^2={float(np.median(means)):.6g}")

    if float(np.median(spans)) > 1e6 and float(np.median(means)) < 1e-3:
        print("DIAGNOSIS: windows are huge in bp and have near-zero LD signal -> MLM near chance is expected.")
        print("FIX: subset a contiguous bp region, or window by bp-span, not by SNP index after global thinning.")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
