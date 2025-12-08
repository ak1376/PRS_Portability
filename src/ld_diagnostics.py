#!/usr/bin/env python3
"""
LD diagnostics for genotype VAE.

Given:
  - real genotypes:   (n_individuals, n_snps)  e.g. 0/1/2 or in [0, 1]
  - recon genotypes:  same shape as real, from VAE

This script:
  * chooses a contiguous SNP window
  * computes LD (r²) matrices for real vs recon
  * draws LD heatmaps
  * computes LD decay curves
  * computes a scalar similarity metric: corr(real_r2, recon_r2)

Usage example:

  python -m src.ld_diagnostics \
      --real experiments/out_of_africa/genotypes/all_individuals.npy \
      --recon experiments/out_of_africa/vae/recon_all.npy \
      --outdir experiments/out_of_africa/vae/ld_diagnostics \
      --window-size 500 \
      --max-snps-heatmap 200 \
      --seed 42
"""

import argparse
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt


def _choose_window(n_snps: int, window_size: int, start: int | None, seed: int | None):
    if start is not None:
        start = max(0, min(start, n_snps - 1))
        end = min(n_snps, start + window_size)
        return start, end

    if n_snps <= window_size:
        return 0, n_snps

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n_snps - window_size))
    end = start + window_size
    return start, end


def _r2_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute r² LD matrix from genotype matrix X (n_individuals, n_snps).

    Returns:
      r2: (n_snps, n_snps)
    """
    # Center columns
    Xc = X - X.mean(axis=0, keepdims=True)
    # Standard deviation per SNP
    std = Xc.std(axis=0, ddof=1)
    # Avoid divide-by-zero: keep only SNPs with non-zero variance
    nonzero = std > 0
    if nonzero.sum() < 2:
        raise ValueError("Not enough polymorphic sites to compute LD.")

    Xnz = Xc[:, nonzero] / std[nonzero]  # standardize
    # Correlation matrix
    R = np.corrcoef(Xnz, rowvar=False)
    # r²
    R2 = R ** 2
    # Put back into a full matrix (fill monomorphic SNPs with 0 LD)
    n = X.shape[1]
    full = np.zeros((n, n), dtype=np.float32)
    idx = np.where(nonzero)[0]
    full[np.ix_(idx, idx)] = R2
    # Make sure there are no NaNs
    full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
    return full


def _upper_tri_vec(A: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(A.shape[0], k=1)
    return A[i, j]


def _ld_decay(r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute LD decay as mean r² vs SNP distance in index units.
    r2: (L, L)
    Returns:
      distances (1..L-1), mean_r2[dist]
    """
    L = r2.shape[0]
    dists = np.arange(1, L)
    mean_r2 = np.zeros_like(dists, dtype=float)

    for k, d in enumerate(dists, start=0):
        # pairs (i, i+d)
        i = np.arange(0, L - d)
        j = i + d
        vals = r2[i, j]
        if vals.size == 0:
            mean_r2[k] = np.nan
        else:
            mean_r2[k] = np.nanmean(vals)

    return dists, mean_r2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True, help="Path to real genotype .npy")
    p.add_argument("--recon", required=True, help="Path to reconstructed genotype .npy")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument(
        "--window-size",
        type=int,
        default=500,
        help="Number of contiguous SNPs to use for LD diagnostics (default: 500)",
    )
    p.add_argument(
        "--start",
        type=int,
        default=None,
        help="Optional explicit window start SNP index (0-based). "
             "If not given, a random window is chosen.",
    )
    p.add_argument(
        "--max-snps-heatmap",
        type=int,
        default=200,
        help="If window has more SNPs than this, subsample columns for heatmaps "
             "to keep plots readable.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for choosing window if --start is not given.",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load genotypes
    # ------------------------------------------------------------------
    real = np.load(args.real)   # (n_individuals, n_snps)
    recon = np.load(args.recon) # same shape

    if real.shape != recon.shape:
        raise ValueError(f"Shape mismatch: real {real.shape} vs recon {recon.shape}")

    n_ind, n_snps = real.shape
    print(f"[ld_diagnostics] Loaded {n_ind} individuals, {n_snps} SNPs")

    # Optionally rescale if you stored 0/1/2 but VAE outputs [0,1]
    # Uncomment if needed:
    # real = real / 2.0
    # recon = np.clip(recon, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Choose window
    # ------------------------------------------------------------------
    start, end = _choose_window(
        n_snps=n_snps,
        window_size=args.window_size,
        start=args.start,
        seed=args.seed,
    )
    print(f"[ld_diagnostics] Using SNP window [{start}:{end}) (size={end - start})")

    real_w = real[:, start:end]
    recon_w = recon[:, start:end]
    Lw = real_w.shape[1]

    # ------------------------------------------------------------------
    # Compute LD (r²) matrices
    # ------------------------------------------------------------------
    print("[ld_diagnostics] Computing LD (r²) matrices...")
    r2_real = _r2_matrix(real_w)
    r2_recon = _r2_matrix(recon_w)

    # ------------------------------------------------------------------
    # Scalar LD similarity metric
    # ------------------------------------------------------------------
    v_real = _upper_tri_vec(r2_real)
    v_recon = _upper_tri_vec(r2_recon)

    # Add a tiny jitter if both are constant
    if np.all(v_real == v_real[0]) or np.all(v_recon == v_recon[0]):
        ld_corr = float("nan")
    else:
        ld_corr = float(np.corrcoef(v_real, v_recon)[0, 1])

    metrics = {
        "n_individuals": int(n_ind),
        "n_snps_total": int(n_snps),
        "window_start": int(start),
        "window_end": int(end),
        "window_size": int(Lw),
        "ld_r2_correlation": ld_corr,
    }

    metrics_path = outdir / "ld_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[ld_diagnostics] LD r² correlation (upper triangle): {ld_corr:.4f}")
    print(f"[ld_diagnostics] Saved metrics to {metrics_path}")

    # ------------------------------------------------------------------
    # LD decay curves
    # ------------------------------------------------------------------
    print("[ld_diagnostics] Computing LD decay...")
    dist_real, decay_real = _ld_decay(r2_real)
    dist_recon, decay_recon = _ld_decay(r2_recon)

    plt.figure(figsize=(6, 4))
    plt.plot(dist_real, decay_real, label="Real", linewidth=1.0)
    plt.plot(dist_recon, decay_recon, label="Recon", linewidth=1.0)
    plt.xlabel("SNP distance (indices)")
    plt.ylabel("Mean r²")
    plt.title("LD decay (real vs recon)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ld_decay.png", dpi=150)
    plt.close()
    print(f"[ld_diagnostics] Saved LD decay plot to {outdir / 'ld_decay.png'}")

    # ------------------------------------------------------------------
    # Heatmaps (optionally subsample SNPs)
    # ------------------------------------------------------------------
    if Lw > args.max_snps_heatmap:
        # Take evenly spaced subset of SNPs for plotting
        idx = np.linspace(0, Lw - 1, args.max_snps_heatmap, dtype=int)
        r2_real_plot = r2_real[np.ix_(idx, idx)]
        r2_recon_plot = r2_recon[np.ix_(idx, idx)]
    else:
        r2_real_plot = r2_real
        r2_recon_plot = r2_recon

    # Real LD heatmap
    plt.figure(figsize=(5, 4))
    im = plt.imshow(r2_real_plot, aspect="auto", origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, label="r²")
    plt.title("Real LD (r²)")
    plt.xlabel("SNP index (subset)")
    plt.ylabel("SNP index (subset)")
    plt.tight_layout()
    plt.savefig(outdir / "ld_heatmap_real.png", dpi=150)
    plt.close()

    # Recon LD heatmap
    plt.figure(figsize=(5, 4))
    im = plt.imshow(r2_recon_plot, aspect="auto", origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, label="r²")
    plt.title("Reconstructed LD (r²)")
    plt.xlabel("SNP index (subset)")
    plt.ylabel("SNP index (subset)")
    plt.tight_layout()
    plt.savefig(outdir / "ld_heatmap_recon.png", dpi=150)
    plt.close()

    print(f"[ld_diagnostics] Saved LD heatmaps to {outdir}")


if __name__ == "__main__":
    main()
