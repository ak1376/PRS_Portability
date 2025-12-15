#!/usr/bin/env python3
"""
Genotype reconstruction diagnostics for VAE.

Given:
  - real genotypes:   (n_individuals, n_snps)  in [0, 1, 2] or [0, 1]
  - recon genotypes:  same shape as real, from VAE

This script:
  * creates heatmaps of original vs reconstructed genotypes
  * computes reconstruction error heatmaps
  * creates MSE histograms (per-individual and per-SNP)
  * computes summary reconstruction metrics

Usage example:
  python -m src.genotype_diagnostics \
      --real experiments/out_of_africa/genotypes/all_individuals.npy \
      --recon experiments/out_of_africa/vae/reconstructed_genotypes.npy \
      --outdir experiments/out_of_africa/vae/genotype_diagnostics \
      --max-individuals 100 \
      --max-snps 1000 \
      --seed 42
"""

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _subsample_for_plot(matrix: np.ndarray, max_rows: int, max_cols: int, seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsample matrix for plotting if it's too large.
    
    Returns:
        subsampled_matrix, row_indices, col_indices
    """
    n_rows, n_cols = matrix.shape
    
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Sample row indices
    if n_rows > max_rows:
        row_idx = np.sort(rng.choice(n_rows, size=max_rows, replace=False))
    else:
        row_idx = np.arange(n_rows)
    
    # Sample column indices
    if n_cols > max_cols:
        col_idx = np.sort(rng.choice(n_cols, size=max_cols, replace=False))
    else:
        col_idx = np.arange(n_cols)
    
    subsampled = matrix[np.ix_(row_idx, col_idx)]
    return subsampled, row_idx, col_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True, help="Path to real genotype .npy")
    p.add_argument("--recon", required=True, help="Path to reconstructed genotype .npy")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument(
        "--max-individuals",
        type=int,
        default=100,
        help="Maximum number of individuals to show in heatmaps (default: 100)",
    )
    p.add_argument(
        "--max-snps",
        type=int,
        default=1000,
        help="Maximum number of SNPs to show in heatmaps (default: 1000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling",
    )
    p.add_argument(
        "--scale-to-diploid",
        action="store_true",
        help="If set, multiply genotypes by 2 to convert [0,1] to [0,2] scale",
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
    print(f"[genotype_diagnostics] Loaded {n_ind} individuals, {n_snps} SNPs")

    # Scale to diploid if needed
    if args.scale_to_diploid:
        print("[genotype_diagnostics] Scaling genotypes from [0,1] to [0,2]")
        real = real * 2
        recon = recon * 2
    
    # Clip reconstructed values to valid range
    recon = np.clip(recon, 0, 2)

    print(f"[genotype_diagnostics] Real genotypes: min={real.min():.3f}, max={real.max():.3f}")
    print(f"[genotype_diagnostics] Recon genotypes: min={recon.min():.3f}, max={recon.max():.3f}")

    # ------------------------------------------------------------------
    # Compute reconstruction metrics
    # ------------------------------------------------------------------
    error = np.abs(real - recon)
    mse_per_individual = np.mean((real - recon)**2, axis=1)  # MSE for each individual
    mse_per_snp = np.mean((real - recon)**2, axis=0)  # MSE for each SNP
    overall_mse = np.mean((real - recon)**2)
    overall_mae = np.mean(error)
    
    # Correlation between real and reconstructed
    real_flat = real.flatten()
    recon_flat = recon.flatten()
    if np.std(real_flat) > 0 and np.std(recon_flat) > 0:
        genotype_corr = np.corrcoef(real_flat, recon_flat)[0, 1]
    else:
        genotype_corr = float('nan')

    metrics = {
        "n_individuals": int(n_ind),
        "n_snps": int(n_snps),
        "overall_mse": float(overall_mse),
        "overall_mae": float(overall_mae),
        "genotype_correlation": float(genotype_corr),
        "mse_per_individual_mean": float(np.mean(mse_per_individual)),
        "mse_per_individual_std": float(np.std(mse_per_individual)),
        "mse_per_snp_mean": float(np.mean(mse_per_snp)),
        "mse_per_snp_std": float(np.std(mse_per_snp)),
    }

    # Save metrics
    metrics_path = outdir / "genotype_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[genotype_diagnostics] Overall MSE: {overall_mse:.6f}")
    print(f"[genotype_diagnostics] Overall MAE: {overall_mae:.6f}")
    print(f"[genotype_diagnostics] Genotype correlation: {genotype_corr:.4f}")
    print(f"[genotype_diagnostics] Saved metrics to {metrics_path}")

    # ------------------------------------------------------------------
    # Subsample for heatmaps
    # ------------------------------------------------------------------
    real_plot, ind_idx, snp_idx = _subsample_for_plot(
        real, args.max_individuals, args.max_snps, args.seed
    )
    recon_plot = recon[np.ix_(ind_idx, snp_idx)]
    error_plot = error[np.ix_(ind_idx, snp_idx)]
    
    plot_individuals = len(ind_idx)
    plot_snps = len(snp_idx)
    print(f"[genotype_diagnostics] Plotting {plot_individuals} individuals × {plot_snps} SNPs")

    # ------------------------------------------------------------------
    # Genotype heatmaps
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Real genotypes
    im1 = axes[0].imshow(real_plot, aspect='auto', vmin=0, vmax=2, cmap='viridis')
    axes[0].set_title('Real Genotypes')
    axes[0].set_xlabel('SNP index')
    axes[0].set_ylabel('Individual index')
    plt.colorbar(im1, ax=axes[0], label='Genotype dosage')
    
    # Reconstructed genotypes
    im2 = axes[1].imshow(recon_plot, aspect='auto', vmin=0, vmax=2, cmap='viridis')
    axes[1].set_title('Reconstructed Genotypes')
    axes[1].set_xlabel('SNP index')
    axes[1].set_ylabel('Individual index')
    plt.colorbar(im2, ax=axes[1], label='Genotype dosage')
    
    # Reconstruction error
    im3 = axes[2].imshow(error_plot, aspect='auto', vmin=0, vmax=2, cmap='Reds')
    axes[2].set_title('Reconstruction Error (|Real - Recon|)')
    axes[2].set_xlabel('SNP index')
    axes[2].set_ylabel('Individual index')
    plt.colorbar(im3, ax=axes[2], label='Absolute error')
    
    plt.tight_layout()
    plt.savefig(outdir / "genotype_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[genotype_diagnostics] Saved genotype heatmaps to {outdir / 'genotype_heatmaps.png'}")

    # ------------------------------------------------------------------
    # MSE histograms
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # MSE per individual
    axes[0, 0].hist(mse_per_individual, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('MSE per individual')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'MSE per Individual (mean={np.mean(mse_per_individual):.4f})')
    axes[0, 0].axvline(np.mean(mse_per_individual), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # MSE per SNP
    axes[0, 1].hist(mse_per_snp, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('MSE per SNP')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'MSE per SNP (mean={np.mean(mse_per_snp):.4f})')
    axes[0, 1].axvline(np.mean(mse_per_snp), color='red', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    # Reconstruction error distribution
    axes[1, 0].hist(error.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Absolute reconstruction error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Reconstruction Error Distribution (MAE={overall_mae:.4f})')
    axes[1, 0].axvline(overall_mae, color='red', linestyle='--', label='Mean')
    axes[1, 0].legend()
    
    # Box plot: Reconstructed values by real genotype category
    n_points = min(10000, real.size)  # Subsample for analysis
    idx = np.random.choice(real.size, size=n_points, replace=False)
    real_sample = real.flat[idx]
    recon_sample = recon.flat[idx]
    real_sample_int = np.round(real_sample).astype(int)  # Convert to discrete categories
    
    # Create box plot data
    box_data = []
    box_labels = []
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for geno in [0, 1, 2]:
        mask = real_sample_int == geno
        if np.sum(mask) > 0:  # Only include if we have data
            recon_values = recon_sample[mask]
            box_data.append(recon_values)
            box_labels.append(f'Real = {geno}\n(n={np.sum(mask)})')
    
    # Create box plot
    bp = axes[1, 1].boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                           notch=True, showfliers=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add horizontal lines at ideal reconstruction values
    ideal_values = [0.0, 0.5, 1.0]
    for i, ideal in enumerate(ideal_values[:len(box_data)]):
        axes[1, 1].axhline(y=ideal, color='red', linestyle='--', alpha=0.5, 
                          xmin=(i+0.7)/len(box_data), xmax=(i+1.3)/len(box_data))
    
    axes[1, 1].set_ylabel('Reconstructed genotypes')
    axes[1, 1].set_xlabel('Real genotype category')
    axes[1, 1].set_title(f'Reconstruction by Genotype Category (r={genotype_corr:.3f})')
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "reconstruction_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[genotype_diagnostics] Saved reconstruction metrics to {outdir / 'reconstruction_metrics.png'}")

    print(f"[genotype_diagnostics] All diagnostics saved to {outdir}")


if __name__ == "__main__":
    main()