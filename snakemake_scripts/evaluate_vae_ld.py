#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Make sure we can import src.*
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.genotype_vae import GenotypeVAE  # type: ignore


def compute_ld_matrix(G: np.ndarray) -> np.ndarray:
    """
    Compute SNP×SNP r^2 matrix from genotype matrix G (N_individuals, M_snps).

    Steps:
      - center each SNP
      - standardize
      - correlation matrix
      - square to get r^2
    """
    # center
    Gc = G - G.mean(axis=0, keepdims=True)

    # std with ddof=1
    std = Gc.std(axis=0, ddof=1, keepdims=True)
    std[std == 0.0] = 1.0
    Gz = Gc / std

    corr = np.corrcoef(Gz, rowvar=False)  # (M, M)
    r2 = corr ** 2
    return r2


def main():
    p = argparse.ArgumentParser("Evaluate how well the VAE preserves LD structure")
    p.add_argument("--genotype", required=True, help="original genotype matrix .npy")
    p.add_argument("--meta", required=True, help="meta.pkl (not strictly needed but checked)")
    p.add_argument("--model", required=True, help="path to trained genotype_vae.pt")
    p.add_argument("--outdir", required=True, help="directory for LD comparison outputs")

    # Must match the training architecture
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--beta", type=float, default=0.0)

    p.add_argument("--max-snps", type=int, default=500,
                   help="max number of SNPs to subsample for LD comparison")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -----------------------------
    # Load data & model
    # -----------------------------
    geno = np.load(args.genotype).astype(np.float32)  # (N_ind, M_snps)

    # You’re not actually using meta here, but we touch it so we fail loudly if misaligned
    import pickle
    import pandas as pd

    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    meta = pd.DataFrame(meta)

    if geno.shape[0] != len(meta):
        raise ValueError(
            f"Genotype rows ({geno.shape[0]}) and meta rows ({len(meta)}) differ. "
            "Check that build_genotypes_for_vae kept ordering consistent."
        )

    N, M = geno.shape
    m = min(args.max_snps, M)

    # sample SNPs for LD computation (to keep it tractable)
    snp_idx = np.random.choice(M, size=m, replace=False)
    G_sub = geno[:, snp_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate_vae_ld] Using device: {device}")
    print(f"[evaluate_vae_ld] N={N}, M={M}, using m={m} SNPs for LD comparison.")

    # Build same architecture as in training
    model = GenotypeVAE(
        input_dim=M,
        width=args.hidden_dim,
        depth=args.depth,
        latent_dim=args.latent_dim,
        beta=args.beta,
    ).to(device)

    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # -----------------------------
    # Reconstruct genotypes
    # -----------------------------
    X = torch.tensor(geno, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon, mu, logvar = model(X)
    recon_np = recon.cpu().numpy()

    G_hat_sub = recon_np[:, snp_idx]

    # -----------------------------
    # Compute LD matrices
    # -----------------------------
    print("[evaluate_vae_ld] Computing LD matrices (r^2) for original and reconstructed genotypes...")
    ld_orig = compute_ld_matrix(G_sub)      # (m, m)
    ld_recon = compute_ld_matrix(G_hat_sub) # (m, m)

    # Upper triangles (excluding diagonal)
    iu = np.triu_indices(m, k=1)
    ld_orig_vec = ld_orig[iu]
    ld_recon_vec = ld_recon[iu]

    # Pearson correlation between LD entries
    ld_corr = np.corrcoef(ld_orig_vec, ld_recon_vec)[0, 1]
    print(f"[evaluate_vae_ld] Correlation between original and reconstructed r^2: {ld_corr:.4f}")

    # -----------------------------
    # Save numeric summary
    # -----------------------------
    summary_path = outdir / "ld_comparison_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"N_individuals: {N}\n")
        f.write(f"M_snps_total: {M}\n")
        f.write(f"M_snps_used: {m}\n")
        f.write(f"LD_r2_correlation: {ld_corr:.6f}\n")

    # Also save the sampled vectors if you ever want to post-process
    np.save(outdir / "ld_orig_vec.npy", ld_orig_vec)
    np.save(outdir / "ld_recon_vec.npy", ld_recon_vec)

    # -----------------------------
    # Plots
    # -----------------------------
    # 1. Scatter / hexbin of r^2 (orig vs recon)
    plt.figure(figsize=(6, 5))
    plt.hexbin(ld_orig_vec, ld_recon_vec, gridsize=80, mincnt=1)
    plt.xlabel("Original r²")
    plt.ylabel("Reconstructed r²")
    plt.title(f"LD (r²) comparison\ncorr = {ld_corr:.3f}")
    plt.colorbar(label="Pair count")
    plt.tight_layout()
    plt.savefig(outdir / "ld_scatter_orig_vs_recon.png", dpi=150)
    plt.close()

    # 2. Histogram of differences
    diff = ld_recon_vec - ld_orig_vec
    plt.figure(figsize=(6, 4))
    plt.hist(diff, bins=50, density=True)
    plt.xlabel("r²_recon - r²_orig")
    plt.ylabel("Density")
    plt.title("Distribution of LD (r²) differences")
    plt.tight_layout()
    plt.savefig(outdir / "ld_difference_hist.png", dpi=150)
    plt.close()

    print(f"[evaluate_vae_ld] Wrote summary and plots to {outdir}")


if __name__ == "__main__":
    main()
