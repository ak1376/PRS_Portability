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

from src.genotype_vae import GenotypeVAE, GenotypeCNNVAE  # type: ignore


def compute_ld_matrix(G: np.ndarray) -> np.ndarray:
    """
    Compute SNP×SNP r^2 matrix from genotype matrix G (N_individuals, M_snps).

    Steps:
      - center each SNP
      - standardize
      - correlation matrix
      - square to get r^2
    """
    Gc = G - G.mean(axis=0, keepdims=True)
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
    p.add_argument("--arch", type=str, default="mlp", choices=["mlp", "cnn"], help="Architecture type")
    p.add_argument("--channels", type=str, default="32,64,128", help="CNN channel configuration (comma-separated)")
    p.add_argument("--kernel-size", type=int, default=5, help="CNN kernel size")
    p.add_argument("--dropout", type=float, default=0.0, help="CNN dropout rate")
    p.add_argument("--use-batchnorm", type=int, default=1, help="CNN use batch normalization (1=True, 0=False)")

    # NOTE: kept only for Snakemake compatibility; we do NOT use this
    p.add_argument(
        "--max-snps",
        type=int,
        default=0,
        help="(ignored here; max_snps was already applied when building all_individuals.npy)",
    )

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -----------------------------
    # Load data & meta
    # -----------------------------
    geno = np.load(args.genotype).astype(np.float32)  # (N_ind, M_snps)

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

    # ---------------------------------------------------------
    # Ensure genotypes are in dosage (0/1/2) space, like training
    # ---------------------------------------------------------
    g_min = float(np.nanmin(geno))
    g_max = float(np.nanmax(geno))
    print(f"[evaluate_vae_ld] Raw genotype stats: min={g_min}, max={g_max}")

    if g_max <= 1.0 + 1e-6:
        print(
            "[evaluate_vae_ld] Detected genotypes scaled to [0,1]; "
            "rescaling back to ~{0,1,2} by *2."
        )
        geno = np.clip(geno, 0.0, 1.0) * 2.0

    if (geno < 0.0).any() or (geno > 2.0).any():
        bad = np.sum((geno < 0.0) | (geno > 2.0))
        frac_bad = bad / geno.size
        print(
            f"[evaluate_vae_ld] Warning: found {bad} genotype entries "
            f"({frac_bad:.4%}) outside [0,2]. Clipping."
        )
        geno = np.clip(geno, 0.0, 2.0)

    geno_dosage = geno.astype(np.float32)  # for LD
    geno_scaled = geno_dosage / 2.0       # for VAE input

    # -----------------------------
    # Restrict to validation individuals (if val_idx.npy exists)
    # -----------------------------
    model_path = Path(args.model)
    split_dir = model_path.parent
    val_idx_path = split_dir / "val_idx.npy"

    if val_idx_path.exists():
        val_idx = np.load(val_idx_path)
        print(
            f"[evaluate_vae_ld] Found val_idx at {val_idx_path}. "
            f"Restricting LD evaluation to {len(val_idx)} validation individuals."
        )
        geno_dosage = geno_dosage[val_idx, :]
        geno_scaled = geno_scaled[val_idx, :]
        meta = meta.iloc[val_idx].reset_index(drop=True)
    else:
        print(
            "[evaluate_vae_ld] WARNING: val_idx.npy not found next to model; "
            "evaluating LD on ALL individuals."
        )

    N, M = geno_dosage.shape

    # -----------------------------
    # Use ALL segregating SNPs in this subset
    # -----------------------------
    per_snp_var = geno_dosage.var(axis=0, ddof=1)
    seg_mask = per_snp_var > 0.0
    seg_idx = np.where(seg_mask)[0]

    if len(seg_idx) < 2:
        raise ValueError(
            f"[evaluate_vae_ld] Not enough segregating SNPs in subset to compute LD "
            f"(found {len(seg_idx)})."
        )

    snp_idx = seg_idx
    m = len(seg_idx)
    print(
        f"[evaluate_vae_ld] Using ALL {m} segregating SNPs for LD comparison "
        "(genotype matrix was already subset upstream)."
    )

    G_sub = geno_dosage[:, snp_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate_vae_ld] Using device: {device}")
    print(
        f"[evaluate_vae_ld] N={N}, M_total={M}, seg_SNPs={len(seg_idx)}, "
        f"using m={m} SNPs for LD comparison."
    )

    # -----------------------------
    # Build same architecture as in training
    # -----------------------------
    if args.arch == "cnn":
        # Parse channel configuration
        channels = tuple(int(c) for c in args.channels.split(","))
        model = GenotypeCNNVAE(
            input_dim=M,
            latent_dim=args.latent_dim,
            channels=channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            use_batchnorm=bool(args.use_batchnorm),
            beta=args.beta,
        ).to(device)
    else:
        # MLP architecture
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
    # Reconstruct genotypes (scaled input)
    # -----------------------------
    X = torch.tensor(geno_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon_scaled, mu, logvar = model(X)

    # back to dosage scale for LD
    recon_dosage = (recon_scaled.cpu().numpy()) * 2.0
    G_hat_sub = recon_dosage[:, snp_idx]

    # -----------------------------
    # Compute LD matrices
    # -----------------------------
    print(
        "[evaluate_vae_ld] Computing LD matrices (r^2) for original and "
        "reconstructed genotypes..."
    )
    ld_orig = compute_ld_matrix(G_sub)      # (m, m)
    ld_recon = compute_ld_matrix(G_hat_sub) # (m, m)

    iu = np.triu_indices(m, k=1)
    ld_orig_vec = ld_orig[iu]
    ld_recon_vec = ld_recon[iu]

    # NaN-safe correlation between LD entries (use ALL valid pairs)
    valid = np.isfinite(ld_orig_vec) & np.isfinite(ld_recon_vec)
    valid_count = int(valid.sum())
    if valid_count < 2:
        ld_corr = float("nan")
        print(
            "[evaluate_vae_ld] WARNING: fewer than 2 finite LD entries after "
            "filtering; LD correlation is undefined (NaN)."
        )
    else:
        ld_corr = np.corrcoef(ld_orig_vec[valid], ld_recon_vec[valid])[0, 1]
        print(
            f"[evaluate_vae_ld] Correlation between original and reconstructed r^2: "
            f"{ld_corr:.4f} (using {valid_count} valid pairs)"
        )

    ld_orig_valid = ld_orig_vec[valid]
    ld_recon_valid = ld_recon_vec[valid]

    # -----------------------------
    # Save numeric summary + LD vectors
    # -----------------------------
    summary_path = outdir / "ld_comparison_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"N_individuals_used: {N}\n")
        f.write(f"M_snps_total: {M}\n")
        f.write(f"M_snps_segregating: {len(seg_idx)}\n")
        f.write(f"M_snps_used_for_LD: {m}\n")
        f.write(f"LD_pairs_valid: {valid_count}\n")
        f.write(f"LD_r2_correlation: {ld_corr:.6f}\n")

    np.save(outdir / "ld_orig_vec.npy", ld_orig_vec)
    np.save(outdir / "ld_recon_vec.npy", ld_recon_vec)

    # =========================================================
    #   PLOT 1 — Binned LD reconstruction curve
    #   (saved as ld_scatter_orig_vs_recon.png for Snakemake)
    # =========================================================
    bins = np.linspace(0.0, 1.0, 21)  # 20 bins
    digitized = np.digitize(ld_orig_valid, bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    mean_recon = []
    for i in range(1, len(bins)):
        mask = digitized == i
        if np.any(mask):
            mean_recon.append(ld_recon_valid[mask].mean())
        else:
            mean_recon.append(np.nan)
    mean_recon = np.array(mean_recon)

    plt.figure(figsize=(6, 5))
    plt.plot(bin_centers, mean_recon, marker="o")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.xlabel("Original r² (bin centers)")
    plt.ylabel("Mean reconstructed r²")
    plt.title(f"Binned LD reconstruction curve\ncorr = {ld_corr:.3f}")
    plt.legend()
    plt.tight_layout()
    # Name kept to match Snakemake output
    plt.savefig(outdir / "ld_scatter_orig_vs_recon.png", dpi=150)
    plt.close()

    # =========================================================
    #   PLOT 2 — 1D histogram of LD reconstruction errors
    # =========================================================
    diff = ld_recon_valid - ld_orig_valid

    plt.figure(figsize=(6, 4))
    plt.hist(diff, bins=50, density=True)
    plt.xlabel("r²_recon - r²_orig")
    plt.ylabel("Density")
    plt.title("Distribution of LD (r²) differences")
    plt.tight_layout()
    plt.savefig(outdir / "ld_difference_hist.png", dpi=150)
    plt.close()

    print(
        "[evaluate_vae_ld] Wrote summary, binned curve "
        "and histogram to", outdir
    )


if __name__ == "__main__":
    main()
