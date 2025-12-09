#!/usr/bin/env python3
import numpy as np
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# make src importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.genotype_vae import GenotypeVAE


def load_scaled_geno(geno_path, meta_path):
    """
    Load genotypes and meta, ensure dosages in [0,2], then scale to [0,1] by /2.
    """
    geno = np.load(geno_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    g_min = float(np.nanmin(geno))
    g_max = float(np.nanmax(geno))
    print(f"Raw genotype stats: min={g_min}, max={g_max}")

    # undo any prior [0,1] scaling if needed
    if g_max <= 1.0 + 1e-6:
        print("Detected genotypes scaled to [0,1]; rescaling back to ~{0,1,2} by *2.")
        geno = np.clip(geno, 0.0, 1.0) * 2.0
        g_min = float(np.nanmin(geno))
        g_max = float(np.nanmax(geno))
        print(f"Post-rescale genotype stats: min={g_min}, max={g_max}")

    # clip garbage
    if g_min < 0.0 or g_max > 2.0:
        bad = np.sum((geno < 0.0) | (geno > 2.0))
        print(f"Clipping {bad} entries outside [0,2].")
        geno = np.clip(geno, 0.0, 2.0)
        g_min = float(np.nanmin(geno))
        g_max = float(np.nanmax(geno))
        print(f"Post-clip genotype stats: min={g_min}, max={g_max}")

    geno_dosage = geno.astype(np.float32)
    geno_scaled = geno_dosage / 2.0  # 0, 0.5, 1
    print(
        f"Scaled genotype stats: min={geno_scaled.min():.4f}, "
        f"max={geno_scaled.max():.4f}"
    )
    return geno_scaled, np.array(meta["population"])


def main():
    geno_path = "experiments/out_of_africa/genotypes/all_individuals.npy"
    meta_path = "experiments/out_of_africa/genotypes/meta.pkl"

    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    maf_threshold = 0.05

    # KL weight for this debug run
    BETA = 0.1   # try 0.1 first; later you can change to 0.0 or 1.0


    geno_scaled, pops = load_scaled_geno(geno_path, meta_path)  # (N, M_full)
    N, M_full = geno_scaled.shape
    print(f"\nLoaded genotype matrix with shape: N={N}, M={M_full}")

    # -------- SNP subsetting --------
    SUBSET_SNPS = 5000
    SUBSET_MODE = "contiguous"

    if SUBSET_SNPS is not None and SUBSET_SNPS < M_full:
        if SUBSET_MODE == "contiguous":
            start = 0
            end = start + SUBSET_SNPS
            snp_idx = np.arange(start, end)
            print(
                f"Subsetting SNPs (contiguous): using SNPs [{start}:{end}) "
                f"out of {M_full} total."
            )
        else:
            rng = np.random.default_rng(0)
            snp_idx = np.sort(rng.choice(M_full, size=SUBSET_SNPS, replace=False))
            print(
                f"Subsetting SNPs (random): using {SUBSET_SNPS} of {M_full} total."
            )
        geno_scaled = geno_scaled[:, snp_idx]
        _, M = geno_scaled.shape
        print(f"Genotype shape after subsetting: N={N}, M={M}")
    else:
        M = M_full
        snp_idx = np.arange(M_full)
        print("No SNP subsetting applied; using all SNPs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- pick one batch to overfit ----
    batch_size = 64
    X_batch_full = torch.tensor(
        geno_scaled[:batch_size], dtype=torch.float32, device=device
    )  # (B, M)
    input_dim = X_batch_full.shape[1]
    print(f"Overfitting batch of size {batch_size}, input_dim = {input_dim}")

    # ---- MAF computation ----
    X_np = X_batch_full.detach().cpu().numpy()  # (B, M)
    allele_freq = X_np.mean(axis=0)
    maf = np.minimum(allele_freq, 1.0 - allele_freq)
    maf_mask = maf >= maf_threshold
    n_kept = int(maf_mask.sum())
    frac_kept = n_kept / maf.size

    print(
        f"MAF filter: threshold={maf_threshold:.3f}, "
        f"kept {n_kept} / {maf.size} SNPs ({frac_kept:.2%})."
    )

    if n_kept > 0:
        X_batch_maf = torch.tensor(
            X_np[:, maf_mask], dtype=torch.float32, device=device
        )
    else:
        X_batch_maf = None

    # ---- baselines ----
    with torch.no_grad():
        mean_vec_all = X_batch_full.mean(dim=0, keepdim=True)
        baseline_mse_all = F.mse_loss(
            mean_vec_all.expand_as(X_batch_full), X_batch_full
        ).item()

        if X_batch_maf is not None:
            mean_vec_maf = X_batch_maf.mean(dim=0, keepdim=True)
            baseline_mse_maf = F.mse_loss(
                mean_vec_maf.expand_as(X_batch_maf), X_batch_maf
            ).item()
        else:
            baseline_mse_maf = None

    print(f"\nBaseline per-entry MSE (ALL SNPs): {baseline_mse_all:.6f}")
    if baseline_mse_maf is not None:
        print(
            f"Baseline per-entry MSE (MAF ≥ {maf_threshold:.2f}): "
            f"{baseline_mse_maf:.6f}"
        )
    print()

    # ---- model: GenotypeVAE in deterministic mode (no sampling, beta=0) ----
    model = GenotypeVAE(
        input_dim=input_dim,
        width=1024,
        depth=1,
        latent_dim=64,
        beta=0.0,               # still no KL
        deterministic_latent=False   # turn sampling back on
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    # ---- training loop (VAE loss: binomial NLL + beta * KL) ----
    n_epochs = 1000
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # forward pass
        recon_p, mu, logvar = model(X_batch_full)

        # VAE loss: Binomial(2, p) NLL + BETA * KL
        loss, recon_nll, kl = model.loss_function(
            recon_p,
            X_batch_full,  # note: scaled; loss_function handles scaling internally
            mu,
            logvar,
            beta=BETA,
        )

        loss.backward()
        optimizer.step()

        # logging every 50 epochs (and epoch 1)
        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                recon_full, mu_full, logvar_full = model(X_batch_full)

                # For comparison to all previous runs, still compute MSE
                mse_all = F.mse_loss(recon_full, X_batch_full).item()

                if X_batch_maf is not None:
                    recon_maf = recon_full[:, maf_mask]
                    mse_maf = F.mse_loss(recon_maf, X_batch_maf).item()

                    # recompute NLL + KL for logging (on full batch)
                    _, recon_nll_log, kl_log = model.loss_function(
                        recon_full, X_batch_full, mu_full, logvar_full, beta=BETA
                    )

                    print(
                        f"[epoch {epoch:04d}] "
                        f"loss={loss.item():.6f} "
                        f"(recon_nll={recon_nll_log.item():.6f}, KL={kl_log.item():.6f}, beta={BETA}) | "
                        f"MSE_all={mse_all:.6f} (baseline {baseline_mse_all:.6f}) | "
                        f"MSE_maf={mse_maf:.6f} (baseline {baseline_mse_maf:.6f})"
                    )
                else:
                    _, recon_nll_log, kl_log = model.loss_function(
                        recon_full, X_batch_full, mu_full, logvar_full, beta=BETA
                    )
                    print(
                        f"[epoch {epoch:04d}] "
                        f"loss={loss.item():.6f} "
                        f"(recon_nll={recon_nll_log.item():.6f}, KL={kl_log.item():.6f}, beta={BETA}) | "
                        f"MSE_all={mse_all:.6f} (baseline {baseline_mse_all:.6f})"
                    )

    # ---- final metrics ----
    model.eval()
    with torch.no_grad():
        recon_full, _, _ = model(X_batch_full)
    recon_np = recon_full.cpu().numpy()

    per_entry_mse_all = np.mean((recon_np - X_np) ** 2)
    print(f"\nFinal per-entry MSE (ALL SNPs, scaled): {per_entry_mse_all:.6f}")
    print(f"Baseline MSE (ALL SNPs):                {baseline_mse_all:.6f}")

    close_all = np.abs(recon_np - X_np) < 0.1
    frac_close_all = close_all.mean()
    print(f"Fraction of entries within 0.1 (ALL SNPs): {frac_close_all:.4f}")

    if X_batch_maf is not None:
        X_np_maf = X_np[:, maf_mask]
        recon_np_maf = recon_np[:, maf_mask]

        per_entry_mse_maf = np.mean((recon_np_maf - X_np_maf) ** 2)
        print(
            f"\nFinal per-entry MSE (MAF ≥ {maf_threshold:.2f}, scaled): "
            f"{per_entry_mse_maf:.6f}"
        )
        print(
            f"Baseline MSE (MAF ≥ {maf_threshold:.2f}): "
            f"{baseline_mse_maf:.6f}"
        )

        close_maf = np.abs(recon_np_maf - X_np_maf) < 0.1
        frac_close_maf = close_maf.mean()
        print(
            f"Fraction of entries within 0.1 "
            f"(MAF ≥ {maf_threshold:.2f}): {frac_close_maf:.4f}"
        )


if __name__ == "__main__":
    main()
