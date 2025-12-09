#!/usr/bin/env python3
import numpy as np
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# make src importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SimpleAutoencoder(nn.Module):
    """
    Deterministic autoencoder: no stochastic z, no KL, no dropout.
    Just to test whether we can *memorize* a single batch.
    """
    def __init__(self, input_dim: int, width: int = 512, depth: int = 3, latent_dim: int = 64):
        super().__init__()
        layers = []
        d_in = input_dim

        # encoder
        for _ in range(depth):
            layers.append(nn.Linear(d_in, width))
            layers.append(nn.ReLU())
            d_in = width
        self.encoder = nn.Sequential(*layers)
        self.z_layer = nn.Linear(d_in, latent_dim)

        # decoder
        dec_layers = []
        d_in = latent_dim
        for _ in range(depth):
            dec_layers.append(nn.Linear(d_in, width))
            dec_layers.append(nn.ReLU())
            d_in = width
        dec_layers.append(nn.Linear(d_in, input_dim))
        dec_layers.append(nn.Sigmoid())  # inputs are in [0,1]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        h = self.encoder(x)
        z = self.z_layer(h)
        recon = self.decoder(z)
        return recon


def load_scaled_geno(geno_path, meta_path):
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

    # clip garbage
    if g_min < 0.0 or g_max > 2.0:
        bad = np.sum((geno < 0.0) | (geno > 2.0))
        print(f"Clipping {bad} entries outside [0,2].")
        geno = np.clip(geno, 0.0, 2.0)

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
        print(f"Baseline per-entry MSE (MAF ≥ {maf_threshold:.2f}): {baseline_mse_maf:.6f}")
    print()

    # ---- model: simple deterministic autoencoder ----
    model = SimpleAutoencoder(
        input_dim=input_dim,
        width=1024,      # wider
        depth=1,         # shallower
        latent_dim=1024  # no serious bottleneck
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---- training loop ----
    n_epochs = 1000
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        recon = model(X_batch_full)
        loss = F.mse_loss(recon, X_batch_full)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                recon_full = model(X_batch_full)
                mse_all = F.mse_loss(recon_full, X_batch_full).item()

                if X_batch_maf is not None:
                    recon_maf = recon_full[:, maf_mask]
                    mse_maf = F.mse_loss(recon_maf, X_batch_maf).item()
                    print(
                        f"[epoch {epoch:04d}] MSE_all={mse_all:.6f} "
                        f"(baseline {baseline_mse_all:.6f}) | "
                        f"MSE_maf={mse_maf:.6f} "
                        f"(baseline {baseline_mse_maf:.6f})"
                    )
                else:
                    print(
                        f"[epoch {epoch:04d}] MSE_all={mse_all:.6f} "
                        f"(baseline {baseline_mse_all:.6f})"
                    )

    # ---- final metrics ----
    model.eval()
    with torch.no_grad():
        recon_full = model(X_batch_full)
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
