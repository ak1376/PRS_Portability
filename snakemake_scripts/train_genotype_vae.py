#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import yaml  # NEW: for YAML config

# --- make sure we can import src.* no matter where Snakemake runs from ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.genotype_vae import GenotypeVAE
from src.plotting_helpers import plot_latent_pca


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def make_splits(N, tune_fraction, val_fraction, seed=42):
    """
    Return (train_idx, tune_idx, val_idx) as index arrays.
    Fractions are w.r.t. total N.
    """
    if tune_fraction + val_fraction >= 1.0:
        raise ValueError("tune_fraction + val_fraction must be < 1.0")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)

    n_tune = int(N * tune_fraction)
    n_val = int(N * val_fraction)
    n_train = N - n_tune - n_val

    train_idx = perm[:n_train]
    tune_idx = perm[n_train:n_train + n_tune]
    val_idx = perm[n_train + n_tune:]

    return train_idx, tune_idx, val_idx


def make_loader(base_ds, idx, batch_size, shuffle):
    """
    Build a DataLoader from a TensorDataset and an index array.
    """
    if len(idx) == 0:
        # Empty dataset
        return DataLoader(
            TensorDataset(base_ds.tensors[0][:0]),
            batch_size=batch_size,
            shuffle=False,
        )
    subset = Subset(base_ds, idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def train_interleaved_epoch(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    beta,
    phase,
    epoch,
    batch_csv_path=None,
):
    """
    Interleaved train + val per batch.

    For each *training* batch:
      1) Do a train step (backprop) on that batch.
      2) Immediately evaluate on one validation batch (no grad).

    If batch_csv_path is not None, append one row per batch to that CSV:
      phase, epoch, batch_idx, split, loss, recon, kl, mse

    Returns epoch-averaged:
        train_loss, train_recon, train_kl, train_mse,
        val_loss,   val_recon,   val_kl,   val_mse
    """
    model.train()
    train_loss_sum = train_recon_sum = train_kl_sum = 0.0
    val_loss_sum = val_recon_sum = val_kl_sum = 0.0
    train_mse_sum = 0.0
    val_mse_sum = 0.0
    n_train = 0
    n_val = 0

    val_iter = iter(val_loader)
    batch_idx = 0

    csv_writer = None
    f_csv = None
    if batch_csv_path is not None:
        f_csv = batch_csv_path.open("a", newline="")
        csv_writer = csv.writer(f_csv)

    try:
        for batch in train_loader:
            x_train = batch[0].to(device)  # scaled genotypes ∈ [0,1]
            bs = x_train.size(0)
            n_train += bs

            # ---- train step ----
            optimizer.zero_grad()
            recon, mu, logvar = model(x_train)
            loss, recon_loss, kl_loss = model.loss_function(
                recon, x_train, mu, logvar, beta=beta
            )
            mse = F.mse_loss(recon, x_train, reduction="mean")

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * bs
            train_recon_sum += recon_loss.item() * bs
            train_kl_sum += kl_loss.item() * bs
            train_mse_sum += mse.item() * bs

            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        phase,
                        epoch,
                        batch_idx,
                        "train",
                        loss.item(),
                        recon_loss.item(),
                        kl_loss.item(),
                        mse.item(),
                    ]
                )

            # ---- one validation batch (no grad) ----
            try:
                vbatch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                try:
                    vbatch = next(val_iter)
                except StopIteration:
                    # empty val_loader
                    batch_idx += 1
                    continue

            x_val = vbatch[0].to(device)
            vbs = x_val.size(0)
            if vbs > 0:
                model.eval()
                with torch.no_grad():
                    v_recon, v_mu, v_logvar = model(x_val)
                    v_loss, v_recon_loss, v_kl_loss = model.loss_function(
                        v_recon, x_val, v_mu, v_logvar, beta=beta
                    )
                    v_mse = F.mse_loss(v_recon, x_val, reduction="mean")

                model.train()

                val_loss_sum += v_loss.item() * vbs
                val_recon_sum += v_recon_loss.item() * vbs
                val_kl_sum += v_kl_loss.item() * vbs
                val_mse_sum += v_mse.item() * vbs
                n_val += vbs

                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            phase,
                            epoch,
                            batch_idx,
                            "val",
                            v_loss.item(),
                            v_recon_loss.item(),
                            v_kl_loss.item(),
                            v_mse.item(),
                        ]
                    )

            batch_idx += 1
    finally:
        if f_csv is not None:
            f_csv.close()

    # Aggregate to means “per individual”
    if n_train == 0:
        mean_train_loss = mean_train_recon = mean_train_kl = mean_train_mse = 0.0
    else:
        mean_train_loss = train_loss_sum / n_train
        mean_train_recon = train_recon_sum / n_train
        mean_train_kl = train_kl_sum / n_train
        mean_train_mse = train_mse_sum / n_train

    if n_val == 0:
        mean_val_loss = mean_val_recon = mean_val_kl = mean_val_mse = 0.0
    else:
        mean_val_loss = val_loss_sum / n_val
        mean_val_recon = val_recon_sum / n_val
        mean_val_kl = val_kl_sum / n_val
        mean_val_mse = val_mse_sum / n_val

    return (
        mean_train_loss,
        mean_train_recon,
        mean_train_kl,
        mean_train_mse,
        mean_val_loss,
        mean_val_recon,
        mean_val_kl,
        mean_val_mse,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genotype", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--model-config",
        required=True,
        help="YAML file with 'model' and 'training' sections.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Load YAML config
    # -------------------------------------------------------------
    with open(args.model_config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    latent_dim = int(model_cfg.get("latent_dim", 32))
    hidden_dim = int(model_cfg.get("hidden_dim", 512))
    depth = int(model_cfg.get("depth", 6))
    # If you later add more knobs (activation, batchnorm, dropout), read them here.

    batch_size = int(train_cfg.get("batch_size", 128))
    epochs = int(train_cfg.get("epochs", 100))
    epochs_tune = int(train_cfg.get("epochs_tune", 40))
    lr = float(train_cfg.get("lr", 1e-4))
    tune_fraction = float(train_cfg.get("tune_fraction", 0.2))
    val_fraction = float(train_cfg.get("val_fraction", 0.1))
    seed = int(train_cfg.get("seed", 42))

    raw_beta_grid = train_cfg.get("beta_grid", [0.0, 0.001, 0.01, 0.1, 1.0])
    if isinstance(raw_beta_grid, str):
        beta_grid = [float(x) for x in raw_beta_grid.split(",") if x.strip() != ""]
    else:
        beta_grid = [float(x) for x in raw_beta_grid]

    print("=== Loaded VAE config ===")
    print("Model:", model_cfg)
    print("Training:", train_cfg)
    print(f"Using beta_grid: {beta_grid}")

    # -------------------------------------------------------------
    # Reproducibility-ish
    # -------------------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare CSV for per-batch logging (tuning + final)
    batch_csv_path = outdir / "vae_batch_losses.csv"
    with batch_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phase",  # "tune" or "final"
                "epoch",  # 1-based epoch index
                "batch_idx",  # 0,1,2,...
                "split",  # "train" or "val"
                "loss",
                "recon",
                "kl",
                "mse",
            ]
        )

    # -------------------------------------------------------------
    # Load data: genotypes + metadata
    #   Expect diploid genotypes ~ {0,1,2}.
    #   We always internally scale to [0,1] by /2 for training,
    #   mirroring popVAE's 0, 0.5, 1 representation.
    # -------------------------------------------------------------
    geno = np.load(args.genotype)  # shape: (individuals, SNPs)
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    pops = np.array(meta["population"])

    g_min = float(np.nanmin(geno))
    g_max = float(np.nanmax(geno))
    print(f"Raw genotype stats: min={g_min}, max={g_max}")

    # If things are already in [0,1], assume upstream scaling and bring back to ~{0,1,2}
    if g_max <= 1.0 + 1e-6:
        print("Detected genotypes scaled to [0,1]; rescaling back to ~{0,1,2} by *2.")
        geno = np.clip(geno, 0.0, 1.0) * 2.0
        g_min = float(np.nanmin(geno))
        g_max = float(np.nanmax(geno))
        print(f"Post-rescale genotype stats: min={g_min}, max={g_max}")
    else:
        print("Assuming raw diploid genotypes (0,1,2) or close.")

    # Clip any stray garbage (e.g. rare 3/4 from upstream bugs)
    if g_min < 0.0 or g_max > 2.0:
        bad = np.sum((geno < 0.0) | (geno > 2.0))
        frac_bad = bad / geno.size
        print(
            f"Warning: found {bad} genotype entries ({frac_bad:.4%}) "
            "outside [0,2]. Clipping them into [0,2]."
        )
        geno = np.clip(geno, 0.0, 2.0)
        g_min = float(np.nanmin(geno))
        g_max = float(np.nanmax(geno))
        print(f"Post-clip genotype stats: min={g_min}, max={g_max}")

    # Define dosage and scaled versions:
    # - geno_dosage: 0/1/2 (used for interpretation / LD)
    # - geno_scaled: in [0,1] via /2 (used for training)
    geno_dosage = geno.astype(np.float32)
    geno_scaled = geno_dosage / 2.0
    print(
        f"Scaled genotype stats (dosage/2): "
        f"min={float(geno_scaled.min()):.4f}, max={float(geno_scaled.max()):.4f}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TensorDataset uses scaled genotypes for training (targets in [0,1])
    X_scaled = torch.tensor(geno_scaled, dtype=torch.float32)
    base_ds = TensorDataset(X_scaled)
    N = X_scaled.shape[0]

    # -------------------------------------------------------------
    # Make train / tune / val splits (by indices)
    # -------------------------------------------------------------
    train_idx, tune_idx, val_idx = make_splits(
        N, tune_fraction, val_fraction, seed=seed
    )
    print(
        f"Split sizes -> train: {len(train_idx)}, "
        f"tune: {len(tune_idx)}, val: {len(val_idx)}"
    )

    train_loader_tune = make_loader(base_ds, train_idx, batch_size, shuffle=True)
    tune_loader = make_loader(base_ds, tune_idx, batch_size, shuffle=False)

    # -------------------------------------------------------------
    # Hyperparameter search over beta on the tuning set
    # -------------------------------------------------------------
    beta_results = []
    print(f"Hyperparameter search over beta values: {beta_grid}")

    for beta in beta_grid:
        print(f"\n=== Tuning for beta={beta} ===")
        model = GenotypeVAE(
            input_dim=geno_scaled.shape[1],
            width=hidden_dim,
            depth=depth,
            latent_dim=latent_dim,
            beta=beta,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        best_tune_loss = float("inf")

        for epoch in range(epochs_tune):
            (
                train_loss,
                train_rec,
                train_kl,
                train_mse,
                tune_loss,
                tune_rec,
                tune_kl,
                tune_mse,
            ) = train_interleaved_epoch(
                model,
                train_loader_tune,
                tune_loader,
                optimizer,
                device,
                beta,
                phase="tune",
                epoch=epoch + 1,
                batch_csv_path=batch_csv_path,
            )

            if tune_loss < best_tune_loss:
                best_tune_loss = tune_loss

            print(
                f"[tune][beta={beta:.4g}][epoch={epoch+1:03d}] "
                f"train_loss={train_loss:.4f}, train_rec={train_rec:.4f}, "
                f"train_KL={train_kl:.4f}, train_MSE={train_mse:.6f} | "
                f"tune_loss={tune_loss:.4f}, tune_rec={tune_rec:.4f}, "
                f"tune_KL={tune_kl:.4f}, tune_MSE={tune_mse:.6f}"
            )

        beta_results.append({"beta": beta, "best_tune_loss": best_tune_loss})

    # pick best beta
    best = min(beta_results, key=lambda d: d["best_tune_loss"])
    best_beta = best["beta"]
    print(f"\n=== Best beta: {best_beta} (tune_loss={best['best_tune_loss']:.4f}) ===")

    # save tuning results
    with (outdir / "vae_beta_search.json").open("w") as f:
        json.dump(
            {
                "beta_results": beta_results,
                "best_beta": best_beta,
                "tune_fraction": tune_fraction,
                "val_fraction": val_fraction,
            },
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final training with best beta
    #   - train on (train + tune)
    #   - validate on val
    # -------------------------------------------------------------
    final_train_idx = np.concatenate([train_idx, tune_idx])
    final_train_loader = make_loader(base_ds, final_train_idx, batch_size, shuffle=True)
    final_val_loader = make_loader(base_ds, val_idx, batch_size, shuffle=False)

    model = GenotypeVAE(
        input_dim=geno_scaled.shape[1],
        width=hidden_dim,
        depth=depth,
        latent_dim=latent_dim,
        beta=best_beta,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "train_mse": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
        "val_mse": [],
    }

    for epoch in range(epochs):
        (
            train_loss,
            train_rec,
            train_kl,
            train_mse,
            val_loss,
            val_rec,
            val_kl,
            val_mse,
        ) = train_interleaved_epoch(
            model,
            final_train_loader,
            final_val_loader,
            optimizer,
            device,
            beta=best_beta,
            phase="final",
            epoch=epoch + 1,
            batch_csv_path=batch_csv_path,
        )

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_rec)
        history["train_kl"].append(train_kl)
        history["train_mse"].append(train_mse)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_rec)
        history["val_kl"].append(val_kl)
        history["val_mse"].append(val_mse)

        print(
            f"[final][epoch={epoch+1:03d}] "
            f"Train: loss={train_loss:.4f}, rec={train_rec:.4f}, KL={train_kl:.4f}, MSE={train_mse:.6f} | "
            f"Val: loss={val_loss:.4f}, rec={val_rec:.4f}, KL={val_kl:.4f}, MSE={val_mse:.6f}"
        )

    # -------------------------------------------------------------
    # Save loss curves (CSV + PNG)
    # -------------------------------------------------------------
    loss_csv = outdir / "vae_losses.csv"
    with loss_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_recon",
                "train_kl",
                "train_mse",
                "val_loss",
                "val_recon",
                "val_kl",
                "val_mse",
            ]
        )
        for i in range(len(history["train_loss"])):
            writer.writerow(
                [
                    i + 1,
                    history["train_loss"][i],
                    history["train_recon"][i],
                    history["train_kl"][i],
                    history["train_mse"][i],
                    history["val_loss"][i],
                    history["val_recon"][i],
                    history["val_kl"][i],
                    history["val_mse"][i],
                ]
            )

    plt.figure(figsize=(8, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"VAE Training Loss Curves (best beta={best_beta})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "vae_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------
    torch.save(model.state_dict(), outdir / "genotype_vae.pt")

    # -------------------------------------------------------------
    # Latent PCA plot using full dataset (best beta)
    # -------------------------------------------------------------
    model.eval()
    X_full = X_scaled.to(device)  # scaled genotypes in [0,1]
    with torch.no_grad():
        mu, logvar = model.encode(X_full)
    latent = mu.cpu().numpy()

    plot_latent_pca(
        latent,
        pops,
        outdir / "latent_pca_by_population.png",
    )

    # -------------------------------------------------------------
    # Save reconstructions for LD diagnostics
    #   - Input is scaled genotypes X_scaled ∈ [0,1]
    #   - Decoder outputs recon_scaled ∈ [0,1]
    #   - We rescale to 0/1/2 dosages by *2 for LD evaluation.
    # -------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        recon_scaled, mu, logvar = model(X_full)  # recon_scaled ∈ [0,1]
    recon_scaled_np = recon_scaled.cpu().numpy()

    recon_dosage_np = recon_scaled_np * 2.0
    np.save(outdir / "recon_all.npy", recon_dosage_np)
    # Optionally also save scaled:
    # np.save(outdir / "recon_all_scaled.npy", recon_scaled_np)


if __name__ == "__main__":
    main()
