# snakemake_scripts/train_genotype_vae.py

import argparse
import csv
import json
import numpy as np
import pickle
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt

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
    if len(idx) == 0:
        # Empty dataset
        return DataLoader(TensorDataset(base_ds.tensors[0][:0]), batch_size=batch_size)
    subset = Subset(base_ds, idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0

    for batch in loader:
        x = batch[0].to(device)
        bs = x.size(0)
        n_samples += bs

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, recon_loss, kl_loss = model.loss_function(
            recon, x, mu, logvar, beta=beta
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_kl += kl_loss.item() * bs

    if n_samples == 0:
        return 0.0, 0.0, 0.0

    return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples


def eval_epoch(model, loader, device, beta):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            bs = x.size(0)
            n_samples += bs

            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = model.loss_function(
                recon, x, mu, logvar, beta=beta
            )

            total_loss += loss.item() * bs
            total_recon += recon_loss.item() * bs
            total_kl += kl_loss.item() * bs

    if n_samples == 0:
        return 0.0, 0.0, 0.0

    return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genotype", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--outdir", required=True)

    # model size
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)

    # training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--epochs-tune", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)

    # data splits
    p.add_argument("--tune-fraction", type=float, default=0.2)
    p.add_argument("--val-fraction", type=float, default=0.1)

    # beta grid (for hyperparameter search)
    p.add_argument(
        "--beta-grid",
        type=str,
        default="0.0,0.001,0.01,0.1,1.0",
        help="Comma-separated list of beta values to try.",
    )

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------
    geno = np.load(args.genotype)  # (individuals, SNPs)
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    pops = np.array(meta["population"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = torch.tensor(geno, dtype=torch.float32)
    base_ds = TensorDataset(X)
    N = X.shape[0]

    # -------------------------------------------------------------
    # Make train / tune / val splits (by indices)
    # -------------------------------------------------------------
    train_idx, tune_idx, val_idx = make_splits(
        N, args.tune_fraction, args.val_fraction, seed=args.seed
    )
    print(
        f"Split sizes -> train: {len(train_idx)}, "
        f"tune: {len(tune_idx)}, val: {len(val_idx)}"
    )

    train_loader_tune = make_loader(
        base_ds, train_idx, args.batch_size, shuffle=True
    )
    tune_loader = make_loader(
        base_ds, tune_idx, args.batch_size, shuffle=False
    )

    # -------------------------------------------------------------
    # Hyperparameter search over beta on the tuning set
    # -------------------------------------------------------------
    beta_grid = [float(x) for x in args.beta_grid.split(",") if x.strip() != ""]
    beta_results = []

    print(f"Hyperparameter search over beta values: {beta_grid}")

    for beta in beta_grid:
        print(f"\n=== Tuning for beta={beta} ===")
        model = GenotypeVAE(
            input_dim=geno.shape[1],
            width=args.hidden_dim,
            depth=args.depth,
            latent_dim=args.latent_dim,
            beta=beta,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_tune_loss = float("inf")

        for epoch in range(args.epochs_tune):
            train_loss, _, _ = train_epoch(
                model, train_loader_tune, optimizer, device, beta
            )
            tune_loss, _, _ = eval_epoch(
                model, tune_loader, device, beta
            )

            if tune_loss < best_tune_loss:
                best_tune_loss = tune_loss

            print(
                f"[tune][beta={beta:.4g}][epoch={epoch+1:03d}] "
                f"train={train_loss:.4f} | tune={tune_loss:.4f}"
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
                "tune_fraction": args.tune_fraction,
                "val_fraction": args.val_fraction,
            },
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final training with best beta
    #   - train on (train + tune)
    #   - validate on val
    # -------------------------------------------------------------
    # merge train + tune indices for final training
    final_train_idx = np.concatenate([train_idx, tune_idx])
    final_train_loader = make_loader(
        base_ds, final_train_idx, args.batch_size, shuffle=True
    )
    final_val_loader = make_loader(
        base_ds, val_idx, args.batch_size, shuffle=False
    )

    model = GenotypeVAE(
        input_dim=geno.shape[1],
        width=args.hidden_dim,
        depth=args.depth,
        latent_dim=args.latent_dim,
        beta=best_beta,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        train_loss, train_rec, train_kl = train_epoch(
            model, final_train_loader, optimizer, device, beta=best_beta
        )
        val_loss, val_rec, val_kl = eval_epoch(
            model, final_val_loader, device, beta=best_beta
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"[final][epoch={epoch+1:03d}] "
            f"Train={train_loss:.4f} (rec={train_rec:.4f}, KL={train_kl:.4f}) | "
            f"Val={val_loss:.4f} (rec={val_rec:.4f}, KL={val_kl:.4f})"
        )

    # -------------------------------------------------------------
    # Save loss curves (CSV + PNG)
    # -------------------------------------------------------------
    loss_csv = outdir / "vae_losses.csv"
    with loss_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, va) in enumerate(
            zip(history["train_loss"], history["val_loss"]), start=1
        ):
            writer.writerow([i, tr, va])

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
    X_full = X.to(device)
    with torch.no_grad():
        mu, logvar = model.encode(X_full)
    latent = mu.cpu().numpy()

    plot_latent_pca(
        latent,
        pops,
        outdir / "latent_pca_by_population.png",
    )

    # ---------------------
    # Save reconstructions for LD diagnostics
    # ---------------------
    model.eval()
    X = torch.tensor(geno, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon, mu, logvar = model(X)
    recon_np = recon.cpu().numpy()
    np.save(outdir / "recon_all.npy", recon_np)



if __name__ == "__main__":
    main()
