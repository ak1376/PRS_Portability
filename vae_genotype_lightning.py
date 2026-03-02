#!/usr/bin/env python3
"""
vae_genotype_lightning.py

Self-contained starter script to:
1) Load a diploid genotype matrix X (n_inds, n_snps) in {0,1,2} from .npy or .trees/.ts
2) Load a meta/phenotype table with columns: ["individual_id", "population"] (and optionally phenotype)
3) Train a simple MLP VAE on DISCOVERY population only (with a discovery train/val split)
4) Evaluate reconstruction on:
   - discovery_val
   - discovery_all
   - target_all
5) Save plots every N epochs:
   - reconstruction heatmaps (true vs recon) on discovery_val and target
   - AF scatter (true p vs recon p) on discovery_val and target
6) Save a JSON summary + preprocessing metadata + kept SNP mask + model checkpoint
7) Save training/validation curves:
   - loss_curves.png (train/total, val/disc_total, val/targ_total)
   - mse_curves.png  (val/disc_mse, val/targ_mse)
   - metrics_epoch.csv

Key Lightning/DDP fixes:
- Multiple val dataloaders cause Lightning to suffix metric names with `/dataloader_idx_X`.
  We avoid that by logging with `add_dataloader_idx=False`.
- Use `sync_dist=True` for epoch-level metrics in DDP so checkpointing sees the reduced value.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

# Matplotlib only (no seaborn)
import matplotlib.pyplot as plt

# Optional: for .trees loading
try:
    import tskit  # type: ignore
except Exception:
    tskit = None


# ============================================================
# Loading
# ============================================================

def load_genotype_matrix(path: str) -> np.ndarray:
    """
    Load genotype matrix from .npy or .trees/.ts.

    Returns:
        X: (num_inds, num_sites) diploid genotype counts in {0,1,2}
    """
    path = str(path)
    if path.endswith(".npy"):
        X = np.load(path)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D genotype matrix, got shape {X.shape}")
        return X

    if path.endswith(".trees") or path.endswith(".ts"):
        if tskit is None:
            raise ImportError("tskit is required to load .trees/.ts. Install tskit or provide .npy.")
        ts = tskit.load(path)
        G_hap = ts.genotype_matrix()  # (sites, samples/nodes), entries in {0,1}

        if ts.num_individuals > 0:
            num_inds = ts.num_individuals
            num_sites = ts.num_sites
            G_ind = np.zeros((num_inds, num_sites), dtype=np.float32)

            # Sum haplotypes per individual (assumes diploid with 2 nodes; works if >2 too)
            for i, ind in enumerate(ts.individuals()):
                nodes = ind.nodes
                if len(nodes) > 0:
                    G_ind[i] = G_hap[:, nodes].sum(axis=1)
            return G_ind

        # If no individuals defined, treat haplotypes as individuals
        return G_hap.T.astype(np.float32)

    raise ValueError(f"Unknown genotype format: {path}")


def load_meta(path: str) -> pd.DataFrame:
    """
    Load meta/phenotype DataFrame from .pkl or .csv.
    Must include at least: population, individual_id (recommended).
    """
    path = str(path)
    if path.endswith(".pkl"):
        df = pd.read_pickle(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unknown meta format: {path}")

    if "population" not in df.columns:
        raise ValueError("Meta file must contain a 'population' column.")
    if "individual_id" in df.columns:
        df = df.sort_values("individual_id").reset_index(drop=True)
    else:
        # assume already aligned with genotype rows
        df = df.reset_index(drop=True)
        df["individual_id"] = np.arange(len(df), dtype=int)

    return df


# ============================================================
# Dataset processing (simple, optional)
# ============================================================

@dataclass
class PreprocessConfig:
    maf_min: float = 0.0          # e.g. 0.01
    drop_monomorphic: bool = True
    standardize: bool = False     # if True: z-score per SNP using discovery TRAIN stats
    seed: int = 0


def compute_af(X: np.ndarray) -> np.ndarray:
    """Allele frequency per SNP, assuming diploid counts in {0,1,2}."""
    denom = 2.0 * X.shape[0]
    return X.sum(axis=0) / max(denom, 1.0)


def filter_snps_by_maf(X: np.ndarray, maf_min: float, drop_monomorphic: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter SNPs by MAF computed on X.
    Returns: (X_filt, keep_mask)
    """
    p = compute_af(X)
    maf = np.minimum(p, 1.0 - p)
    keep = np.ones(X.shape[1], dtype=bool)
    if drop_monomorphic:
        keep &= (p > 0.0) & (p < 1.0)
    if maf_min > 0.0:
        keep &= maf >= maf_min
    return X[:, keep], keep


def standardize_snps(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score per SNP; safe std."""
    std2 = std.copy()
    std2[std2 < 1e-8] = 1.0
    return (X - mean) / std2


# ============================================================
# PyTorch datasets
# ============================================================

class GenotypeDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


# ============================================================
# VAE model (Lightning)
# ============================================================

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: Tuple[int, ...], latent_dim: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(d, latent_dim)
        self.logvar = nn.Linear(d, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden: Tuple[int, ...], output_dim: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        d = latent_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(d, output_dim)  # logits per SNP

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(z))


class GenotypeVAE(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        enc_hidden: Tuple[int, ...] = (512, 256),
        dec_hidden: Tuple[int, ...] = (256, 512),
        latent_dim: int = 32,
        lr: float = 1e-3,
        beta_kl: float = 1.0,
        dropout: float = 0.0,
        use_standardized_input: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = MLPEncoder(input_dim, enc_hidden, latent_dim, dropout=dropout)
        self.decoder = MLPDecoder(latent_dim, dec_hidden, input_dim, dropout=dropout)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar

    def _loss(
        self, x: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruction:
          - If inputs are raw genotypes in {0,1,2}: use BCEWithLogits on x/2
          - If standardized: use MSE (because BCE no longer makes sense)
        """
        if not self.hparams.use_standardized_input:
            x_frac = x / 2.0
            recon = F.binary_cross_entropy_with_logits(logits, x_frac, reduction="sum")
        else:
            recon = F.mse_loss(logits, x, reduction="sum")

        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
        total = recon + self.hparams.beta_kl * kl
        return total, recon, kl

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        logits, mu, logvar = self(x)
        loss, recon, kl = self._loss(x, logits, mu, logvar)

        bsz = x.shape[0]
        # Keep names stable; include epoch aggregation; sync in DDP
        self.log("train/total", loss / bsz, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/recon", recon / bsz, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/kl", kl / bsz, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch
        logits, mu, logvar = self(x)
        loss, recon, kl = self._loss(x, logits, mu, logvar)

        # Recon metric in genotype space:
        if not self.hparams.use_standardized_input:
            x_hat = 2.0 * torch.sigmoid(logits)
        else:
            x_hat = logits

        mse = torch.mean((x_hat - x) ** 2)

        prefix = "disc" if dataloader_idx == 0 else "targ"
        bsz = x.shape[0]

        # IMPORTANT: add_dataloader_idx=False keeps metric keys EXACTLY `val/disc_mse`, etc.
        # sync_dist=True aggregates across GPUs in DDP.
        self.log(f"val/{prefix}_mse", mse,
                 prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, add_dataloader_idx=False)

        self.log(f"val/{prefix}_total", loss / bsz,
                 prog_bar=False, on_step=False, on_epoch=True,
                 sync_dist=True, add_dataloader_idx=False)

        self.log(f"val/{prefix}_recon", recon / bsz,
                 prog_bar=False, on_step=False, on_epoch=True,
                 sync_dist=True, add_dataloader_idx=False)

        self.log(f"val/{prefix}_kl", kl / bsz,
                 prog_bar=False, on_step=False, on_epoch=True,
                 sync_dist=True, add_dataloader_idx=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ============================================================
# Plotting callback (save every N epochs)
# ============================================================

class ReconPlotCallback(pl.Callback):
    """
    Saves plots every `plot_every` epochs:
      - recon heatmap (true vs recon) for a subset of individuals and SNPs
      - AF scatter (true p vs recon p) for the same subset SNPs
    Does it for both discovery_val loader and target loader.
    Only global rank 0 writes files.
    """
    def __init__(
        self,
        outdir: Path,
        plot_every: int = 5,
        n_inds: int = 20,
        n_snps: int = 2000,
        seed: int = 0,
    ):
        super().__init__()
        self.outdir = Path(outdir)
        self.plot_every = plot_every
        self.n_inds = n_inds
        self.n_snps = n_snps
        self.rng = np.random.default_rng(seed)

        (self.outdir / "plots").mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _make_plots_for_loader(self, pl_module: GenotypeVAE, loader: DataLoader, tag: str, epoch: int):
        pl_module.eval()

        batch = next(iter(loader))
        x = batch.to(pl_module.device)

        n_inds = min(self.n_inds, x.shape[0])
        m = x.shape[1]
        n_snps = min(self.n_snps, m)

        snp_idx = self.rng.choice(m, size=n_snps, replace=False)
        x_sub = x[:n_inds, snp_idx]

        logits, _, _ = pl_module(x[:n_inds])

        if not pl_module.hparams.use_standardized_input:
            x_hat = 2.0 * torch.sigmoid(logits)[:, snp_idx]
        else:
            x_hat = logits[:, snp_idx]

        x_np = x_sub.detach().cpu().numpy()
        xh_np = x_hat.detach().cpu().numpy()

        # Heatmaps
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        im1 = ax1.imshow(x_np, aspect="auto")
        ax1.set_title(f"{tag}: True")
        ax1.set_xlabel("SNPs"); ax1.set_ylabel("Individuals")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(1, 2, 2)
        im2 = ax2.imshow(xh_np, aspect="auto")
        ax2.set_title(f"{tag}: Recon")
        ax2.set_xlabel("SNPs"); ax2.set_ylabel("Individuals")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        fig.tight_layout()
        fig.savefig(self.outdir / "plots" / f"recon_{tag}_epoch{epoch:04d}.png", dpi=150)
        plt.close(fig)

        # AF scatter (subset SNPs)
        if not pl_module.hparams.use_standardized_input:
            true_p = x_sub.mean(dim=0) / 2.0
            recon_p = x_hat.mean(dim=0) / 2.0

            tp = true_p.detach().cpu().numpy()
            rp = recon_p.detach().cpu().numpy()

            fig = plt.figure(figsize=(5, 5))
            plt.scatter(tp, rp, s=8, alpha=0.6)
            mx = float(max(tp.max(), rp.max(), 1e-6))
            plt.plot([0, mx], [0, mx], ls="--")
            plt.xlabel("True AF (batch)")
            plt.ylabel("Recon AF (batch)")
            plt.title(f"{tag}: AF reconstruction")
            plt.tight_layout()
            fig.savefig(self.outdir / "plots" / f"af_{tag}_epoch{epoch:04d}.png", dpi=150)
            plt.close(fig)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: GenotypeVAE) -> None:
        if not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch + 1
        if self.plot_every <= 0 or (epoch % self.plot_every != 0):
            return

        if not isinstance(trainer.val_dataloaders, list) or len(trainer.val_dataloaders) < 2:
            return

        disc_loader = trainer.val_dataloaders[0]
        targ_loader = trainer.val_dataloaders[1]

        self._make_plots_for_loader(pl_module, disc_loader, "disc", epoch)
        self._make_plots_for_loader(pl_module, targ_loader, "targ", epoch)


# ============================================================
# Metric curves callback (loss curves + MSE curves)
# ============================================================

class MetricCurvesCallback(pl.Callback):
    """
    Collect epoch-level metrics and save:
      - metrics_epoch.csv
      - loss_curves.png (train/total, val/disc_total, val/targ_total)
      - mse_curves.png  (val/disc_mse, val/targ_mse)
    Only global rank 0 writes files.
    """
    def __init__(self, outdir: Path):
        super().__init__()
        self.outdir = Path(outdir)
        self.rows: List[Dict[str, float]] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        m = trainer.callback_metrics

        def get_float(key: str) -> Optional[float]:
            v = m.get(key, None)
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            try:
                return float(v)
            except Exception:
                return None

        epoch = int(trainer.current_epoch + 1)
        row: Dict[str, float] = {"epoch": float(epoch)}

        # Try both possible keys for epoch-aggregated train metric
        # (Lightning often stores epoch aggregation as "<name>_epoch")
        for k in ["train/total_epoch", "train/total"]:
            val = get_float(k)
            if val is not None:
                row["train/total"] = val
                break

        for k in ["val/disc_total", "val/targ_total", "val/disc_recon", "val/targ_recon", "val/disc_kl", "val/targ_kl",
                  "val/disc_mse", "val/targ_mse"]:
            val = get_float(k)
            if val is not None:
                row[k] = val

        self.rows.append(row)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        if len(self.rows) == 0:
            return

        self.outdir.mkdir(parents=True, exist_ok=True)

        # Write CSV
        csv_path = self.outdir / "metrics_epoch.csv"
        keys = sorted({k for r in self.rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

        def series(key: str) -> Tuple[np.ndarray, np.ndarray]:
            xs, ys = [], []
            for r in self.rows:
                if "epoch" in r and key in r:
                    xs.append(r["epoch"])
                    ys.append(r[key])
            return np.asarray(xs), np.asarray(ys)

        # Loss curves
        fig = plt.figure(figsize=(7, 5))
        for key in ["train/total", "val/disc_total", "val/targ_total"]:
            x, y = series(key)
            if len(x) > 0:
                plt.plot(x, y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (per-individual; your logging units)")
        plt.title("Training / validation loss curves")
        plt.legend()
        plt.tight_layout()
        fig.savefig(self.outdir / "loss_curves.png", dpi=150)
        plt.close(fig)

        # MSE curves
        fig = plt.figure(figsize=(7, 5))
        for key in ["val/disc_mse", "val/targ_mse"]:
            x, y = series(key)
            if len(x) > 0:
                plt.plot(x, y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction MSE (genotype space)")
        plt.title("Validation MSE curves")
        plt.legend()
        plt.tight_layout()
        fig.savefig(self.outdir / "mse_curves.png", dpi=150)
        plt.close(fig)


# ============================================================
# DataModule: train on discovery only, validate on disc-val and target
# ============================================================

class GenotypeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X_disc: np.ndarray,
        X_targ: np.ndarray,
        batch_size: int = 64,
        num_workers: int = 0,
        val_frac: float = 0.2,
        seed: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.X_disc = X_disc
        self.X_targ = X_targ
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.seed = seed
        self.pin_memory = pin_memory

        self.ds_train: Optional[Dataset] = None
        self.ds_val_disc: Optional[Dataset] = None
        self.ds_val_targ: Optional[Dataset] = None
        self._disc_train_idx: Optional[np.ndarray] = None
        self._disc_val_idx: Optional[np.ndarray] = None

    def setup(self, stage: Optional[str] = None) -> None:
        rng = np.random.default_rng(self.seed)
        n = self.X_disc.shape[0]
        perm = rng.permutation(n)
        n_val = int(round(self.val_frac * n))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        self._disc_train_idx = train_idx
        self._disc_val_idx = val_idx

        self.ds_train = GenotypeDataset(self.X_disc[train_idx])
        self.ds_val_disc = GenotypeDataset(self.X_disc[val_idx])
        self.ds_val_targ = GenotypeDataset(self.X_targ)

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> List[DataLoader]:
        assert self.ds_val_disc is not None and self.ds_val_targ is not None
        return [
            DataLoader(self.ds_val_disc, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory),
            DataLoader(self.ds_val_targ, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory),
        ]


# ============================================================
# Evaluation + baselines
# ============================================================

@torch.no_grad()
def recon_mse_raw_genotypes(model: GenotypeVAE, loader: DataLoader, device: torch.device) -> float:
    """Reconstruction MSE in genotype count space (assumes raw genotype input)."""
    model.eval()
    total_se = 0.0
    total_n = 0

    for x in loader:
        x = x.to(device)
        logits, _, _ = model(x)
        x_hat = 2.0 * torch.sigmoid(logits)
        se = torch.sum((x_hat - x) ** 2).item()
        total_se += se
        total_n += x.numel()

    return total_se / max(total_n, 1)


def baseline_af_only_mse(X_disc_train: np.ndarray, X_eval: np.ndarray) -> float:
    """Predict everyone as 2*p_disc_train (AF-only). Return MSE in genotype space."""
    p = compute_af(X_disc_train)
    pred = 2.0 * p[None, :]
    mse = np.mean((pred - X_eval) ** 2)
    return float(mse)


def baseline_pca_mse(X_disc_train: np.ndarray, X_eval: np.ndarray, n_components: int = 32) -> Optional[float]:
    """
    PCA reconstruction baseline (linear autoencoder).
    Uses scikit-learn if available; returns None if not installed.
    """
    try:
        from sklearn.decomposition import PCA  # type: ignore
    except Exception:
        return None

    pca = PCA(n_components=min(n_components, X_disc_train.shape[0], X_disc_train.shape[1]))
    pca.fit(X_disc_train)
    X_eval_recon = pca.inverse_transform(pca.transform(X_eval))
    mse = float(np.mean((X_eval_recon - X_eval) ** 2))
    return mse


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genotype", required=True, help=".npy or .trees/.ts genotype source")
    ap.add_argument("--meta", required=True, help=".pkl or .csv metadata with 'population' column")

    ap.add_argument("--discovery-pop", default="CEU")
    ap.add_argument("--target-pop", default="YRI")
    ap.add_argument("--outdir", required=True)

    # preprocessing
    ap.add_argument("--maf-min", type=float, default=0.0, help="MAF filter based on discovery TRAIN only (recommended)")
    ap.add_argument("--keep-monomorphic", action="store_true", help="If set, do NOT drop monomorphic SNPs")
    ap.add_argument("--standardize", action="store_true", help="Z-score SNPs using discovery TRAIN stats (turns recon into MSE model)")

    # model
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--enc-hidden", type=int, nargs="+", default=[512, 256])
    ap.add_argument("--dec-hidden", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--beta-kl", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)

    # training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)

    # plots
    ap.add_argument("--plot-every", type=int, default=5)
    ap.add_argument("--plot-n-inds", type=int, default=20)
    ap.add_argument("--plot-n-snps", type=int, default=2000)

    # PCA baseline
    ap.add_argument("--pca-components", type=int, default=32)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    # A100 tensor cores: optional speedup
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Repro
    pl.seed_everything(args.seed, workers=True)

    # Load
    X = load_genotype_matrix(args.genotype).astype(np.float32)
    meta = load_meta(args.meta)

    if X.shape[0] != len(meta):
        raise ValueError(f"Row mismatch: X has {X.shape[0]} inds, meta has {len(meta)} rows. Ensure ordering matches.")

    # Split pops
    pop = meta["population"].astype(str).values
    disc_mask = (pop == args.discovery_pop)
    targ_mask = (pop == args.target_pop)

    if disc_mask.sum() == 0:
        raise ValueError(f"No discovery individuals for pop={args.discovery_pop}. Available={np.unique(pop)}")
    if targ_mask.sum() == 0:
        raise ValueError(f"No target individuals for pop={args.target_pop}. Available={np.unique(pop)}")

    X_disc_all = X[disc_mask]
    X_targ_all = X[targ_mask]

    # Discovery train/val split indices now (needed for MAF/standardization computed on TRAIN only)
    rng = np.random.default_rng(args.seed)
    n_disc = X_disc_all.shape[0]
    perm = rng.permutation(n_disc)
    n_val = int(round(args.val_frac * n_disc))
    disc_val_idx = perm[:n_val]
    disc_train_idx = perm[n_val:]

    X_disc_train = X_disc_all[disc_train_idx]
    X_disc_val = X_disc_all[disc_val_idx]

    # SNP filtering based on discovery TRAIN
    drop_mono = not args.keep_monomorphic
    if args.maf_min > 0.0 or drop_mono:
        X_disc_train_f, keep = filter_snps_by_maf(X_disc_train, maf_min=args.maf_min, drop_monomorphic=drop_mono)
        X_disc_val_f = X_disc_val[:, keep]
        X_targ_all_f = X_targ_all[:, keep]
        X_disc_all_f = X_disc_all[:, keep]
    else:
        keep = np.ones(X.shape[1], dtype=bool)
        X_disc_train_f, X_disc_val_f, X_targ_all_f, X_disc_all_f = X_disc_train, X_disc_val, X_targ_all, X_disc_all

    # Optional standardization using discovery TRAIN
    use_standardized = bool(args.standardize)
    if use_standardized:
        mean = X_disc_train_f.mean(axis=0)
        std = X_disc_train_f.std(axis=0, ddof=1)
        X_disc_train_f = standardize_snps(X_disc_train_f, mean, std).astype(np.float32)
        X_disc_val_f = standardize_snps(X_disc_val_f, mean, std).astype(np.float32)
        X_targ_all_f = standardize_snps(X_targ_all_f, mean, std).astype(np.float32)
        X_disc_all_f = standardize_snps(X_disc_all_f, mean, std).astype(np.float32)

        np.save(outdir / "standardize_mean.npy", mean.astype(np.float32))
        np.save(outdir / "standardize_std.npy", std.astype(np.float32))

    # Save preprocessing info
    prep_info = {
        "genotype_path": str(args.genotype),
        "meta_path": str(args.meta),
        "discovery_pop": args.discovery_pop,
        "target_pop": args.target_pop,
        "n_disc_all": int(X_disc_all.shape[0]),
        "n_targ_all": int(X_targ_all.shape[0]),
        "m_snps_raw": int(X.shape[1]),
        "m_snps_kept": int(keep.sum()),
        "maf_min": float(args.maf_min),
        "drop_monomorphic": bool(drop_mono),
        "standardize": bool(use_standardized),
        "val_frac": float(args.val_frac),
        "seed": int(args.seed),
    }
    (outdir / "preprocess.json").write_text(json.dumps(prep_info, indent=2))

    # DataModule uses discovery_all + internal split, and target_all as a second val loader
    dm = GenotypeDataModule(
        X_disc=X_disc_all_f,
        X_targ=X_targ_all_f,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        seed=args.seed,
        pin_memory=True,
    )
    dm.setup()

    # Model
    input_dim = int(X_disc_all_f.shape[1])
    model = GenotypeVAE(
        input_dim=input_dim,
        enc_hidden=tuple(args.enc_hidden),
        dec_hidden=tuple(args.dec_hidden),
        latent_dim=args.latent_dim,
        lr=args.lr,
        beta_kl=args.beta_kl,
        dropout=args.dropout,
        use_standardized_input=use_standardized,
    )

    callbacks: List[pl.Callback] = [
        ReconPlotCallback(
            outdir=outdir,
            plot_every=args.plot_every,
            n_inds=args.plot_n_inds,
            n_snps=args.plot_n_snps,
            seed=args.seed,
        ),
        MetricCurvesCallback(outdir=outdir),
        pl.callbacks.ModelCheckpoint(
            dirpath=str(outdir),
            filename="vae-{epoch:04d}",
            save_top_k=1,
            monitor="val/disc_mse",  # NOTE: exists because we log with add_dataloader_idx=False
            mode="min",
        ),
    ]

    # If you have only a few batches/epoch, lower log interval to see logs each epoch
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=1,
        default_root_dir=str(outdir),
    )

    trainer.fit(model, datamodule=dm)

    # ========================================================
    # Final evaluation + baselines
    # ========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # loaders
    disc_all_loader = DataLoader(GenotypeDataset(X_disc_all_f), batch_size=args.batch_size, shuffle=False)
    disc_val_loader = dm.val_dataloader()[0]
    targ_loader = dm.val_dataloader()[1]

    results: Dict[str, Any] = {}
    results["vae"] = {}

    if not use_standardized:
        results["vae"]["disc_val_mse"] = recon_mse_raw_genotypes(model, disc_val_loader, device)
        results["vae"]["disc_all_mse"] = recon_mse_raw_genotypes(model, disc_all_loader, device)
        results["vae"]["targ_all_mse"] = recon_mse_raw_genotypes(model, targ_loader, device)

        # AF-only baseline (computed from discovery TRAIN on filtered SNPs)
        X_disc_train_raw = X_disc_train_f
        X_disc_val_raw = X_disc_val_f
        X_disc_all_raw = X_disc_all_f
        X_targ_all_raw = X_targ_all_f

        results["baseline_af"] = {
            "disc_val_mse": baseline_af_only_mse(X_disc_train_raw, X_disc_val_raw),
            "disc_all_mse": baseline_af_only_mse(X_disc_train_raw, X_disc_all_raw),
            "targ_all_mse": baseline_af_only_mse(X_disc_train_raw, X_targ_all_raw),
        }

        pca_mse_disc_val = baseline_pca_mse(X_disc_train_raw, X_disc_val_raw, n_components=args.pca_components)
        pca_mse_targ = baseline_pca_mse(X_disc_train_raw, X_targ_all_raw, n_components=args.pca_components)
        if pca_mse_disc_val is not None and pca_mse_targ is not None:
            results["baseline_pca"] = {
                "n_components": int(args.pca_components),
                "disc_val_mse": float(pca_mse_disc_val),
                "targ_all_mse": float(pca_mse_targ),
            }
        else:
            results["baseline_pca"] = {"note": "scikit-learn not installed; PCA baseline skipped."}
    else:
        results["note"] = "Standardized mode uses MSE on standardized inputs; AF plots/baselines are less meaningful."

    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # Save kept SNP mask
    np.save(outdir / "kept_snps_mask.npy", keep.astype(np.bool_))
    print(f"[done] outputs in: {outdir}")


if __name__ == "__main__":
    main()