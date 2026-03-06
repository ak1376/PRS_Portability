#!/usr/bin/env python3
"""
src/vae/plotting.py

What it does (ONLY):
1) Make epoch-mean plots from Lightning CSVLogger metrics.csv
   (and a few diagnostic scatters, plus val/loss LAST and MAX aggregations)

2) Compute recon on genotype arrays using best.ckpt, and write recon_summary.txt
   comparing model vs mean-allele-frequency (MAF) baseline on:
     - CEU_train
     - CEU_val
     - (optional) YRI_target

Notes:
- Plots are computed from "epoch-mean over rows" aggregation, matching your notebook.
- Recon summary defaults to evaluating recon from MASKED inputs if your LitVAE exposes
  the masking helpers used during training; otherwise it falls back to clean recon.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import yaml

from src.vae.lit_model import LitVAE
from src.vae.model import VAEConfig


# =========================
# Reconstruction scatter plot
# =========================
def save_recon_scatter(
    recon_npz: Path,
    outpath: Path,
    *,
    alpha: float = 0.3,
    point_size: float = 5,
    max_points: int = 50000,
) -> dict:
    """
    Create scatter plot of true vs reconstructed values at masked positions.
    Returns dict with stats (corr, mse, n_masked).
    """
    data = np.load(recon_npz)
    x_true = data["x_true"]
    recon = data["recon"]
    mask = data["mask"]

    # Extract values at masked positions
    true_masked = x_true[mask]
    recon_masked = recon[mask]

    n_masked = len(true_masked)
    if n_masked == 0:
        return {"corr": float("nan"), "mse": float("nan"), "n_masked": 0}

    corr = float(np.corrcoef(true_masked, recon_masked)[0, 1])
    mse = float(np.mean((true_masked - recon_masked) ** 2))

    # Subsample if too many points
    if n_masked > max_points:
        idx = np.random.choice(n_masked, max_points, replace=False)
        true_plot = true_masked[idx]
        recon_plot = recon_masked[idx]
    else:
        true_plot = true_masked
        recon_plot = recon_masked

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_plot, recon_plot, alpha=alpha, s=point_size, c="steelblue", edgecolors="none")

    # Add diagonal line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("True value (at masked positions)", fontsize=12)
    ax.set_ylabel("Reconstructed value", fontsize=12)
    ax.set_title(f"VAE Reconstruction at Masked Positions\nCorr={corr:.3f}, MSE={mse:.4f}, N={n_masked:,}", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"corr": corr, "mse": mse, "n_masked": n_masked}


# =========================
# Metrics.csv loading
# =========================
def find_latest_version_dir(lightning_logs_dir: Path) -> Path:
    if not lightning_logs_dir.exists():
        raise FileNotFoundError(f"Missing logdir: {lightning_logs_dir}")

    candidates: list[tuple[int, Path]] = []
    for p in lightning_logs_dir.glob("version_*"):
        m = re.search(r"version_(\d+)", p.name)
        if m:
            candidates.append((int(m.group(1)), p))

    if not candidates:
        raise FileNotFoundError(f"No version_* dirs under: {lightning_logs_dir}")

    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def load_metrics_csv(lightning_logs_dir: Path) -> pd.DataFrame:
    vdir = find_latest_version_dir(lightning_logs_dir)
    metrics_path = vdir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")
    return pd.read_csv(metrics_path)


# =========================
# Epoch aggregation (MATCHES your notebook)
# =========================
def epoch_agg(df: pd.DataFrame, cols: list[str], how: str = "mean") -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame({"epoch": []})

    d = df[["epoch"] + cols].copy()
    d = d.dropna(subset=cols, how="all")
    d = d.dropna(subset=["epoch"])
    if d.empty:
        return pd.DataFrame({"epoch": []})

    g = d.groupby("epoch", as_index=False)

    if how == "mean":
        out = g[cols].mean(numeric_only=True)
    elif how == "median":
        out = g[cols].median(numeric_only=True)
    elif how == "max":
        out = g[cols].max(numeric_only=True)
    elif how == "last":
        # define "last" by row order within each epoch
        # (Lightning logs are typically increasing step, so this behaves as expected)
        out = g.tail(1)[["epoch"] + cols]
    else:
        raise ValueError("how must be one of: mean|median|max|last")

    return out.sort_values("epoch").reset_index(drop=True)


def save_lineplot(
    E: pd.DataFrame,
    outpath: Path,
    y1: str,
    y2: Optional[str] = None,
    *,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    logy: bool = False,
) -> None:
    if E.empty or y1 not in E.columns:
        return
    if y2 is not None and y2 not in E.columns:
        y2 = None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(E["epoch"], E[y1], label=y1)
    if y2 is not None:
        ax.plot(E["epoch"], E[y2], label=y2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel if ylabel else y1)
    ax.set_title(title if title else (f"{y1}" + (f" vs {y2}" if y2 else "")))
    ax.legend()
    ax.grid(True, alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def save_scatter(
    E: pd.DataFrame,
    outpath: Path,
    x: str,
    y: str,
    *,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    if E.empty or x not in E.columns or y not in E.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(E[x], E[y], s=18)
    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def make_notebook_equivalent_plots(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    cols = [
        "train/loss", "val/loss",
        "train/mse_masked", "val/mse_masked",
        "train/mse_clean_all", "val/mse_clean_all",
        "train/mse_corrupt_all", "val/mse_corrupt_all",
        "train/kl", "val/kl",
        "train/mask_frac", "val/mask_frac",
        "train/delta_in_l1", "val/delta_in_l1",
    ]

    E_mean = epoch_agg(df, cols, how="mean")
    E_last = epoch_agg(df, cols, how="last")
    E_max  = epoch_agg(df, cols, how="max")

    # Core curves (mean)
    save_lineplot(E_mean, outdir / "epoch_loss_mean.png",
                  "train/loss", "val/loss", title="Epoch loss (mean over rows)", ylabel="loss")
    save_lineplot(E_mean, outdir / "masked_mse_mean.png",
                  "train/mse_masked", "val/mse_masked", title="Masked MSE (mean over rows)", ylabel="masked MSE")
    save_lineplot(E_mean, outdir / "clean_mse_mean.png",
                  "train/mse_clean_all", "val/mse_clean_all", title="Clean-pass MSE (mean over rows)", ylabel="MSE")

    if "train/mse_corrupt_all" in E_mean.columns and "val/mse_corrupt_all" in E_mean.columns:
        save_lineplot(E_mean, outdir / "corrupt_mse_mean.png",
                      "train/mse_corrupt_all", "val/mse_corrupt_all",
                      title="Corrupt-pass MSE (mean over rows)", ylabel="MSE")

    save_lineplot(E_mean, outdir / "kl_mean_logy.png",
                  "train/kl", "val/kl", title="KL (mean over rows)", ylabel="KL", logy=True)

    save_lineplot(E_mean, outdir / "mask_frac_mean.png",
                  "train/mask_frac", "val/mask_frac", title="Mask fraction (mean over rows)", ylabel="mask_frac")

    if "train/delta_in_l1" in E_mean.columns and "val/delta_in_l1" in E_mean.columns:
        save_lineplot(E_mean, outdir / "delta_in_l1_mean.png",
                      "train/delta_in_l1", "val/delta_in_l1",
                      title="Corruption strength Δ|x_in-x| (mean over rows)", ylabel="delta_in_l1")

    # Diagnostics: spikes due to aggregation
    save_lineplot(E_last, outdir / "val_loss_last.png",
                  "val/loss", None, title="val/loss using LAST row (often spiky)", ylabel="val/loss")
    save_lineplot(E_max, outdir / "val_loss_max.png",
                  "val/loss", None, title="val/loss using MAX row (worst-case row)", ylabel="val/loss")

    # NEW: scatters you asked for
    save_scatter(E_mean, outdir / "scatter_val_masked_vs_clean.png",
                 "val/mse_masked", "val/mse_clean_all",
                 title="Val: masked vs clean MSE",
                 xlabel="val/mse_masked (epoch-mean)",
                 ylabel="val/mse_clean_all (epoch-mean)")

    save_scatter(E_mean, outdir / "scatter_val_kl_vs_loss.png",
                 "val/kl", "val/loss",
                 title="Is val/loss driven by KL?",
                 xlabel="val/kl (epoch-mean)",
                 ylabel="val/loss (epoch-mean)")

    if "val/delta_in_l1" in E_mean.columns and "val/mse_masked" in E_mean.columns:
        save_scatter(E_mean, outdir / "scatter_val_delta_vs_masked.png",
                     "val/delta_in_l1", "val/mse_masked",
                     title="Does harder corruption -> worse val masked MSE?",
                     xlabel="val/delta_in_l1 (epoch-mean)",
                     ylabel="val/mse_masked (epoch-mean)")


# =========================
# Model loading + recon
# =========================
def _as_int(x, default: int) -> int:
    if x is None:
        return default
    if isinstance(x, bool):
        return int(x)
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _as_float(x, default: float) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _as_bool(x, default: bool) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "f", "0", "no", "n", "off"}:
            return False
    return default


def load_cfg_from_resolved_yaml(resolved_yaml: Path) -> VAEConfig:
    d = yaml.safe_load(resolved_yaml.read_text())
    data = d.get("data", {}) or {}
    model = d.get("model", {}) or {}
    training = d.get("training", {}) or {}
    masking = d.get("masking", {}) or {}

    input_len = _as_int(data.get("input_len"), default=0)
    if input_len <= 0:
        raise ValueError(f"data.input_len must be a positive int in {resolved_yaml}")

    hidden = model.get("hidden_channels", [32, 64, 128])
    if hidden is None:
        hidden = [32, 64, 128]
    hidden_channels = tuple(_as_int(x, 0) for x in hidden)
    if any(h <= 0 for h in hidden_channels):
        raise ValueError(f"model.hidden_channels must be positive ints in {resolved_yaml}: got {hidden_channels}")

    # Handle padding - can be None for fully_conv model
    padding_val = model.get("padding", 4)
    if padding_val is not None:
        padding_val = _as_int(padding_val, default=4)

    return VAEConfig(
        input_len=input_len,
        latent_dim=_as_int(model.get("latent_dim"), default=32),
        hidden_channels=hidden_channels,
        kernel_size=_as_int(model.get("kernel_size"), default=9),
        stride=_as_int(model.get("stride"), default=2),
        padding=padding_val,
        use_batchnorm=_as_bool(model.get("use_batchnorm"), default=True),
        model_type=str(model.get("model_type", "conv")),

        lr=_as_float(training.get("lr"), default=1e-3),
        beta=_as_float(training.get("beta"), default=1.0),
        weight_decay=_as_float(training.get("weight_decay"), default=0.0),

        mask_enabled=_as_bool(masking.get("enabled"), default=False),
        mask_block_len=_as_int(masking.get("block_len"), default=0),      # <-- FIXED (None-safe)
        mask_fill_value=str(masking.get("fill_value") or "mean"),         # <-- FIXED (None-safe)
        weight_masked=_as_float(masking.get("weight_masked"), default=1.0),
        weight_unmasked=_as_float(masking.get("weight_unmasked"), default=0.0),
    )

def load_litvae_checkpoint(ckpt: Path, cfg: VAEConfig) -> LitVAE:
    lit = LitVAE(cfg)
    obj = torch.load(ckpt, map_location="cpu")
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    lit.load_state_dict(state, strict=True)
    lit.eval()
    return lit


@torch.no_grad()
def compute_recon_clean(lit: LitVAE, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model = getattr(lit, "model", lit)
    model.eval().to(device)

    Xt = torch.from_numpy(X).float()
    outs: list[torch.Tensor] = []

    for i in range(0, Xt.shape[0], batch_size):
        xb = Xt[i : i + batch_size].to(device)
        out = model(xb)
        recon = out[0] if isinstance(out, (tuple, list)) else out
        if recon.dim() == 3:
            recon = recon.squeeze(1)
        outs.append(recon.detach().cpu())

    return torch.cat(outs, dim=0).numpy()


@torch.no_grad()
def compute_recon_masked_like_training(
    lit: LitVAE,
    X: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    stage: str,      # "train" | "val" | "target"
    epoch: int,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Best-effort masked eval using LitVAE's masking helpers.
    If missing, caller should fall back to clean recon.
    """
    # Require these utilities to exist
    required = ["_mask_enabled", "_make_contiguous_mask", "_apply_mask", "model"]
    for r in required:
        if not hasattr(lit, r):
            raise AttributeError(f"LitVAE missing {r}")

    lit.eval().to(device)

    Xt = torch.from_numpy(X).float()
    recons: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    stage_salt = 0 if stage == "train" else (1 if stage == "val" else 2)

    for i in range(0, Xt.shape[0], batch_size):
        xb = Xt[i : i + batch_size].to(device)
        if xb.dim() == 3:
            xb = xb.squeeze(1)
        B, L = xb.shape

        batch_idx = i // batch_size
        seed = int(base_seed) + 1_000_000 * stage_salt + 10_000 * int(epoch) + int(batch_idx)

        if lit._mask_enabled():
            m = lit._make_contiguous_mask(B, L, seed=seed)  # (B,L) bool
            x_in = lit._apply_mask(xb, m)
        else:
            m = torch.zeros((B, L), dtype=torch.bool, device=device)
            x_in = xb

        out = lit.model(x_in)
        if isinstance(out, (tuple, list)):
            recon = out[0]
        else:
            recon = out
        if recon.dim() == 3:
            recon = recon.squeeze(1)

        recons.append(recon.detach().cpu())
        masks.append(m.detach().cpu())

    R = torch.cat(recons, dim=0).numpy()
    M = torch.cat(masks, dim=0).numpy().astype(bool)
    return R, M


# =========================
# Recon summary (model vs MAF)
# =========================
def maf_baseline_predict_from_train(X_train: np.ndarray) -> np.ndarray:
    p = np.mean(X_train, axis=0) / 2.0  # allele freq
    return (2.0 * p[None, :]).astype(np.float32)  # (1, L)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def mse_masked_numpy(X: np.ndarray, R: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(((R - X) ** 2)[mask].mean())


def mae_masked_numpy(X: np.ndarray, R: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.abs(R - X)[mask].mean())


def write_recon_summary(
    outdir: Path,
    splits: Dict[str, np.ndarray],
    recon: Dict[str, np.ndarray],
    maf_pred: Dict[str, np.ndarray],
    masks: Optional[Dict[str, np.ndarray]],
) -> None:
    lines: list[str] = []
    lines.append("# split\tmetric\tmodel\tmaf_baseline\twhere")

    for name in ["CEU_train", "CEU_val", "YRI_target"]:
        if name not in splits:
            continue
        X = splits[name]
        R = recon[name]
        B = maf_pred[name]

        lines.append(f"{name}\tmse\t{_mse(R, X):.6g}\t{_mse(B, X):.6g}\tall")
        lines.append(f"{name}\tmae\t{_mae(R, X):.6g}\t{_mae(B, X):.6g}\tall")

        if masks is not None and name in masks:
            M = masks[name]
            lines.append(f"{name}\tmse\t{mse_masked_numpy(X, R, M):.6g}\t{mse_masked_numpy(X, B, M):.6g}\tmasked_only")
            lines.append(f"{name}\tmae\t{mae_masked_numpy(X, R, M):.6g}\t{mae_masked_numpy(X, B, M):.6g}\tmasked_only")

    (outdir / "recon_summary.txt").write_text("\n".join(lines) + "\n")


# =========================
# Entrypoint
# =========================
@dataclass
class Args:
    logdir: Path
    checkpoint: Path
    resolved_hparams: Path
    outdir: Path
    train_genotype: Path
    val_genotype: Path
    target_genotype: Optional[Path]
    batch_size: int
    device: str
    recon_dir: Optional[Path] = None  # directory with {split}_recon.npz files


def run(a: Args) -> None:
    a.outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1) notebook-equivalent plots ----
    df = load_metrics_csv(a.logdir)
    make_notebook_equivalent_plots(df, a.outdir / "plots")

    # ---- 2) recon summary (model vs MAF baseline) ----
    splits: Dict[str, np.ndarray] = {
        "CEU_train": np.load(a.train_genotype).astype(np.float32),
        "CEU_val": np.load(a.val_genotype).astype(np.float32),
    }
    if a.target_genotype is not None:
        splits["YRI_target"] = np.load(a.target_genotype).astype(np.float32)

    for k, X in splits.items():
        if X.ndim != 2:
            raise ValueError(f"{k} expected shape (N,L), got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError(f"{k} contains non-finite values.")

    device = torch.device(
        "cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    )

    cfg = load_cfg_from_resolved_yaml(a.resolved_hparams)
    lit = load_litvae_checkpoint(a.checkpoint, cfg)

    base_seed = int(getattr(cfg, "seed", 0))
    epoch_for_eval = 0

    recon: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    masks_available = True

    for name, X in splits.items():
        stage = "train" if name == "CEU_train" else ("val" if name == "CEU_val" else "target")

        # Preferred: masked eval (same style as training), if available
        try:
            R, M = compute_recon_masked_like_training(
                lit, X,
                device=device,
                batch_size=a.batch_size,
                stage=stage,
                epoch=epoch_for_eval,
                base_seed=base_seed,
            )
            recon[name] = R
            masks[name] = M
        except Exception:
            # Fallback: clean recon (still useful for MAF baseline comparison)
            masks_available = False
            recon[name] = compute_recon_clean(lit, X, device=device, batch_size=a.batch_size)

    # MAF baseline from CEU_train
    base = maf_baseline_predict_from_train(splits["CEU_train"])  # (1,L)
    maf_pred: Dict[str, np.ndarray] = {}
    for name, X in splits.items():
        maf_pred[name] = np.repeat(base, repeats=X.shape[0], axis=0).astype(np.float32)

    write_recon_summary(
        a.outdir,
        splits=splits,
        recon=recon,
        maf_pred=maf_pred,
        masks=(masks if masks_available else None),
    )

    # ---- 3) reconstruction scatter plots from saved recon artifacts ----
    if a.recon_dir is not None and a.recon_dir.exists():
        scatter_stats = {}
        for split in ["train", "val", "target"]:
            npz_path = a.recon_dir / f"{split}_recon.npz"
            if npz_path.exists():
                out_png = a.outdir / "plots" / f"recon_scatter_{split}.png"
                stats = save_recon_scatter(npz_path, out_png)
                scatter_stats[split] = stats
                print(f"Saved {out_png} (corr={stats['corr']:.4f}, mse={stats['mse']:.4f})")

        # Write scatter stats summary
        if scatter_stats:
            lines = ["split\tcorr\tmse\tn_masked"]
            for split, s in scatter_stats.items():
                lines.append(f"{split}\t{s['corr']:.6f}\t{s['mse']:.6f}\t{s['n_masked']}")
            (a.outdir / "recon_scatter_stats.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--logdir", type=Path, required=True, help=".../logs (contains version_*/metrics.csv)")
    ap.add_argument("--checkpoint", type=Path, required=True, help=".../checkpoints/best.ckpt")
    ap.add_argument("--resolved-hparams", type=Path, required=True, help=".../hparams.resolved.yaml")
    ap.add_argument("--outdir", type=Path, required=True)

    ap.add_argument("--train-genotype", type=Path, required=True, help="discovery_train.npy")
    ap.add_argument("--val-genotype", type=Path, required=True, help="discovery_val.npy")
    ap.add_argument("--target-genotype", type=Path, default=None, help="optional target.npy (e.g. YRI)")

    ap.add_argument("--recon-dir", type=Path, default=None, help="directory with {split}_recon.npz files")

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = ap.parse_args()

    run(
        Args(
            logdir=args.logdir,
            checkpoint=args.checkpoint,
            resolved_hparams=args.resolved_hparams,
            outdir=args.outdir,
            train_genotype=args.train_genotype,
            val_genotype=args.val_genotype,
            target_genotype=args.target_genotype,
            batch_size=args.batch_size,
            device=args.device,
            recon_dir=args.recon_dir,
        )
    )


if __name__ == "__main__":
    main()