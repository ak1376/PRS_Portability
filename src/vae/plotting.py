#!/usr/bin/env python3
"""
src/vae/plotting.py

Diagnostics:
- Reads Lightning CSVLogger metrics.csv
- Plots epoch and step curves (train/val/target if logged)
- Loads checkpoint using cfg from hparams.resolved.yaml (stable)
- Computes recon diagnostics on genotype matrices
- Adds MAF baseline (computed from discovery_train, applied to all splits)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import yaml

from src.vae.lit_model import LitVAE
from src.vae.model import VAEConfig


# -----------------------------
# Locate and load metrics.csv
# -----------------------------
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


# -----------------------------
# Optional meta parsing (population labels)
# -----------------------------
def load_pickle(p: Path) -> Any:
    import pickle
    with p.open("rb") as f:
        return pickle.load(f)


def extract_population_labels(meta_obj: Any, n: int) -> Optional[np.ndarray]:
    if meta_obj is None:
        return None
    if isinstance(meta_obj, pd.DataFrame):
        for col in ["population", "pop", "pop_id", "population_id"]:
            if col in meta_obj.columns and len(meta_obj) == n:
                return meta_obj[col].astype(str).to_numpy()
    if isinstance(meta_obj, dict):
        for key in ["population", "pop", "pop_id", "population_id"]:
            if key in meta_obj and len(meta_obj[key]) == n:
                return np.asarray(meta_obj[key]).astype(str)
    return None


# -----------------------------
# Helpers: robust metric extraction
# -----------------------------
def _last_per_epoch(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """
    Return last logged value per epoch for a given metric column.
    Works for columns like:
      - train/loss_epoch
      - val/recon_epoch/dataloader_idx_0
      - target/kl_epoch/dataloader_idx_1
    """
    if "epoch" not in df.columns or "step" not in df.columns:
        return None
    if col not in df.columns:
        return None

    d = df.loc[df["epoch"].notna(), ["epoch", "step", col]].dropna()
    if d.empty:
        return None

    d = d.sort_values(["epoch", "step"])
    s = d.groupby("epoch")[col].last()
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s.to_numpy())]
    if s.empty:
        return None
    return s


def _series_with_fallbacks(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
    for c in candidates:
        s = _last_per_epoch(df, c)
        if s is not None:
            return s
    return None


def _extract_steps(df: pd.DataFrame, col: str, max_points: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Return (steps, values) for a step-level metric column.
    """
    if "step" not in df.columns or col not in df.columns:
        return None

    d = df.loc[df["step"].notna(), ["step", col]].dropna()
    if d.empty:
        return None

    steps = pd.to_numeric(d["step"], errors="coerce").to_numpy()
    vals = pd.to_numeric(d[col], errors="coerce").to_numpy()
    m = np.isfinite(steps) & np.isfinite(vals)
    steps, vals = steps[m], vals[m]
    if steps.size == 0:
        return None

    order = np.argsort(steps)
    steps, vals = steps[order], vals[order]

    if steps.size > max_points:
        idx = np.linspace(0, steps.size - 1, max_points, dtype=int)
        steps, vals = steps[idx], vals[idx]

    return steps, vals


def _plot_epoch_triplet(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
    ylabel: str,
    train_candidates: list[str],
    val_candidates: list[str],
    target_candidates: list[str],
) -> None:
    tr = _series_with_fallbacks(df, train_candidates)
    va = _series_with_fallbacks(df, val_candidates)
    tg = _series_with_fallbacks(df, target_candidates)

    if tr is None and va is None and tg is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if tr is not None:
        ax.plot(tr.index.to_numpy(), tr.to_numpy(), label="Train (CEU)", linewidth=2)
    if va is not None:
        ax.plot(va.index.to_numpy(), va.to_numpy(), label="Val (CEU)", linewidth=2)
    if tg is not None:
        ax.plot(tg.index.to_numpy(), tg.to_numpy(), label="Target (YRI)", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_step_triplet(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
    ylabel: str,
    train_col: str,
    val_cols: list[str],
    target_cols: list[str],
    max_points: int,
) -> None:
    tr = _extract_steps(df, train_col, max_points=max_points)
    va = None
    for c in val_cols:
        va = _extract_steps(df, c, max_points=max_points)
        if va is not None:
            break
    tg = None
    for c in target_cols:
        tg = _extract_steps(df, c, max_points=max_points)
        if tg is not None:
            break

    if tr is None and va is None and tg is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    if tr is not None:
        ax.plot(tr[0], tr[1], label="Train (CEU)", linewidth=2)
    if va is not None:
        ax.plot(va[0], va[1], label="Val (CEU)", linewidth=2)
    if tg is not None:
        ax.plot(tg[0], tg[1], label="Target (YRI)", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# Public plotting API used by run_diagnostics
# -----------------------------
def plot_epoch_curves(df: pd.DataFrame, outdir: Path) -> None:
    """
    Produces:
      - loss_epoch.png
      - recon_epoch.png   <-- your requested line plot
      - kl_epoch.png
    """
    outdir.mkdir(parents=True, exist_ok=True)

    _plot_epoch_triplet(
        df,
        outdir / "loss_epoch.png",
        title="Total loss (epoch)",
        ylabel="loss",
        train_candidates=["train/loss_epoch"],
        val_candidates=["val/loss_epoch/dataloader_idx_0", "val/loss_epoch"],
        target_candidates=["target/loss_epoch/dataloader_idx_1", "target/loss_epoch"],
    )

    _plot_epoch_triplet(
        df,
        outdir / "recon_epoch.png",
        title="Reconstruction loss (epoch)",
        ylabel="recon",
        train_candidates=["train/recon_epoch"],
        val_candidates=["val/recon_epoch/dataloader_idx_0", "val/recon_epoch"],
        target_candidates=["target/recon_epoch/dataloader_idx_1", "target/recon_epoch"],
    )

    _plot_epoch_triplet(
        df,
        outdir / "kl_epoch.png",
        title="KL term (epoch)",
        ylabel="kl",
        train_candidates=["train/kl_epoch"],
        val_candidates=["val/kl_epoch/dataloader_idx_0", "val/kl_epoch"],
        target_candidates=["target/kl_epoch/dataloader_idx_1", "target/kl_epoch"],
    )


def plot_step_curves(df: pd.DataFrame, outdir: Path, max_points: int = 5000) -> None:
    """
    Produces:
      - loss_steps.png
      - recon_steps.png
      - kl_steps.png

    Note: step curves will often look "misaligned" across train/val/target because
    those are logged at different times/frequencies, and val/target only happen at epoch boundaries.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    _plot_step_triplet(
        df,
        outdir / "loss_steps.png",
        title="Total loss (steps)",
        ylabel="loss",
        train_col="train/loss_step",
        val_cols=["val/loss_step/dataloader_idx_0", "val/loss_step"],
        target_cols=["target/loss_step/dataloader_idx_1", "target/loss_step"],
        max_points=max_points,
    )

    _plot_step_triplet(
        df,
        outdir / "recon_steps.png",
        title="Reconstruction loss (steps)",
        ylabel="recon",
        train_col="train/recon_step",
        val_cols=["val/recon_step/dataloader_idx_0", "val/recon_step"],
        target_cols=["target/recon_step/dataloader_idx_1", "target/recon_step"],
        max_points=max_points,
    )

    _plot_step_triplet(
        df,
        outdir / "kl_steps.png",
        title="KL term (steps)",
        ylabel="kl",
        train_col="train/kl_step",
        val_cols=["val/kl_step/dataloader_idx_0", "val/kl_step"],
        target_cols=["target/kl_step/dataloader_idx_1", "target/kl_step"],
        max_points=max_points,
    )


# -----------------------------
# Stable checkpoint loading
# -----------------------------
def load_cfg_from_resolved_yaml(resolved_yaml: Path) -> VAEConfig:
    d = yaml.safe_load(resolved_yaml.read_text())
    model = d.get("model", {}) or {}
    training = d.get("training", {}) or {}
    input_len = int((d.get("data", {}) or {}).get("input_len"))

    return VAEConfig(
        input_len=input_len,
        latent_dim=int(model.get("latent_dim", 32)),
        hidden_channels=tuple(int(x) for x in model.get("hidden_channels", [32, 64, 128])),
        kernel_size=int(model.get("kernel_size", 9)),
        stride=int(model.get("stride", 2)),
        padding=int(model.get("padding", 4)),
        use_batchnorm=bool(model.get("use_batchnorm", True)),
        lr=float(training.get("lr", 1e-3)),
        beta=float(training.get("beta", 1.0)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def load_litvae_checkpoint(ckpt: Path, cfg: VAEConfig) -> LitVAE:
    lit = LitVAE(cfg)
    obj = torch.load(ckpt, map_location="cpu")
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    lit.load_state_dict(state, strict=True)
    lit.eval()
    return lit


@torch.no_grad()
def compute_recon(lit: LitVAE, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model = getattr(lit, "model", lit)
    model.eval().to(device)

    Xt = torch.from_numpy(X).float()
    outs = []
    for i in range(0, Xt.shape[0], batch_size):
        xb = Xt[i : i + batch_size].to(device)
        with torch.cuda.amp.autocast(enabled=False):
            out = model(xb)
        recon = out[0] if isinstance(out, (tuple, list)) else out
        outs.append(recon.detach().cpu())
    return torch.cat(outs, dim=0).numpy()


# -----------------------------
# Baseline: MAF predictor (from CEU train only)
# -----------------------------
def maf_baseline_predict(X_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """
    Predict genotype by 2*p where p is allele frequency estimated from training.
    Input X in {0,1,2} (or normalized-but-still-on-genotype-scale if you used divide_by_2, etc.)
    """
    # Estimate p from train: mean genotype / 2
    p = np.mean(X_train, axis=0) / 2.0
    return (2.0 * p[None, :]).astype(np.float32)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def write_recon_summary_splits(
    outdir: Path,
    splits: Dict[str, np.ndarray],
    recon: Dict[str, np.ndarray],
    maf_pred: Dict[str, np.ndarray],
) -> None:
    """
    Writes recon_summary.txt with per-split model and baseline metrics.
    """
    lines = []
    lines.append("# split\tmetric\tmodel\tmaf_baseline")
    for name in ["CEU_train", "CEU_val", "YRI_target"]:
        if name not in splits:
            continue
        X = splits[name]
        R = recon[name]
        B = maf_pred[name]

        if not np.isfinite(R).all():
            lines.append(f"{name}\tWARN\tmodel_recon_has_nan_or_inf\t-")

        lines.append(f"{name}\tmse\t{_mse(R, X):.6g}\t{_mse(B, X):.6g}")
        lines.append(f"{name}\tmae\t{_mae(R, X):.6g}\t{_mae(B, X):.6g}")

    (outdir / "recon_summary.txt").write_text("\n".join(lines) + "\n")


def plot_recon_heatmap(X: np.ndarray, R: np.ndarray, outpath: Path, n_show: int = 32) -> None:
    abs_err = np.abs(R - X)
    chosen = np.arange(min(n_show, X.shape[0]))
    H = abs_err[chosen, :]

    plt.figure(figsize=(10, 4))
    plt.imshow(H, aspect="auto")
    plt.colorbar()
    plt.xlabel("site")
    plt.ylabel("individual")
    plt.title("abs(recon - true)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# Public entrypoint
# -----------------------------
@dataclass
class PlotArgs:
    logdir: Path
    checkpoint: Path
    resolved_hparams: Path
    outdir: Path

    # Split-mode inputs (preferred)
    train_genotype: Optional[Path] = None
    val_genotype: Optional[Path] = None
    target_genotype: Optional[Path] = None

    # Legacy mode input
    genotype: Optional[Path] = None

    # Optional meta (not required for your new split files)
    meta: Optional[Path] = None

    batch_size: int = 256
    max_step_points: int = 5000
    device: str = "auto"


def run_diagnostics(a: PlotArgs) -> None:
    a.outdir.mkdir(parents=True, exist_ok=True)

    # ---- curves from Lightning logs ----
    df = load_metrics_csv(a.logdir)
    plot_epoch_curves(df, a.outdir)
    plot_step_curves(df, a.outdir, max_points=a.max_step_points)

    # ---- choose data mode ----
    splits: Dict[str, np.ndarray] = {}

    if a.train_genotype is not None or a.val_genotype is not None or a.target_genotype is not None:
        if a.train_genotype is None or a.val_genotype is None or a.target_genotype is None:
            raise ValueError("Split-mode requires train_genotype, val_genotype, and target_genotype.")

        splits["CEU_train"] = np.load(a.train_genotype).astype(np.float32)
        splits["CEU_val"] = np.load(a.val_genotype).astype(np.float32)
        splits["YRI_target"] = np.load(a.target_genotype).astype(np.float32)
    else:
        if a.genotype is None:
            raise ValueError("Provide either split-mode genotypes or legacy genotype.")
        X = np.load(a.genotype).astype(np.float32)
        splits["ALL"] = X

    # sanity
    for k, X in splits.items():
        if X.ndim != 2:
            raise ValueError(f"{k} expected shape (N,L), got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError(f"{k} contains non-finite values.")

    # ---- device + model ----
    device = torch.device(
        "cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    )
    cfg = load_cfg_from_resolved_yaml(a.resolved_hparams)
    lit = load_litvae_checkpoint(a.checkpoint, cfg)

    # ---- recon per split ----
    recon: Dict[str, np.ndarray] = {}
    for name, X in splits.items():
        recon[name] = compute_recon(lit, X, device=device, batch_size=a.batch_size)

    # ---- MAF baseline (if we have CEU_train) ----
    maf_pred: Dict[str, np.ndarray] = {}
    if "CEU_train" in splits:
        Xtr = splits["CEU_train"]
        for name, X in splits.items():
            maf_pred[name] = maf_baseline_predict(Xtr, X)
    else:
        # fallback: baseline computed from the same data (not ideal, but avoids crashing in legacy mode)
        X0 = next(iter(splits.values()))
        for name, X in splits.items():
            maf_pred[name] = maf_baseline_predict(X0, X)

    # ---- write summary ----
    if set(splits.keys()) >= {"CEU_train", "CEU_val", "YRI_target"}:
        write_recon_summary_splits(a.outdir, splits=splits, recon=recon, maf_pred=maf_pred)
    else:
        # legacy summary
        lines = ["# split\tmetric\tmodel\tmaf_baseline"]
        for name, X in splits.items():
            R = recon[name]
            B = maf_pred[name]
            lines.append(f"{name}\tmse\t{_mse(R, X):.6g}\t{_mse(B, X):.6g}")
            lines.append(f"{name}\tmae\t{_mae(R, X):.6g}\t{_mae(B, X):.6g}")
        (a.outdir / "recon_summary.txt").write_text("\n".join(lines) + "\n")

    # ---- heatmaps (one per split if present) ----
    if "CEU_train" in splits:
        plot_recon_heatmap(splits["CEU_train"], recon["CEU_train"], a.outdir / "recon_abs_error_heatmap_ceu_train.png")
        plot_recon_heatmap(splits["CEU_val"], recon["CEU_val"], a.outdir / "recon_abs_error_heatmap_ceu_val.png")
        plot_recon_heatmap(splits["YRI_target"], recon["YRI_target"], a.outdir / "recon_abs_error_heatmap_yri_target.png")

        # Keep backwards-compatible filename your Snakefile expects
        shutil_path = a.outdir / "recon_abs_error_heatmap.png"
        # write CEU_val as the default representative
        plot_recon_heatmap(splits["CEU_val"], recon["CEU_val"], shutil_path)
    else:
        # legacy: just one
        name = next(iter(splits.keys()))
        plot_recon_heatmap(splits[name], recon[name], a.outdir / "recon_abs_error_heatmap.png")