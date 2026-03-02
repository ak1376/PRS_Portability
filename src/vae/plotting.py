#!/usr/bin/env python3
"""
src/vae/plotting.py

Heavy diagnostics:
- Reads Lightning CSVLogger metrics.csv
- Plots epoch and step curves
- Loads checkpoint using cfg from hparams.resolved.yaml (stable)
- Computes recon diagnostics on genotype matrix
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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

    candidates = []
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
# Meta parsing (population labels)
# -----------------------------
def load_pickle(p: Path) -> Any:
    import pickle
    with p.open("rb") as f:
        return pickle.load(f)


def extract_population_labels(meta_obj: Any, n: int) -> Optional[np.ndarray]:
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
# Plot helpers
# -----------------------------
def _safe_line(ax, x, y, label: str) -> bool:
    y = np.asarray(y)
    m = np.isfinite(y)
    if m.any():
        ax.plot(np.asarray(x)[m], y[m], label=label, linewidth=2)
        return True
    return False


def plot_epoch_curves(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if "epoch" not in df.columns:
        return
    epoch_df = df[df["epoch"].notna()].copy()
    if epoch_df.empty:
        return

    epoch_df.sort_values(["epoch", "step"], inplace=True)
    grouped = epoch_df.groupby("epoch").last(numeric_only=False).reset_index()
    x = grouped["epoch"].to_numpy()

    for (train_key, val_key, fname, title, ylabel) in [
        ("train/loss_epoch", "val/loss_epoch", "loss_epoch.png", "Loss (epoch)", "loss"),
        ("train/recon_epoch", "val/recon_epoch", "recon_epoch.png", "Recon (epoch)", "recon"),
        ("train/kl_epoch", "val/kl_epoch", "kl_epoch.png", "KL (epoch)", "kl"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ok = False
        if train_key in grouped.columns:
            ok |= _safe_line(ax, x, grouped[train_key].to_numpy(), "Training")
        if val_key in grouped.columns:
            ok |= _safe_line(ax, x, grouped[val_key].to_numpy(), "Validation")

        ax.set_title(title)
        if ok:
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No valid data (NaN/Inf)", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=200)
        plt.close()


def plot_step_curves(df: pd.DataFrame, outdir: Path, max_points: int = 5000, drop_sanity: bool = True) -> None:
    """
    Plot step-wise curves from Lightning CSVLogger.

    Key fix:
    - DO NOT downsample the whole dataframe first (that breaks alignment because train/val
      metrics live on different rows).
    - Instead, extract each (step, metric) series independently, then downsample that series.
    - Optionally drop sanity-check validation points (epoch == -1).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    if "step" not in df.columns:
        return

    step_df = df[df["step"].notna()].copy()

    # Drop Lightning sanity-check points (these are validation-only and happen before training)
    if drop_sanity and "epoch" in step_df.columns:
        # Lightning uses epoch == -1 for sanity-check validation
        step_df = step_df[(step_df["epoch"].isna()) | (step_df["epoch"] >= 0)]

    if step_df.empty:
        return

    step_df.sort_values("step", inplace=True)

    def _extract_series(col: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if col not in step_df.columns:
            return None

        s = step_df[["step", col]].dropna()
        if s.empty:
            return None

        # ensure numeric
        steps = pd.to_numeric(s["step"], errors="coerce").to_numpy()
        vals = pd.to_numeric(s[col], errors="coerce").to_numpy()

        m = np.isfinite(steps) & np.isfinite(vals)
        steps, vals = steps[m], vals[m]
        if steps.size == 0:
            return None

        # Downsample THIS series only
        if steps.size > max_points:
            idx = np.linspace(0, steps.size - 1, max_points, dtype=int)
            steps, vals = steps[idx], vals[idx]

        return steps, vals

    for metric, (tcol, vcol) in {
        "loss": ("train/loss_step", "val/loss_step"),
        "recon": ("train/recon_step", "val/recon_step"),
        "kl": ("train/kl_step", "val/kl_step"),
    }.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        ok = False

        ts = _extract_series(tcol)
        if ts is not None:
            ok |= _safe_line(ax, ts[0], ts[1], "Training")

        vs = _extract_series(vcol)
        if vs is not None:
            ok |= _safe_line(ax, vs[0], vs[1], "Validation")

        ax.set_title(f"{metric.upper()} vs steps")
        if ok:
            ax.set_xlabel("step")
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No valid data (NaN/Inf)",
                ha="center", va="center", transform=ax.transAxes
            )

        plt.tight_layout()
        plt.savefig(outdir / f"{metric}_steps.png", dpi=200)
        plt.close()

# -----------------------------
# Stable checkpoint loading
# -----------------------------
def load_cfg_from_resolved_yaml(resolved_yaml: Path) -> VAEConfig:
    d = yaml.safe_load(resolved_yaml.read_text())
    model = d.get("model", {}) or {}
    training = d.get("training", {}) or {}
    input_len = int((d.get("data", {}) or {}).get("input_len"))

    cfg = VAEConfig(
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
    return cfg


def load_litvae_checkpoint(ckpt: Path, cfg: VAEConfig) -> LitVAE:
    # Create module with correct cfg and load weights
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

    # Important: avoid autocast in eval diagnostics
    for i in range(0, Xt.shape[0], batch_size):
        xb = Xt[i : i + batch_size].to(device)
        with torch.cuda.amp.autocast(enabled=False):
            out = model(xb)
        # support either recon-only or (recon, mu, logvar)
        recon = out[0] if isinstance(out, (tuple, list)) else out
        outs.append(recon.detach().cpu())

    R = torch.cat(outs, dim=0).numpy()
    return R


def write_recon_summary(X: np.ndarray, R: np.ndarray, pops: Optional[np.ndarray], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    def _mse(a, b):
        v = np.mean((a - b) ** 2)
        return v

    def _mae(a, b):
        v = np.mean(np.abs(a - b))
        return v

    lines = []
    if not np.isfinite(R).all():
        lines.append("WARN\tReconstruction contains NaN/Inf (model/ckpt/cfg mismatch or numerical issue)")
    lines.append(f"ALL\tmse\t{_mse(R, X):.6g}")
    lines.append(f"ALL\tmae\t{_mae(R, X):.6g}")

    if pops is not None:
        for u in np.unique(pops):
            m = pops == u
            if m.sum() == 0:
                continue
            lines.append(f"{u}\tmse\t{_mse(R[m], X[m]):.6g}")
            lines.append(f"{u}\tmae\t{_mae(R[m], X[m]):.6g}")

    (outdir / "recon_summary.txt").write_text("\n".join(lines) + "\n")


def plot_recon_heatmap(X: np.ndarray, R: np.ndarray, outdir: Path, n_show: int = 32) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
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
    plt.savefig(outdir / "recon_abs_error_heatmap.png", dpi=200)
    plt.close()


# -----------------------------
# Public entrypoint
# -----------------------------
@dataclass
class PlotArgs:
    logdir: Path
    checkpoint: Path
    resolved_hparams: Path
    genotype: Path
    meta: Path
    outdir: Path
    batch_size: int = 256
    max_step_points: int = 5000
    device: str = "auto"


def run_diagnostics(a: PlotArgs) -> None:
    a.outdir.mkdir(parents=True, exist_ok=True)

    df = load_metrics_csv(a.logdir)
    plot_epoch_curves(df, a.outdir)
    plot_step_curves(df, a.outdir, max_points=a.max_step_points)

    X = np.load(a.genotype).astype(np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected genotype shape (N,L), got {X.shape}")

    meta_obj = load_pickle(a.meta)
    pops = extract_population_labels(meta_obj, X.shape[0])

    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    cfg = load_cfg_from_resolved_yaml(a.resolved_hparams)

    lit = load_litvae_checkpoint(a.checkpoint, cfg)
    R = compute_recon(lit, X, device=device, batch_size=a.batch_size)

    write_recon_summary(X, R, pops, a.outdir)
    plot_recon_heatmap(X, R, a.outdir)

    # Always ensure expected file exists for Snakemake
    if not (a.outdir / "loss_epoch.png").exists():
        (a.outdir / "loss_epoch.png").write_bytes(b"")