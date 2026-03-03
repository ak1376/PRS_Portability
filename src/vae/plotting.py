#!/usr/bin/env python3
"""
src/vae/plotting.py

Minimal VAE diagnostics (no heatmaps).

What it does:
- Reads Lightning CSVLogger metrics.csv from logdir/version_*/metrics.csv
- Writes ONLY the meaningful plots for masked inpainting:
    Epoch:
      * loss_epoch.png
      * recon_masked_epoch.png
      * recon_unmasked_epoch.png
      * masked_over_unmasked_epoch.png
      * ratio_masked_over_nomask_epoch.png   (PRIMARY benchmark)
      * recon_nomask_all_epoch.png           (baseline clean recon)
    Steps (optional but handy when debugging training dynamics):
      * loss_steps.png
      * recon_masked_steps.png
      * recon_unmasked_steps.png
      * ratio_masked_over_nomask_steps.png
- Computes recon on genotype arrays and writes recon_summary.txt with:
      mse/mae for model and MAF baseline for CEU_train/CEU_val/YRI_target (or ALL)

Assumptions:
- Training writes:
    outdir/logs/version_*/metrics.csv
    outdir/hparams.resolved.yaml
    outdir/checkpoints/best.ckpt
- LitVAE wraps a .model returning (recon, mu, logvar) or recon directly.
"""

from __future__ import annotations

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


# ============================================================
# Locate and load metrics.csv
# ============================================================
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


def mse_masked_numpy(X: np.ndarray, R: np.ndarray, mask: np.ndarray) -> float:
    """
    X, R: (N,L)
    mask: (N,L) boolean, True = evaluate here
    """
    if mask.sum() == 0:
        return float("nan")
    diff2 = (R - X) ** 2
    return float(diff2[mask].mean())


def mae_masked_numpy(X: np.ndarray, R: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    diff = np.abs(R - X)
    return float(diff[mask].mean())

# ============================================================
# Metric extraction helpers
# ============================================================
def _last_per_epoch(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
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
    return None if s.empty else s


def _series_with_fallbacks(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
    for c in candidates:
        s = _last_per_epoch(df, c)
        if s is not None:
            return s
    return None


def _extract_steps(df: pd.DataFrame, col: str, max_points: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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


# ============================================================
# Plot helpers
# ============================================================
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

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_epoch_ratio(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
    train_num: list[str],
    train_den: list[str],
    val_num: list[str],
    val_den: list[str],
    target_num: list[str],
    target_den: list[str],
) -> None:
    def safe_ratio(num: Optional[pd.Series], den: Optional[pd.Series]) -> Optional[pd.Series]:
        if num is None or den is None:
            return None
        idx = num.index.intersection(den.index)
        if len(idx) == 0:
            return None
        r = (num.loc[idx] / den.loc[idx]).astype(float)
        r = r[np.isfinite(r.to_numpy())]
        return None if r.empty else r

    tr = safe_ratio(_series_with_fallbacks(df, train_num), _series_with_fallbacks(df, train_den))
    va = safe_ratio(_series_with_fallbacks(df, val_num), _series_with_fallbacks(df, val_den))
    tg = safe_ratio(_series_with_fallbacks(df, target_num), _series_with_fallbacks(df, target_den))

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
    ax.set_ylabel("ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
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

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_step_ratio(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
    train_num: str,
    train_den: str,
    val_num_candidates: list[str],
    val_den_candidates: list[str],
    target_num_candidates: list[str],
    target_den_candidates: list[str],
    max_points: int,
) -> None:
    def safe_ratio_steps(numden: Optional[Tuple[np.ndarray, np.ndarray]], denden: Optional[Tuple[np.ndarray, np.ndarray]]):
        if numden is None or denden is None:
            return None
        # Align by nearest indices after sorting (they should share steps but CSV can be sparse)
        s1, v1 = numden
        s2, v2 = denden
        if s1.size == 0 or s2.size == 0:
            return None
        # Intersect on exact step values
        common = np.intersect1d(s1, s2)
        if common.size == 0:
            return None
        m1 = np.isin(s1, common)
        m2 = np.isin(s2, common)
        r = v1[m1] / (v2[m2] + 1e-12)
        return common, r

    tr_num = _extract_steps(df, train_num, max_points=max_points)
    tr_den = _extract_steps(df, train_den, max_points=max_points)
    tr = safe_ratio_steps(tr_num, tr_den)

    va_num = None
    for c in val_num_candidates:
        va_num = _extract_steps(df, c, max_points=max_points)
        if va_num is not None:
            break
    va_den = None
    for c in val_den_candidates:
        va_den = _extract_steps(df, c, max_points=max_points)
        if va_den is not None:
            break
    va = safe_ratio_steps(va_num, va_den)

    tg_num = None
    for c in target_num_candidates:
        tg_num = _extract_steps(df, c, max_points=max_points)
        if tg_num is not None:
            break
    tg_den = None
    for c in target_den_candidates:
        tg_den = _extract_steps(df, c, max_points=max_points)
        if tg_den is not None:
            break
    tg = safe_ratio_steps(tg_num, tg_den)

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
    ax.set_ylabel("ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ============================================================
# Public plotting API (MINIMAL)
# ============================================================
def plot_epoch_curves(df: pd.DataFrame, outdir: Path) -> None:
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


def plot_masking_epoch_curves(df: pd.DataFrame, outdir: Path) -> None:
    """
    Minimal set for evaluating masking. These are the only ones I'd keep.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    _plot_epoch_triplet(
        df,
        outdir / "recon_masked_epoch.png",
        title="Recon on MASKED SNPs (epoch)",
        ylabel="recon_masked",
        train_candidates=["train/recon_masked_epoch"],
        val_candidates=["val/recon_masked_epoch/dataloader_idx_0", "val/recon_masked_epoch"],
        target_candidates=["target/recon_masked_epoch/dataloader_idx_1", "target/recon_masked_epoch"],
    )

    _plot_epoch_triplet(
        df,
        outdir / "recon_unmasked_epoch.png",
        title="Recon on UNMASKED SNPs (epoch)",
        ylabel="recon_unmasked",
        train_candidates=["train/recon_unmasked_epoch"],
        val_candidates=["val/recon_unmasked_epoch/dataloader_idx_0", "val/recon_unmasked_epoch"],
        target_candidates=["target/recon_unmasked_epoch/dataloader_idx_1", "target/recon_unmasked_epoch"],
    )

    _plot_epoch_ratio(
        df,
        outdir / "masked_over_unmasked_epoch.png",
        title="MASKED / UNMASKED recon (epoch)",
        train_num=["train/recon_masked_epoch"],
        train_den=["train/recon_unmasked_epoch"],
        val_num=["val/recon_masked_epoch/dataloader_idx_0", "val/recon_masked_epoch"],
        val_den=["val/recon_unmasked_epoch/dataloader_idx_0", "val/recon_unmasked_epoch"],
        target_num=["target/recon_masked_epoch/dataloader_idx_1", "target/recon_masked_epoch"],
        target_den=["target/recon_unmasked_epoch/dataloader_idx_1", "target/recon_unmasked_epoch"],
    )

    _plot_epoch_triplet(
        df,
        outdir / "recon_nomask_all_epoch.png",
        title="No-mask recon MSE on ALL SNPs (epoch) — clean baseline pass",
        ylabel="recon_nomask_all",
        train_candidates=["train/recon_nomask_all_epoch"],
        val_candidates=["val/recon_nomask_all_epoch/dataloader_idx_0", "val/recon_nomask_all_epoch"],
        target_candidates=["target/recon_nomask_all_epoch/dataloader_idx_1", "target/recon_nomask_all_epoch"],
    )

    # PRIMARY benchmark: should go DOWN with learning (and ideally < 1 at some point)
    _plot_epoch_triplet(
        df,
        outdir / "ratio_masked_over_nomask_epoch.png",
        title="Masked / No-mask recon (epoch) — PRIMARY benchmark",
        ylabel="ratio_masked_over_nomask",
        train_candidates=["train/ratio_masked_over_nomask_epoch"],
        val_candidates=["val/ratio_masked_over_nomask_epoch/dataloader_idx_0", "val/ratio_masked_over_nomask_epoch"],
        target_candidates=["target/ratio_masked_over_nomask_epoch/dataloader_idx_1", "target/ratio_masked_over_nomask_epoch"],
    )


def plot_step_curves(df: pd.DataFrame, outdir: Path, max_points: int = 5000) -> None:
    """
    Keep steps super minimal: 1 loss plot + 1 ratio plot.
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

    _plot_step_ratio(
        df,
        outdir / "ratio_masked_over_nomask_steps.png",
        title="Masked / No-mask recon (steps) — PRIMARY benchmark",
        train_num="train/recon_masked_step",
        train_den="train/recon_nomask_all_step",
        val_num_candidates=["val/recon_masked_step/dataloader_idx_0", "val/recon_masked_step"],
        val_den_candidates=["val/recon_nomask_all_step/dataloader_idx_0", "val/recon_nomask_all_step"],
        target_num_candidates=["target/recon_masked_step/dataloader_idx_1", "target/recon_masked_step"],
        target_den_candidates=["target/recon_nomask_all_step/dataloader_idx_1", "target/recon_nomask_all_step"],
        max_points=max_points,
    )


# ============================================================
# Stable checkpoint loading
# ============================================================
def load_cfg_from_resolved_yaml(resolved_yaml: Path) -> VAEConfig:
    d = yaml.safe_load(resolved_yaml.read_text())
    data = d.get("data", {}) or {}
    model = d.get("model", {}) or {}
    training = d.get("training", {}) or {}
    masking = d.get("masking", {}) or {}

    input_len = int(data.get("input_len"))

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
        mask_enabled=bool(masking.get("enabled", False)),
        mask_block_len=int(masking.get("block_len", 0)),
        mask_fill_value=str(masking.get("fill_value", "mean")),
        weight_masked=float(masking.get("weight_masked", 1.0)),
        weight_unmasked=float(masking.get("weight_unmasked", 0.0)),
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
    outs: list[torch.Tensor] = []

    for i in range(0, Xt.shape[0], batch_size):
        xb = Xt[i : i + batch_size].to(device)
        out = model(xb)
        recon = out[0] if isinstance(out, (tuple, list)) else out
        outs.append(recon.detach().cpu())

    return torch.cat(outs, dim=0).numpy()

@torch.no_grad()
def compute_recon_masked_eval(
    lit: LitVAE,
    X: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    stage: str,          # "train" | "val" | "target"
    epoch: int,          # use 0 for diagnostics
    base_seed: int,      # cfg.seed
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      recon: (N,L) model output when fed masked inputs
      mask:  (N,L) bool numpy array of which positions were masked
    """
    lit.eval().to(device)
    model = lit  # use LitVAE utilities for masking

    Xt = torch.from_numpy(X).float()
    recons: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    stage_salt = 0 if stage == "train" else (1 if stage == "val" else 2)

    for i in range(0, Xt.shape[0], batch_size):
        xb = Xt[i : i + batch_size].to(device)
        if xb.dim() == 3:
            xb = xb.squeeze(1)
        B, L = xb.shape

        # recreate the exact seed schedule you used in training
        # (epoch fixed for diagnostics; batch_idx = i//batch_size)
        batch_idx = i // batch_size
        seed = int(base_seed) + 1_000_000 * stage_salt + 10_000 * int(epoch) + int(batch_idx)

        if model._mask_enabled():
            m = model._make_contiguous_mask(B, L, seed=seed)  # (B,L) bool on device
            x_in = model._apply_mask(xb, m)
        else:
            m = torch.zeros((B, L), dtype=torch.bool, device=device)
            x_in = xb

        recon, mu, logvar = model.model(x_in)  # ConvVAE1D forward
        if recon.dim() == 3:
            recon = recon.squeeze(1)

        recons.append(recon.detach().cpu())
        masks.append(m.detach().cpu())

    R = torch.cat(recons, dim=0).numpy()
    M = torch.cat(masks, dim=0).numpy().astype(bool)
    return R, M


# ============================================================
# Baseline: MAF predictor + summary
# ============================================================
def maf_baseline_predict(X_train: np.ndarray) -> np.ndarray:
    p = np.mean(X_train, axis=0) / 2.0
    return (2.0 * p[None, :]).astype(np.float32)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def write_recon_summary(
    outdir: Path,
    splits: Dict[str, np.ndarray],
    recon: Dict[str, np.ndarray],
    maf_pred: Dict[str, np.ndarray],
    masks: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    lines: list[str] = []
    lines.append("# split\tmetric\tmodel\tmaf_baseline\twhere")

    for name in ["CEU_train", "CEU_val", "YRI_target", "ALL"]:
        if name not in splits:
            continue
        X = splits[name]
        R = recon[name]
        B = maf_pred[name]

        # full matrix
        lines.append(f"{name}\tmse\t{_mse(R, X):.6g}\t{_mse(B, X):.6g}\tall")
        lines.append(f"{name}\tmae\t{_mae(R, X):.6g}\t{_mae(B, X):.6g}\tall")

        # masked-only (if available)
        if masks is not None and name in masks:
            M = masks[name]
            lines.append(f"{name}\tmse\t{mse_masked_numpy(X, R, M):.6g}\t{mse_masked_numpy(X, B, M):.6g}\tmasked_only")
            lines.append(f"{name}\tmae\t{mae_masked_numpy(X, R, M):.6g}\t{mae_masked_numpy(X, B, M):.6g}\tmasked_only")
            # useful ratio
            denom = mse_masked_numpy(X, B, M)
            num = mse_masked_numpy(X, R, M)
            if np.isfinite(num) and np.isfinite(denom) and denom > 0:
                lines.append(f"{name}\tratio_mse_model_over_maf\t{(num/denom):.6g}\t-\tmasked_only")

    (outdir / "recon_summary.txt").write_text("\n".join(lines) + "\n")

# ============================================================
# Entrypoint
# ============================================================
@dataclass
class PlotArgs:
    logdir: Path
    checkpoint: Path
    resolved_hparams: Path
    outdir: Path

    train_genotype: Optional[Path] = None
    val_genotype: Optional[Path] = None
    target_genotype: Optional[Path] = None
    genotype: Optional[Path] = None

    batch_size: int = 256
    max_step_points: int = 5000
    device: str = "auto"


def run_diagnostics(a: PlotArgs) -> None:
    """
    Diagnostics:
      1) Plot Lightning curves from metrics.csv
      2) Compute recon + MAF baseline and write recon_summary.txt

    Notes:
      - If you want summary metrics to reflect your *masked-inpainting* objective,
        compute recon using your masked-eval helper (compute_recon_masked_eval) so
        recon is produced from x_in (masked) not from clean x.
      - If you don't have compute_recon_masked_eval, you can swap it to compute_recon(...)
        and set masks=None (but then "masked" metrics in the summary won't be meaningful).
    """
    a.outdir.mkdir(parents=True, exist_ok=True)

    # ---- plots from Lightning logs ----
    df = load_metrics_csv(a.logdir)
    plot_epoch_curves(df, a.outdir)
    plot_masking_epoch_curves(df, a.outdir)
    plot_step_curves(df, a.outdir, max_points=a.max_step_points)

    # ---- load genotype arrays (for recon_summary.txt) ----
    splits: Dict[str, np.ndarray] = {}
    if any(p is not None for p in [a.train_genotype, a.val_genotype, a.target_genotype]):
        if a.train_genotype is None or a.val_genotype is None or a.target_genotype is None:
            raise ValueError("Split-mode requires train_genotype, val_genotype, and target_genotype.")
        splits["CEU_train"] = np.load(a.train_genotype).astype(np.float32)
        splits["CEU_val"] = np.load(a.val_genotype).astype(np.float32)
        splits["YRI_target"] = np.load(a.target_genotype).astype(np.float32)
    else:
        if a.genotype is None:
            raise ValueError("Provide either split-mode genotypes or legacy genotype.")
        splits["ALL"] = np.load(a.genotype).astype(np.float32)

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

    # ---- recon (+ masks for masked-only summary metrics) ----
    recon: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}

    # Deterministic snapshot for diagnostics
    base_seed = int(getattr(cfg, "seed", 0))
    epoch_for_eval = 0

    for name, X in splits.items():
        # Map split name -> stage salt used in LitVAE._shared_step
        stage = "train" if name == "CEU_train" else ("val" if name == "CEU_val" else "target")

        # Preferred: evaluate the same way you train (masked input)
        R, M = compute_recon_masked_eval(
            lit,
            X,
            device=device,
            batch_size=a.batch_size,
            stage=stage,
            epoch=epoch_for_eval,
            base_seed=base_seed,
        )
        recon[name] = R
        masks[name] = M

        # If you *don't* have compute_recon_masked_eval yet, use this fallback:
        # recon[name] = compute_recon(lit, X, device=device, batch_size=a.batch_size)
        # (and later set masks=None)

    # ---- MAF baseline (train-based if possible) ----
    maf_pred: Dict[str, np.ndarray] = {}
    if "CEU_train" in splits:
        base = maf_baseline_predict(splits["CEU_train"])  # expected shape (1,L)
    else:
        base = maf_baseline_predict(next(iter(splits.values())))  # expected shape (1,L)

    for name, X in splits.items():
        if base.ndim != 2 or base.shape[0] != 1 or base.shape[1] != X.shape[1]:
            raise ValueError(f"maf_baseline_predict returned {base.shape}, expected (1, L={X.shape[1]})")
        maf_pred[name] = np.repeat(base, repeats=X.shape[0], axis=0).astype(np.float32)

    # ---- summary ----
    write_recon_summary(
        a.outdir,
        splits=splits,
        recon=recon,
        maf_pred=maf_pred,
        masks=masks,  # change to None if you used the compute_recon fallback
    )