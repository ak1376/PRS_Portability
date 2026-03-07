#!/usr/bin/env python3
"""
src/vae/plotting.py

Diagnostics for genotype-classification VAE runs.

What it does:
1) Loads Lightning CSVLogger metrics.csv and makes epoch-aggregated plots
2) Reads saved reconstruction artifacts from recon/{train,val,target}_recon.npz
3) Writes a reconstruction summary against simple train-set baselines
4) Saves masked-position dosage scatter plots and optional confusion matrices

Supported recon npz layouts:

Newer format:
- x_true    : true genotype array, shape (N, L), values in {0,1,2}
- x_masked  : masked/corrupted model input, shape (N, L) [optional for plotting]
- mask      : boolean mask, shape (N, L), True = masked position
- pred      : predicted genotype classes, shape (N, L)
- prob_0    : predicted P(genotype=0), shape (N, L)
- prob_1    : predicted P(genotype=1), shape (N, L)
- prob_2    : predicted P(genotype=2), shape (N, L)

Older format:
- x_true    : true genotype array, shape (N, L)
- mask      : boolean mask, shape (N, L)
- recon     : one of:
    * continuous dosage prediction, shape (N, L)
    * class logits/probs, shape (N, 3, L)
    * class logits/probs, shape (N, L, 3)

Notes:
- Some output filenames remain intentionally backward-compatible with the Snakefile:
    masked_mse_mean.png  -> actually plots masked CE
    clean_mse_mean.png  -> actually plots clean-pass CE
- recon_scatter_{split}.png plots expected dosage vs true genotype at masked sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# Args
# =========================================================

@dataclass
class Args:
    logdir: Path
    outdir: Path
    train_genotype: Path
    val_genotype: Optional[Path] = None
    target_genotype: Optional[Path] = None
    recon_dir: Optional[Path] = None
    save_confusion: bool = True


# =========================================================
# metrics.csv loading
# =========================================================

def find_latest_version_dir(lightning_logs_dir: Path) -> Path:
    if not lightning_logs_dir.exists():
        raise FileNotFoundError(f"Missing logdir: {lightning_logs_dir}")

    candidates: list[tuple[int, Path]] = []
    for p in lightning_logs_dir.glob("version_*"):
        m = re.fullmatch(r"version_(\d+)", p.name)
        if m:
            candidates.append((int(m.group(1)), p))

    if not candidates:
        raise FileNotFoundError(f"No version_* dirs under: {lightning_logs_dir}")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_metrics_csv(lightning_logs_dir: Path) -> pd.DataFrame:
    vdir = find_latest_version_dir(lightning_logs_dir)
    metrics_path = vdir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")
    return pd.read_csv(metrics_path)


# =========================================================
# artifact loading helpers
# =========================================================

def _load_np_array(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D genotype array at {path}, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {path}")
    return arr


def _softmax_np(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _recon_to_pred_and_probs(recon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible loader for older recon archives.

    Supports:
      - (N, L): continuous dosage prediction
      - (N, 3, L): class logits/probabilities
      - (N, L, 3): class logits/probabilities

    Returns:
      pred:  (N, L) int64
      probs: (N, 3, L) float32
    """
    recon = np.asarray(recon)

    # Case 1: continuous dosage predictions
    if recon.ndim == 2:
        dosage = recon.astype(np.float32)
        pred = np.clip(np.rint(dosage), 0, 2).astype(np.int64)

        # Pseudo-probabilities for downstream compatibility.
        probs = np.zeros((dosage.shape[0], 3, dosage.shape[1]), dtype=np.float32)
        for c in (0, 1, 2):
            probs[:, c, :] = (pred == c).astype(np.float32)

        return pred, probs

    # Case 2: (N, 3, L)
    if recon.ndim == 3 and recon.shape[1] == 3:
        probs = recon.astype(np.float32)

        sums = probs.sum(axis=1, keepdims=True)
        if np.any(probs < 0) or not np.allclose(sums, 1.0, atol=1e-3):
            probs = _softmax_np(probs, axis=1)

        pred = np.argmax(probs, axis=1).astype(np.int64)
        return pred, probs.astype(np.float32)

    # Case 3: (N, L, 3)
    if recon.ndim == 3 and recon.shape[-1] == 3:
        probs = np.moveaxis(recon.astype(np.float32), -1, 1)  # -> (N, 3, L)

        sums = probs.sum(axis=1, keepdims=True)
        if np.any(probs < 0) or not np.allclose(sums, 1.0, atol=1e-3):
            probs = _softmax_np(probs, axis=1)

        pred = np.argmax(probs, axis=1).astype(np.int64)
        return pred, probs.astype(np.float32)

    raise ValueError(f"Unsupported recon shape: {recon.shape}")


def _load_pred_and_probs(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """
    Supports both:
      newer format:
        pred, prob_0, prob_1, prob_2
      older format:
        recon
    """
    # Newer format
    if all(k in data.files for k in ["pred", "prob_0", "prob_1", "prob_2"]):
        pred = np.asarray(data["pred"], dtype=np.int64)
        prob_0 = np.asarray(data["prob_0"], dtype=np.float32)
        prob_1 = np.asarray(data["prob_1"], dtype=np.float32)
        prob_2 = np.asarray(data["prob_2"], dtype=np.float32)

        if pred.ndim != 2:
            raise ValueError(f"Expected pred shape (N, L), got {pred.shape}")
        if prob_0.shape != pred.shape or prob_1.shape != pred.shape or prob_2.shape != pred.shape:
            raise ValueError(
                "Probability arrays must match pred shape. "
                f"Got pred={pred.shape}, prob_0={prob_0.shape}, prob_1={prob_1.shape}, prob_2={prob_2.shape}"
            )

        probs = np.stack([prob_0, prob_1, prob_2], axis=1)  # (N, 3, L)
        return pred, probs

    # Older format
    if "recon" in data.files:
        return _recon_to_pred_and_probs(data["recon"])

    raise KeyError(
        "Could not find prediction keys in archive. "
        "Expected either ['pred', 'prob_0', 'prob_1', 'prob_2'] or ['recon']. "
        f"Found keys: {list(data.files)}"
    )


def _expected_dosage_from_probs(probs: np.ndarray) -> np.ndarray:
    """
    probs shape: (N, 3, L)
    returns expected dosage in [0, 2], shape (N, L)
    """
    if probs.ndim != 3 or probs.shape[1] != 3:
        raise ValueError(f"Expected probs shape (N, 3, L), got {probs.shape}")
    return probs[:, 1, :] + 2.0 * probs[:, 2, :]


# =========================================================
# epoch aggregation
# =========================================================

def epoch_agg(df: pd.DataFrame, cols: list[str], how: str = "mean") -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols or "epoch" not in df.columns:
        return pd.DataFrame({"epoch": []})

    d = df[["epoch"] + cols].copy()
    d = d.dropna(subset=["epoch"])
    d = d.dropna(subset=cols, how="all")
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
        out = g.tail(1)[["epoch"] + cols].copy()
    else:
        raise ValueError("how must be one of: mean|median|max|last")

    return out.sort_values("epoch").reset_index(drop=True)


# =========================================================
# plotting helpers
# =========================================================

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
        positive = False
        if np.any(E[y1].to_numpy() > 0):
            positive = True
        if y2 is not None and np.any(E[y2].to_numpy() > 0):
            positive = True
        if positive:
            ax.set_yscale("log")

    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel if ylabel is not None else y1)
    ax.set_title(title if title is not None else y1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
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
    ax.set_xlabel(xlabel if xlabel is not None else x)
    ax.set_ylabel(ylabel if ylabel is not None else y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_notebook_equivalent_plots(df: pd.DataFrame, outdir: Path) -> None:
    """
    Keep key filenames compatible with the current Snakefile.

    Backward-compatible filenames:
      - masked_mse_mean.png -> plots masked CE
      - clean_mse_mean.png  -> plots clean-pass CE
    """
    outdir.mkdir(parents=True, exist_ok=True)

    cols = [
        "train/loss", "val/loss",
        "train/ce_masked", "val/ce_masked",
        "train/ce_clean_all", "val/ce_clean_all",
        "train/ce_corrupt_all", "val/ce_corrupt_all",
        "train/acc_masked", "val/acc_masked",
        "train/acc_clean_all", "val/acc_clean_all",
        "train/acc_corrupt_all", "val/acc_corrupt_all",
        "train/kl", "val/kl",
        "train/mask_frac", "val/mask_frac",
        "train/delta_in_l1", "val/delta_in_l1",
        "train/ratio_masked_over_clean", "val/ratio_masked_over_clean",
        "target/loss",
        "target/ce_masked",
        "target/ce_clean_all",
        "target/acc_masked",
        "target/acc_clean_all",
        "target/kl",
    ]

    E_mean = epoch_agg(df, cols, how="mean")
    E_last = epoch_agg(df, cols, how="last")
    E_max = epoch_agg(df, cols, how="max")

    save_lineplot(
        E_mean,
        outdir / "epoch_loss_mean.png",
        "train/loss",
        "val/loss",
        title="Epoch loss (mean over rows)",
        ylabel="loss",
    )

    save_lineplot(
        E_mean,
        outdir / "masked_mse_mean.png",
        "train/ce_masked",
        "val/ce_masked",
        title="Masked cross-entropy (mean over rows)",
        ylabel="masked CE",
    )

    save_lineplot(
        E_mean,
        outdir / "clean_mse_mean.png",
        "train/ce_clean_all",
        "val/ce_clean_all",
        title="Clean-pass cross-entropy (mean over rows)",
        ylabel="clean CE",
    )

    save_lineplot(
        E_mean,
        outdir / "kl_mean_logy.png",
        "train/kl",
        "val/kl",
        title="KL (mean over rows)",
        ylabel="KL",
        logy=True,
    )

    # Additional plots; Snakefile does not require these, but they are useful.
    save_lineplot(
        E_mean,
        outdir / "masked_acc_mean.png",
        "train/acc_masked",
        "val/acc_masked",
        title="Masked accuracy (mean over rows)",
        ylabel="masked accuracy",
    )

    save_lineplot(
        E_mean,
        outdir / "clean_acc_mean.png",
        "train/acc_clean_all",
        "val/acc_clean_all",
        title="Clean-pass accuracy (mean over rows)",
        ylabel="clean accuracy",
    )

    if "train/ce_corrupt_all" in E_mean.columns and "val/ce_corrupt_all" in E_mean.columns:
        save_lineplot(
            E_mean,
            outdir / "corrupt_mse_mean.png",
            "train/ce_corrupt_all",
            "val/ce_corrupt_all",
            title="Corrupt-pass cross-entropy (mean over rows)",
            ylabel="corrupt CE",
        )

    if "train/acc_corrupt_all" in E_mean.columns and "val/acc_corrupt_all" in E_mean.columns:
        save_lineplot(
            E_mean,
            outdir / "corrupt_acc_mean.png",
            "train/acc_corrupt_all",
            "val/acc_corrupt_all",
            title="Corrupt-pass accuracy (mean over rows)",
            ylabel="corrupt accuracy",
        )

    if "train/mask_frac" in E_mean.columns and "val/mask_frac" in E_mean.columns:
        save_lineplot(
            E_mean,
            outdir / "mask_frac_mean.png",
            "train/mask_frac",
            "val/mask_frac",
            title="Mask fraction (mean over rows)",
            ylabel="mask_frac",
        )

    if "train/delta_in_l1" in E_mean.columns and "val/delta_in_l1" in E_mean.columns:
        save_lineplot(
            E_mean,
            outdir / "delta_in_l1_mean.png",
            "train/delta_in_l1",
            "val/delta_in_l1",
            title="Corruption strength Δ|x_in-x| (mean over rows)",
            ylabel="delta_in_l1",
        )

    if "train/ratio_masked_over_clean" in E_mean.columns and "val/ratio_masked_over_clean" in E_mean.columns:
        save_lineplot(
            E_mean,
            outdir / "ratio_masked_over_clean_mean.png",
            "train/ratio_masked_over_clean",
            "val/ratio_masked_over_clean",
            title="Masked CE / clean CE ratio (mean over rows)",
            ylabel="ratio",
        )

    save_lineplot(
        E_last,
        outdir / "val_loss_last.png",
        "val/loss",
        None,
        title="val/loss using LAST row",
        ylabel="val/loss",
    )

    save_lineplot(
        E_max,
        outdir / "val_loss_max.png",
        "val/loss",
        None,
        title="val/loss using MAX row",
        ylabel="val/loss",
    )

    if "val/ce_masked" in E_mean.columns and "val/ce_clean_all" in E_mean.columns:
        save_scatter(
            E_mean,
            outdir / "scatter_val_masked_vs_clean.png",
            "val/ce_masked",
            "val/ce_clean_all",
            title="Val: masked CE vs clean CE",
            xlabel="val/ce_masked (epoch-mean)",
            ylabel="val/ce_clean_all (epoch-mean)",
        )

    if "val/kl" in E_mean.columns and "val/loss" in E_mean.columns:
        save_scatter(
            E_mean,
            outdir / "scatter_val_kl_vs_loss.png",
            "val/kl",
            "val/loss",
            title="Val KL vs loss",
            xlabel="val/kl (epoch-mean)",
            ylabel="val/loss (epoch-mean)",
        )

    if "val/delta_in_l1" in E_mean.columns and "val/ce_masked" in E_mean.columns:
        save_scatter(
            E_mean,
            outdir / "scatter_val_delta_vs_masked.png",
            "val/delta_in_l1",
            "val/ce_masked",
            title="Val corruption strength vs masked CE",
            xlabel="val/delta_in_l1 (epoch-mean)",
            ylabel="val/ce_masked (epoch-mean)",
        )

    if "val/acc_masked" in E_mean.columns and "val/ce_masked" in E_mean.columns:
        save_scatter(
            E_mean,
            outdir / "scatter_val_acc_vs_ce_masked.png",
            "val/acc_masked",
            "val/ce_masked",
            title="Val masked accuracy vs masked CE",
            xlabel="val/acc_masked (epoch-mean)",
            ylabel="val/ce_masked (epoch-mean)",
        )


# =========================================================
# reconstruction summary helpers
# =========================================================

def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def _balanced_accuracy_3class(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")

    recalls = []
    for c in (0, 1, 2):
        idx = (y_true == c)
        if idx.sum() == 0:
            continue
        recalls.append(np.mean(y_pred[idx] == c))

    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.mean((a - b) ** 2))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.all(a == a.flat[0]) or np.all(b == b.flat[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def maf_baseline_predict_from_train(X_train: np.ndarray) -> np.ndarray:
    """
    Per-site expected genotype dosage baseline from training set:
        E[G_j] = mean genotype at site j
    """
    return np.mean(X_train, axis=0, keepdims=True).astype(np.float32)


def maf_baseline_class_from_train(X_train: np.ndarray) -> np.ndarray:
    """
    Per-site majority genotype class baseline from training set.
    """
    X_int = np.clip(np.rint(X_train), 0, 2).astype(np.int64)
    out = np.zeros((1, X_int.shape[1]), dtype=np.int64)
    for j in range(X_int.shape[1]):
        counts = np.bincount(X_int[:, j], minlength=3)
        out[0, j] = int(np.argmax(counts))
    return out


def confusion_matrix_3class(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true.ravel(), y_pred.ravel()):
        if 0 <= t <= 2 and 0 <= p <= 2:
            cm[int(t), int(p)] += 1
    return cm


def save_confusion_matrix_png(cm: np.ndarray, outpath: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["0", "1", "2"])
    ax.set_yticklabels(["0", "1", "2"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_recon_scatter(
    recon_npz: Path,
    outpath: Path,
    *,
    alpha: float = 0.25,
    point_size: float = 5,
    max_points: int = 50000,
    seed: int = 0,
) -> dict:
    """
    Scatter of true genotype dosage vs predicted expected dosage at masked positions.
    """
    data = np.load(recon_npz)

    x_true = np.asarray(data["x_true"], dtype=np.float32)
    mask = np.asarray(data["mask"], dtype=bool)
    _, probs = _load_pred_and_probs(data)
    pred_dosage = _expected_dosage_from_probs(probs)

    if x_true.shape != mask.shape or x_true.shape != pred_dosage.shape:
        raise ValueError(
            f"Shape mismatch in {recon_npz}: "
            f"x_true={x_true.shape}, mask={mask.shape}, pred_dosage={pred_dosage.shape}"
        )

    true_masked = x_true[mask].astype(np.float32)
    pred_masked = pred_dosage[mask].astype(np.float32)

    n_masked = len(true_masked)
    if n_masked == 0:
        return {"corr": float('nan'), "mse": float('nan'), "n_masked": 0}

    corr = _corr(true_masked, pred_masked)
    mse = _mse(true_masked, pred_masked)

    if n_masked > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_masked, size=max_points, replace=False)
        true_plot = true_masked[idx]
        pred_plot = pred_masked[idx]
    else:
        true_plot = true_masked
        pred_plot = pred_masked

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_plot, pred_plot, alpha=alpha, s=point_size, edgecolors="none")

    lo = float(min(true_plot.min(), pred_plot.min()))
    hi = float(max(true_plot.max(), pred_plot.max()))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)

    ax.set_xlabel("True genotype value (masked positions)")
    ax.set_ylabel("Predicted expected dosage")
    ax.set_title(f"Masked-position reconstruction\ncorr={corr:.3f}, mse={mse:.4f}, n={n_masked:,}")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return {"corr": corr, "mse": mse, "n_masked": n_masked}


def summarize_one_split(
    name: str,
    x_true: np.ndarray,
    pred_cls: np.ndarray,
    pred_dosage: np.ndarray,
    mask: np.ndarray,
    train_dosage_baseline: np.ndarray,
    train_class_baseline: np.ndarray,
) -> list[str]:
    """
    Produce lines for recon_summary.txt.
    Format:
      split    metric    model    maf_baseline    where
    """
    lines: list[str] = []

    x_true_cont = x_true.astype(np.float32)
    x_true_cls = np.clip(np.rint(x_true), 0, 2).astype(np.int64)

    dosage_base = np.repeat(train_dosage_baseline, repeats=x_true.shape[0], axis=0)
    class_base = np.repeat(train_class_baseline, repeats=x_true.shape[0], axis=0)

    # all positions
    lines.append(
        f"{name}\tmse\t{_mse(pred_dosage, x_true_cont):.6g}\t{_mse(dosage_base, x_true_cont):.6g}\tall"
    )
    lines.append(
        f"{name}\tmae\t{_mae(pred_dosage, x_true_cont):.6g}\t{_mae(dosage_base, x_true_cont):.6g}\tall"
    )
    lines.append(
        f"{name}\tcorr\t{_corr(pred_dosage.ravel(), x_true_cont.ravel()):.6g}\t"
        f"{_corr(dosage_base.ravel(), x_true_cont.ravel()):.6g}\tall"
    )
    lines.append(
        f"{name}\tacc\t{_accuracy(x_true_cls, pred_cls):.6g}\t{_accuracy(x_true_cls, class_base):.6g}\tall"
    )
    lines.append(
        f"{name}\tbalanced_acc\t{_balanced_accuracy_3class(x_true_cls, pred_cls):.6g}\t"
        f"{_balanced_accuracy_3class(x_true_cls, class_base):.6g}\tall"
    )

    # masked only
    if mask.sum() > 0:
        xt_m_cont = x_true_cont[mask]
        pr_m_cont = pred_dosage[mask]
        db_m_cont = dosage_base[mask]

        xt_m_cls = x_true_cls[mask]
        pr_m_cls = pred_cls[mask]
        cb_m_cls = class_base[mask]

        lines.append(
            f"{name}\tmse\t{_mse(pr_m_cont, xt_m_cont):.6g}\t{_mse(db_m_cont, xt_m_cont):.6g}\tmasked_only"
        )
        lines.append(
            f"{name}\tmae\t{_mae(pr_m_cont, xt_m_cont):.6g}\t{_mae(db_m_cont, xt_m_cont):.6g}\tmasked_only"
        )
        lines.append(
            f"{name}\tcorr\t{_corr(pr_m_cont, xt_m_cont):.6g}\t{_corr(db_m_cont, xt_m_cont):.6g}\tmasked_only"
        )
        lines.append(
            f"{name}\tacc\t{_accuracy(xt_m_cls, pr_m_cls):.6g}\t{_accuracy(xt_m_cls, cb_m_cls):.6g}\tmasked_only"
        )
        lines.append(
            f"{name}\tbalanced_acc\t{_balanced_accuracy_3class(xt_m_cls, pr_m_cls):.6g}\t"
            f"{_balanced_accuracy_3class(xt_m_cls, cb_m_cls):.6g}\tmasked_only"
        )

    return lines


# =========================================================
# main run()
# =========================================================

def run(a: Args) -> None:
    a.outdir.mkdir(parents=True, exist_ok=True)

    # 1) training curves
    df = load_metrics_csv(a.logdir)
    make_notebook_equivalent_plots(df, a.outdir / "plots")

    # 2) load genotype arrays for baselines
    train_X = _load_np_array(a.train_genotype)
    val_X = _load_np_array(a.val_genotype)
    target_X = _load_np_array(a.target_genotype)

    if train_X is None:
        raise ValueError("train_genotype is required")

    dosage_baseline = maf_baseline_predict_from_train(train_X)
    class_baseline = maf_baseline_class_from_train(train_X)

    # 3) reconstruction summaries from recon_dir
    summary_lines = ["# split\tmetric\tmodel\tmaf_baseline\twhere"]
    scatter_stats_lines = ["split\tcorr\tmse\tn_masked"]

    if a.recon_dir is not None and a.recon_dir.exists():
        split_to_true = {
            "train": train_X,
            "val": val_X,
            "target": target_X,
        }
        split_to_name = {
            "train": "CEU_train",
            "val": "CEU_val",
            "target": "YRI_target",
        }

        for split in ["train", "val", "target"]:
            npz_path = a.recon_dir / f"{split}_recon.npz"
            x_ref = split_to_true[split]

            if x_ref is None or not npz_path.exists():
                continue

            data = np.load(npz_path)

            if "x_true" not in data.files:
                raise KeyError(f"'x_true' is not in archive {npz_path}. Found keys: {list(data.files)}")
            if "mask" not in data.files:
                raise KeyError(f"'mask' is not in archive {npz_path}. Found keys: {list(data.files)}")

            x_true = np.asarray(data["x_true"], dtype=np.float32)
            mask = np.asarray(data["mask"], dtype=bool)
            pred_cls, probs = _load_pred_and_probs(data)
            pred_dosage = _expected_dosage_from_probs(probs)

            if x_true.shape != pred_cls.shape or x_true.shape != pred_dosage.shape or x_true.shape != mask.shape:
                raise ValueError(
                    f"Shape mismatch for split={split}: "
                    f"x_true={x_true.shape}, pred_cls={pred_cls.shape}, "
                    f"pred_dosage={pred_dosage.shape}, mask={mask.shape}"
                )

            if x_ref.shape[1] != x_true.shape[1]:
                raise ValueError(
                    f"Saved recon feature count does not match reference for split={split}: "
                    f"{x_true.shape[1]} vs {x_ref.shape[1]}"
                )

            split_name = split_to_name[split]
            summary_lines.extend(
                summarize_one_split(
                    name=split_name,
                    x_true=x_true,
                    pred_cls=pred_cls,
                    pred_dosage=pred_dosage,
                    mask=mask,
                    train_dosage_baseline=dosage_baseline,
                    train_class_baseline=class_baseline,
                )
            )

            scatter_png = a.outdir / "plots" / f"recon_scatter_{split}.png"
            stats = save_recon_scatter(npz_path, scatter_png, seed=0)
            scatter_stats_lines.append(
                f"{split}\t{stats['corr']:.6f}\t{stats['mse']:.6f}\t{stats['n_masked']}"
            )

            if a.save_confusion:
                true_cls = np.clip(np.rint(x_true), 0, 2).astype(np.int64)

                cm_all = confusion_matrix_3class(true_cls, pred_cls)
                save_confusion_matrix_png(
                    cm_all,
                    a.outdir / "plots" / f"confusion_{split}_all.png",
                    f"{split}: confusion matrix (all positions)",
                )

                if mask.sum() > 0:
                    cm_masked = confusion_matrix_3class(true_cls[mask], pred_cls[mask])
                    save_confusion_matrix_png(
                        cm_masked,
                        a.outdir / "plots" / f"confusion_{split}_masked.png",
                        f"{split}: confusion matrix (masked positions)",
                    )

    (a.outdir / "recon_summary.txt").write_text("\n".join(summary_lines) + "\n")
    (a.outdir / "recon_scatter_stats.txt").write_text("\n".join(scatter_stats_lines) + "\n")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)

    ap.add_argument("--train-genotype", type=Path, required=True)
    ap.add_argument("--val-genotype", type=Path, default=None)
    ap.add_argument("--target-genotype", type=Path, default=None)

    ap.add_argument("--recon-dir", type=Path, default=None)
    ap.add_argument("--no-confusion", action="store_true")

    # accepted for compatibility with the current Snakefile; ignored here
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--resolved-hparams", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max-step-points", type=int, default=5000)

    x = ap.parse_args()

    run(
        Args(
            logdir=x.logdir,
            outdir=x.outdir,
            train_genotype=x.train_genotype,
            val_genotype=x.val_genotype,
            target_genotype=x.target_genotype,
            recon_dir=x.recon_dir,
            save_confusion=(not x.no_confusion),
        )
    )


if __name__ == "__main__":
    main()