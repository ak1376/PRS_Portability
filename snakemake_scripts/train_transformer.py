#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformer.data_class import HapDataset, collate_hapbatch
from src.transformer.model import HapMaskTransformer
from src.transformer.masking import mask_haplotype
from src.transformer.train import train_epoch, eval_epoch, debug_snapshot_and_pngs


# -----------------------------------------------------------------------------
# Early stopping
# -----------------------------------------------------------------------------
@dataclass
class EarlyStopConfig:
    enabled: bool = True
    monitor: str = "val_mlm_loss"   # "val_mlm_loss", "val_accuracy", "val_auc"
    mode: str = "min"              # "min" or "max"
    patience: int = 25
    min_delta: float = 1e-4
    burn_in: int = 0               # epochs to wait before stopping


def _is_improvement(curr: float, best: float, *, mode: str, min_delta: float) -> bool:
    if mode not in ("min", "max"):
        raise ValueError(f"early_stopping.mode must be 'min' or 'max', got {mode}")
    if np.isnan(curr):
        return False
    if np.isnan(best):
        return True
    if mode == "min":
        return curr < (best - float(min_delta))
    else:
        return curr > (best + float(min_delta))


# -----------------------------------------------------------------------------
# Dataset split / loaders
# -----------------------------------------------------------------------------
def make_train_val_loaders(
    ds: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_frac: float,
    device: torch.device,
    collate_fn,
):
    """
    Deterministic split of dataset indices into train/val.
    """
    n = len(ds)
    if n < 2:
        raise ValueError(f"Need at least 2 samples to split train/val, got n={n}")

    n_val = int(round(float(val_frac) * n))
    n_val = max(1, min(n - 1, n_val))  # keep both non-empty

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_dl = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=(int(num_workers) > 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=(int(num_workers) > 0),
    )
    return train_dl, val_dl, len(train_ds), len(val_ds)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def write_losses_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write rows with a union-of-keys header (robust if some rows have extra columns).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")

    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_losses(path: Path, epochs: list[int], train_mlm: list[float], val_mlm: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, train_mlm, label="train_mlm")
    plt.plot(epochs, val_mlm, label="val_mlm")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_contrastive_geometry(
    path: Path,
    epochs: list[int],
    pos_cos: list[float],
    neg_cos: list[float],
    perm_neg_cos: Optional[list[float]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, pos_cos, label="pos_cos (z1·z2)")
    plt.plot(epochs, neg_cos, label="neg_cos (z1·z2_shuf)")
    if perm_neg_cos is not None:
        plt.plot(epochs, perm_neg_cos, label="perm_neg_cos (z1·zperm)")
    plt.xlabel("epoch")
    plt.ylabel("cosine")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_confusion_matrix_png(cm: np.ndarray, out_png: Path, *, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])

    # annotate
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def eval_masked_acc_auc(
    model: HapMaskTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    mask_id: int,
    p_mask_site: float,
    class_balance: bool = True,
) -> dict[str, float]:
    """
    Compute accuracy + AUC on masked sites over the given loader.
    Only used if you early-stop on val_accuracy or val_auc (or want to log them).
    """
    from src.transformer.train import _compute_masked_loss  # local import

    model.eval()

    all_targets: list[int] = []
    all_preds: list[int] = []
    all_p1: list[float] = []
    total_loss = 0.0
    total_sites = 0

    for batch in loader:
        hap = batch.hap.to(device)
        pad_mask = batch.pad_mask.to(device) if getattr(batch, "pad_mask", None) is not None else None

        hap_true = hap
        hap_masked, masked_sites = mask_haplotype(hap, mask_id=mask_id, p_mask_site=p_mask_site)

        logits, _z = model(hap_masked, pad_mask=pad_mask)

        loss_mask = masked_sites
        if pad_mask is not None:
            loss_mask = loss_mask & (~pad_mask)

        loss, n = _compute_masked_loss(logits, hap_true, loss_mask, class_balance=class_balance)
        if n == 0:
            continue

        probs = torch.softmax(logits, dim=-1)   # (B,L,2)
        pred = torch.argmax(logits, dim=-1)     # (B,L)

        t = hap_true[loss_mask].detach().cpu().numpy().astype(np.int64)
        p = pred[loss_mask].detach().cpu().numpy().astype(np.int64)
        p1 = probs[loss_mask, 1].detach().cpu().numpy().astype(np.float64)

        all_targets.extend(t.tolist())
        all_preds.extend(p.tolist())
        all_p1.extend(p1.tolist())

        total_loss += float(loss.item()) * n
        total_sites += n

    if len(all_targets) == 0:
        return {"val_accuracy": float("nan"), "val_auc": float("nan"), "val_avg_loss": float("nan")}

    targets = np.asarray(all_targets, dtype=np.int64)
    preds = np.asarray(all_preds, dtype=np.int64)
    p1 = np.asarray(all_p1, dtype=np.float64)

    acc = float(accuracy_score(targets, preds))

    # AUC can fail if only one class appears
    try:
        auc = float(roc_auc_score(targets, p1))
    except Exception:
        auc = float("nan")

    avg_loss = float(total_loss / max(total_sites, 1))

    return {"val_accuracy": acc, "val_auc": auc, "val_avg_loss": avg_loss}


def comprehensive_validation_analysis(
    model: HapMaskTransformer,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    *,
    mask_id: int,
    p_mask_site: float = 0.15,
    class_balance: bool = True,
) -> dict[str, Any]:
    """
    Final validation analysis on the provided loader (use the VAL loader).
    Saves:
      - validation_predictions.npy / validation_targets.npy / validation_probabilities.npy
      - confusion_matrix.png
      - roc_curve.png
      - validation_metrics.json
    """
    from src.transformer.train import _compute_masked_loss

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[int] = []
    all_targets: list[int] = []
    all_probabilities: list[float] = []

    total_loss = 0.0
    total_sites = 0

    print("Collecting validation predictions...")

    with torch.no_grad():
        for batch in loader:
            hap = batch.hap.to(device)
            pad_mask = batch.pad_mask.to(device) if getattr(batch, "pad_mask", None) is not None else None

            hap_true = hap
            hap_masked, masked_sites = mask_haplotype(hap, mask_id=mask_id, p_mask_site=p_mask_site)

            logits, _z = model(hap_masked, pad_mask=pad_mask)

            loss_mask = masked_sites
            if pad_mask is not None:
                loss_mask = loss_mask & (~pad_mask)

            loss, n = _compute_masked_loss(logits, hap_true, loss_mask, class_balance=class_balance)
            if n == 0:
                continue

            probs = torch.softmax(logits, dim=-1)     # (B,L,2)
            pred_class = torch.argmax(logits, dim=-1) # (B,L)

            masked_targets = hap_true[loss_mask].cpu().numpy().astype(np.int64)
            masked_predictions = pred_class[loss_mask].cpu().numpy().astype(np.int64)
            masked_probabilities = probs[loss_mask, 1].cpu().numpy().astype(np.float64)

            all_targets.extend(masked_targets.tolist())
            all_predictions.extend(masked_predictions.tolist())
            all_probabilities.extend(masked_probabilities.tolist())

            total_loss += float(loss.item()) * n
            total_sites += n

    targets = np.asarray(all_targets, dtype=np.int64)
    predictions = np.asarray(all_predictions, dtype=np.int64)
    probabilities = np.asarray(all_probabilities, dtype=np.float64)

    if targets.size == 0:
        raise RuntimeError("No masked sites were collected in validation analysis. Check masking/pad handling.")

    accuracy = float(accuracy_score(targets, predictions))

    try:
        auc_score = float(roc_auc_score(targets, probabilities))
    except Exception:
        auc_score = float("nan")

    avg_loss = float(total_loss / max(total_sites, 1))

    print("Validation Results:")
    print(f"  Total masked sites: {len(targets):,}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc_score:.4f}")
    print(f"  Average loss: {avg_loss:.6f}")

    np.save(output_dir / "validation_predictions.npy", predictions)
    np.save(output_dir / "validation_targets.npy", targets)
    np.save(output_dir / "validation_probabilities.npy", probabilities)

    cm = confusion_matrix(targets, predictions)
    _plot_confusion_matrix_png(cm, output_dir / "confusion_matrix.png", title=f"Confusion Matrix (acc={accuracy:.4f})")

    # ROC curve
    # If only one class present, roc_curve can error
    try:
        fpr, tpr, _thr = roc_curve(targets, probabilities)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={auc_score:.4f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Validation Set")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "auc": auc_score,
        "average_loss": avg_loss,
        "total_masked_sites": int(len(targets)),
        "class_distribution": {
            "class_0_count": int(np.sum(targets == 0)),
            "class_1_count": int(np.sum(targets == 1)),
            "class_0_fraction": float(np.mean(targets == 0)),
            "class_1_fraction": float(np.mean(targets == 1)),
        },
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }

    import json
    with open(output_dir / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Validation analysis saved to: {output_dir}")
    return metrics


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
def _maybe_load_cfg(cfg_path: str | None) -> dict:
    if cfg_path is None:
        return {}
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"--config not found: {cfg_path}")
    return yaml.safe_load(p.read_text()) or {}


def _get_nested(cfg: dict, keys: list[str], default):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _pick(cli_val, yaml_val, default):
    return default if (cli_val is None and yaml_val is None) else (yaml_val if cli_val is None else cli_val)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--hap", type=str, required=True, help="Path to ONE haplotype .npy (N,L)")
    ap.add_argument("--out_model", type=str, required=True)
    ap.add_argument("--out_losses", type=str, required=True)
    ap.add_argument("--out_plot", type=str, required=True)
    ap.add_argument("--out_debug_dir", type=str, required=True, help="Directory for debug PNGs")
    ap.add_argument("--out_ctr_plot", type=str, default=None, help="Optional: contrastive geometry plot path")

    ap.add_argument("--config", type=str, default=None)

    # Model overrides
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--vocab_size", type=int, default=None)
    ap.add_argument("--pad_id", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--pool", type=str, default=None)

    # Training overrides
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--debug_every", type=int, default=None)
    ap.add_argument("--debug_n_show", type=int, default=None)
    ap.add_argument("--debug_max_sites", type=int, default=None)

    # Train/val split
    ap.add_argument("--val_frac", type=float, default=None, help="Fraction of samples used for validation split")

    # Early stopping overrides (optional)
    ap.add_argument("--early_stop", action="store_true", help="Enable early stopping")
    ap.add_argument("--early_monitor", type=str, default=None)
    ap.add_argument("--early_mode", type=str, default=None, choices=["min", "max"])
    ap.add_argument("--early_patience", type=int, default=None)
    ap.add_argument("--early_min_delta", type=float, default=None)
    ap.add_argument("--early_burn_in", type=int, default=None)

    # Masking overrides
    ap.add_argument("--p_mask_site", type=float, default=None)
    ap.add_argument("--mask_id", type=int, default=None)

    # Contrastive
    ap.add_argument("--contrastive", action="store_true", help="Enable contrastive auxiliary loss")
    ap.add_argument("--contrastive_lambda", type=float, default=None)
    ap.add_argument("--contrastive_tau", type=float, default=None)
    ap.add_argument("--permute_every_k", type=int, default=None)
    ap.add_argument("--no_perm_negatives", action="store_true")
    ap.add_argument("--p_mask_site_ctr", type=float, default=None)

    # Windowing overrides
    ap.add_argument("--window_len", type=int, default=None)
    ap.add_argument("--window_mode", type=str, default=None, choices=["random", "first", "middle", "fixed"])
    ap.add_argument("--fixed_start", type=int, default=None)

    args = ap.parse_args()

    # ---------------------
    # Load YAML
    # ---------------------
    cfg = _maybe_load_cfg(args.config)
    base = cfg.get("base", cfg) if isinstance(cfg, dict) else {}
    model_cfg = base.get("model", {}) or {}
    train_cfg = base.get("training", {}) or {}
    masking_cfg = base.get("masking", {}) or {}
    data_cfg = base.get("data", {}) or {}

    # ---------------------
    # Resolve params (CLI > YAML > default)
    # ---------------------
    # Model
    args.d_model = _pick(args.d_model, model_cfg.get("d_model"), 128)
    args.n_heads = _pick(args.n_heads, model_cfg.get("n_heads"), 8)
    args.n_layers = _pick(args.n_layers, model_cfg.get("n_layers"), 6)
    args.dropout = _pick(args.dropout, model_cfg.get("dropout"), 0.1)
    args.vocab_size = _pick(args.vocab_size, model_cfg.get("vocab_size"), 3)
    args.pad_id = _pick(args.pad_id, model_cfg.get("pad_id"), None)
    args.max_len = _pick(args.max_len, model_cfg.get("max_len"), 50_000)
    args.pool = _pick(args.pool, model_cfg.get("pool"), "mean")

    # Training
    args.epochs = _pick(args.epochs, train_cfg.get("epochs"), 10)
    args.batch_size = _pick(args.batch_size, train_cfg.get("batch_size"), 32)
    args.lr = _pick(args.lr, train_cfg.get("lr"), 3e-4)
    args.num_workers = _pick(args.num_workers, train_cfg.get("num_workers"), 2)
    args.grad_clip = _pick(args.grad_clip, train_cfg.get("grad_clip"), 1.0)
    args.weight_decay = _pick(args.weight_decay, train_cfg.get("weight_decay"), 0.0)
    args.debug_every = _pick(args.debug_every, train_cfg.get("debug_every"), 5)
    args.debug_n_show = _pick(args.debug_n_show, train_cfg.get("debug_n_show"), 2)
    args.debug_max_sites = _pick(args.debug_max_sites, train_cfg.get("debug_max_sites"), 256)

    # Train/val split
    args.val_frac = _pick(args.val_frac, train_cfg.get("val_frac"), 0.1)
    if not (0.0 < float(args.val_frac) < 1.0):
        raise ValueError(f"val_frac must be in (0,1), got {args.val_frac}")

    # Early stopping config (YAML defaults)
    es_yaml = train_cfg.get("early_stopping", {}) or {}
    es = EarlyStopConfig(
        enabled=bool(_pick((True if args.early_stop else None), es_yaml.get("enabled"), False)),
        monitor=str(_pick(args.early_monitor, es_yaml.get("monitor"), "val_mlm_loss")),
        mode=str(_pick(args.early_mode, es_yaml.get("mode"), "min")),
        patience=int(_pick(args.early_patience, es_yaml.get("patience"), 25)),
        min_delta=float(_pick(args.early_min_delta, es_yaml.get("min_delta"), 1e-4)),
        burn_in=int(_pick(args.early_burn_in, es_yaml.get("burn_in"), 0)),
    )

    # Masking
    args.p_mask_site = _pick(args.p_mask_site, masking_cfg.get("p_mask_site"), 0.15)
    args.mask_id = _pick(args.mask_id, masking_cfg.get("mask_id"), 2)

    # Windowing
    args.window_len = _pick(args.window_len, data_cfg.get("window_len"), 1024)
    args.window_mode = _pick(args.window_mode, data_cfg.get("window_mode"), "random")
    args.fixed_start = _pick(args.fixed_start, data_cfg.get("fixed_start"), 0)

    # Contrastive YAML block
    ctr_cfg = train_cfg.get("contrastive", {}) or {}
    args.contrastive_lambda = _pick(args.contrastive_lambda, ctr_cfg.get("lambda"), 0.1)
    args.contrastive_tau = _pick(args.contrastive_tau, ctr_cfg.get("tau"), 0.2)
    args.permute_every_k = _pick(args.permute_every_k, ctr_cfg.get("permute_every_k"), 5)
    args.p_mask_site_ctr = _pick(
        args.p_mask_site_ctr,
        _get_nested(base, ["training", "contrastive", "p_mask_site_ctr"], None),
        None,
    )

    ctr_enabled_yaml = bool(ctr_cfg.get("enabled", False))
    args.contrastive = bool(args.contrastive or ctr_enabled_yaml)

    perm_neg_yaml = bool(ctr_cfg.get("permute_negatives", True))
    args.permute_negatives = bool(perm_neg_yaml and (not args.no_perm_negatives))

    # ---------------------
    # Seeds + device
    # ---------------------
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------
    # Load hap
    # ---------------------
    hap_np = np.load(args.hap)
    if hap_np.ndim != 2:
        raise ValueError(f"--hap must be 2D (N,L). Got {hap_np.shape}")
    hap = torch.from_numpy(hap_np).long()

    print(f"[train_transformer_single] hap shape: {tuple(hap.shape)}")
    print(f"[train_transformer_single] window_len={args.window_len} window_mode={args.window_mode} fixed_start={args.fixed_start}")
    print(f"[train_transformer_single] masking p_mask_site={args.p_mask_site} mask_id={args.mask_id}")
    print(f"[train_transformer_single] val_frac={float(args.val_frac):.3f}")
    print(
        f"[train_transformer_single] contrastive={args.contrastive} "
        f"lambda={args.contrastive_lambda} tau={args.contrastive_tau} "
        f"perm_neg={args.permute_negatives} permute_every_k={args.permute_every_k} "
        f"p_mask_site_ctr={args.p_mask_site_ctr}"
    )
    print(
        f"[early_stopping] enabled={es.enabled} monitor={es.monitor} mode={es.mode} "
        f"patience={es.patience} min_delta={es.min_delta} burn_in={es.burn_in}"
    )

    # prior for bias init
    with torch.no_grad():
        pi = float(hap.float().mean().item())
        pi = min(max(pi, 1e-4), 1.0 - 1e-4)
        logit_pi = float(np.log(pi / (1.0 - pi)))
        print(f"[init] pi={pi:.6f} logit_pi={logit_pi:.6f}")

    # ---------------------
    # Dataset + train/val loaders
    # ---------------------
    g = torch.Generator().manual_seed(int(args.seed))
    ds = HapDataset(
        hap_all=hap,
        pad_id=args.pad_id,
        window_len=int(args.window_len) if args.window_len is not None else None,
        window_mode=str(args.window_mode),
        fixed_start=int(args.fixed_start),
        rng=g,
    )

    train_dl, val_dl, n_tr, n_va = make_train_val_loaders(
        ds,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        device=device,
        collate_fn=collate_hapbatch,
    )
    print(f"[split] n_train={n_tr} n_val={n_va}")

    # ---------------------
    # Model
    # ---------------------
    model = HapMaskTransformer(
        vocab_size=int(args.vocab_size),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        pad_id=args.pad_id,
        pool=str(args.pool),
        max_len=int(args.max_len),
    ).to(device)

    # init bias to class prior
    with torch.no_grad():
        model.head.bias.zero_()
        model.head.bias[1] = logit_pi

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    # ---------------------
    # Outputs
    # ---------------------
    out_debug = Path(args.out_debug_dir)
    out_debug.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    epochs: list[int] = []
    train_mlm_losses: list[float] = []
    val_mlm_losses: list[float] = []

    # contrastive tracking
    ctr_pos_cos: list[float] = []
    ctr_neg_cos: list[float] = []
    ctr_perm_neg_cos: list[float] = []

    # ---------------------
    # Early stopping state
    # ---------------------
    best_metric = float("nan")
    best_epoch = 0
    best_state_cpu: Optional[dict[str, torch.Tensor]] = None
    bad_epochs = 0

    # validate monitor name
    valid_monitors = {"val_mlm_loss", "val_accuracy", "val_auc"}
    if es.monitor not in valid_monitors:
        raise ValueError(f"early_stopping.monitor must be one of {sorted(valid_monitors)}, got {es.monitor}")

    # ---------------------
    # Train loop
    # ---------------------
    for ep in range(1, int(args.epochs) + 1):
        tr_out = train_epoch(
            model=model,
            loader=train_dl,
            optimizer=opt,
            device=device,
            mask_id=int(args.mask_id),
            p_mask_site=float(args.p_mask_site),
            p_mask_site_ctr=(None if args.p_mask_site_ctr is None else float(args.p_mask_site_ctr)),
            grad_clip=float(args.grad_clip),
            # contrastive knobs
            use_contrastive=bool(args.contrastive),
            contrastive_lambda=float(args.contrastive_lambda),
            contrastive_tau=float(args.contrastive_tau),
            use_perm_negatives=bool(args.permute_negatives),
            permute_every_k=int(args.permute_every_k),
        )

        tr_mlm = float(tr_out["mlm_loss"])
        tr_ctr = float(tr_out.get("ctr_loss", float("nan")))
        pos_cos = float(tr_out.get("ctr_pos_cos", float("nan")))
        neg_cos = float(tr_out.get("ctr_neg_cos", float("nan")))
        perm_cos = float(tr_out.get("ctr_perm_neg_cos", float("nan")))

        # val loss on VAL loader
        va_loss = float(
            eval_epoch(
                model=model,
                loader=val_dl,
                device=device,
                mask_id=int(args.mask_id),
                p_mask_site=float(args.p_mask_site),
            )
        )

        # only compute acc/auc if needed (monitor or you want to log)
        val_acc = float("nan")
        val_auc = float("nan")
        if es.monitor in ("val_accuracy", "val_auc"):
            m = eval_masked_acc_auc(
                model=model,
                loader=val_dl,
                device=device,
                mask_id=int(args.mask_id),
                p_mask_site=float(args.p_mask_site),
            )
            val_acc = float(m["val_accuracy"])
            val_auc = float(m["val_auc"])

        # logging
        if args.contrastive:
            msg = (
                f"[epoch {ep:03d}] train_mlm={tr_mlm:.6f} train_ctr={tr_ctr:.6f} "
                f"pos_cos={pos_cos:.4f} neg_cos={neg_cos:.4f}"
            )
            if not np.isnan(perm_cos):
                msg += f" perm_neg_cos={perm_cos:.4f}"
            msg += f" val_mlm={va_loss:.6f}"
            if es.monitor in ("val_accuracy", "val_auc"):
                msg += f" val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
            print(msg)
        else:
            msg = f"[epoch {ep:03d}] train_mlm={tr_mlm:.6f} val_mlm={va_loss:.6f}"
            if es.monitor in ("val_accuracy", "val_auc"):
                msg += f" val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
            print(msg)

        # Debug snapshots (use TRAIN loader batch so it’s stable & fast)
        if int(args.debug_every) > 0 and (ep % int(args.debug_every) == 0):
            batch = next(iter(train_dl))
            snap = debug_snapshot_and_pngs(
                model=model,
                batch=batch,
                device=device,
                mask_id=int(args.mask_id),
                p_mask_site=float(args.p_mask_site),
                out_dir=out_debug / f"ep{ep:03d}",
                step_tag=f"ep{ep:03d}",
                n_show=int(args.debug_n_show),
                max_sites=int(args.debug_max_sites),
            )
            print(
                f"[debug ep={ep}] masked_sites={snap['masked_sites']} "
                f"acc={snap['acc']:.3f} pi_masked={snap['pi_masked']:.3f} "
                f"baseMaj={snap['baseline_majority']:.3f}"
            )

        # record row
        row: dict[str, Any] = {
            "epoch": ep,
            "train_mlm_loss": tr_mlm,
            "val_mlm_loss": va_loss,
        }
        if es.monitor in ("val_accuracy", "val_auc"):
            row["val_accuracy"] = val_acc
            row["val_auc"] = val_auc

        if args.contrastive:
            row["train_ctr_loss"] = tr_ctr
            row["ctr_pos_cos"] = pos_cos
            row["ctr_neg_cos"] = neg_cos
            if not np.isnan(perm_cos):
                row["ctr_perm_neg_cos"] = perm_cos
        rows.append(row)

        epochs.append(ep)
        train_mlm_losses.append(tr_mlm)
        val_mlm_losses.append(va_loss)

        if args.contrastive:
            ctr_pos_cos.append(pos_cos)
            ctr_neg_cos.append(neg_cos)
            ctr_perm_neg_cos.append(perm_cos)

        # ---------------------
        # Early stopping check
        # ---------------------
        if es.enabled:
            # get current monitored metric
            if es.monitor == "val_mlm_loss":
                curr = va_loss
            elif es.monitor == "val_accuracy":
                curr = val_acc
            elif es.monitor == "val_auc":
                curr = val_auc
            else:
                curr = float("nan")

            improved = _is_improvement(curr, best_metric, mode=es.mode, min_delta=es.min_delta)
            if improved:
                best_metric = float(curr)
                best_epoch = int(ep)
                bad_epochs = 0

                # save best weights (CPU copy to avoid GPU memory balloon)
                best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                print(f"[early_stopping] new best {es.monitor}={best_metric:.6f} at epoch {best_epoch}")
            else:
                if ep >= int(es.burn_in):
                    bad_epochs += 1
                    print(f"[early_stopping] no improvement (bad_epochs={bad_epochs}/{es.patience})")
                    if bad_epochs >= int(es.patience):
                        print(
                            f"[early_stopping] STOP at epoch {ep} "
                            f"(best epoch={best_epoch}, best {es.monitor}={best_metric:.6f})"
                        )
                        break
                else:
                    # still in burn-in: do not accumulate bad epochs
                    pass

    # ---------------------
    # Restore best weights (if early stopping was enabled)
    # ---------------------
    if es.enabled and best_state_cpu is not None:
        model.load_state_dict(best_state_cpu)
        print(f"[early_stopping] restored best model from epoch {best_epoch} ({es.monitor}={best_metric:.6f})")

    # ---------------------
    # Save checkpoint (BEST model if early stopping enabled)
    # ---------------------
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                # model
                "vocab_size": int(args.vocab_size),
                "d_model": int(args.d_model),
                "n_heads": int(args.n_heads),
                "n_layers": int(args.n_layers),
                "dropout": float(args.dropout),
                "pad_id": args.pad_id,
                "pool": str(args.pool),
                "max_len": int(args.max_len),
                # training
                "lr": float(args.lr),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
                "weight_decay": float(args.weight_decay),
                "grad_clip": float(args.grad_clip),
                "val_frac": float(args.val_frac),
                "early_stopping": {
                    "enabled": bool(es.enabled),
                    "monitor": str(es.monitor),
                    "mode": str(es.mode),
                    "patience": int(es.patience),
                    "min_delta": float(es.min_delta),
                    "burn_in": int(es.burn_in),
                    "best_epoch": int(best_epoch),
                    "best_metric": float(best_metric) if not np.isnan(best_metric) else None,
                },
                # masking
                "p_mask_site": float(args.p_mask_site),
                "mask_id": int(args.mask_id),
                # contrastive
                "contrastive": bool(args.contrastive),
                "contrastive_lambda": float(args.contrastive_lambda),
                "contrastive_tau": float(args.contrastive_tau),
                "permute_negatives": bool(args.permute_negatives),
                "permute_every_k": int(args.permute_every_k),
                "p_mask_site_ctr": (None if args.p_mask_site_ctr is None else float(args.p_mask_site_ctr)),
                # windowing
                "window_len": int(args.window_len) if args.window_len is not None else None,
                "window_mode": str(args.window_mode),
                "fixed_start": int(args.fixed_start),
            },
        },
        out_model,
    )

    # ---------------------
    # Write CSV + plots
    # ---------------------
    write_losses_csv(Path(args.out_losses), rows)
    plot_losses(Path(args.out_plot), epochs, train_mlm_losses, val_mlm_losses)

    if args.contrastive:
        out_ctr_plot = (
            Path(args.out_ctr_plot)
            if args.out_ctr_plot is not None
            else Path(args.out_plot).with_name("contrastive_geometry.png")
        )
        perm_ok = len(ctr_perm_neg_cos) > 0 and (not np.all(np.isnan(np.asarray(ctr_perm_neg_cos, dtype=float))))
        plot_contrastive_geometry(
            out_ctr_plot,
            epochs,
            ctr_pos_cos,
            ctr_neg_cos,
            ctr_perm_neg_cos if perm_ok else None,
        )

    # ---------------------
    # Comprehensive validation analysis (VAL loader)
    # ---------------------
    print("\nPerforming comprehensive validation analysis (VAL split)...")
    validation_dir = Path(args.out_debug_dir).parent / "validation_analysis"
    validation_metrics = comprehensive_validation_analysis(
        model=model,
        loader=val_dl,
        device=device,
        output_dir=validation_dir,
        mask_id=int(args.mask_id),
        p_mask_site=float(args.p_mask_site),
    )

    final_metrics = {
        "final_validation_accuracy": float(validation_metrics["accuracy"]),
        "final_validation_auc": float(validation_metrics["auc"]),
        "total_masked_sites_analyzed": int(validation_metrics["total_masked_sites"]),
        "best_epoch": int(best_epoch) if es.enabled else None,
        "best_monitor": str(es.monitor) if es.enabled else None,
        "best_metric": float(best_metric) if (es.enabled and not np.isnan(best_metric)) else None,
    }

    import json
    final_metrics_path = Path(args.out_debug_dir).parent / "final_validation_metrics.json"
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Final validation metrics saved to: {final_metrics_path}")
    print(f"Detailed validation analysis saved to: {validation_dir}")


if __name__ == "__main__":
    main()
