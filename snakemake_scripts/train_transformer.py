#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# -----------------------------------------------------------------------------
# Repo import path
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Project imports (thin runner: all logic lives in src/transformer/*)
# -----------------------------------------------------------------------------
from src.transformer.data_class import HapDataset, collate_hapbatch
from src.transformer.model import HapMaskTransformer

from src.transformer.splitting import make_train_val_test_loaders
from src.transformer.early_stopping import EarlyStopConfig, EarlyStopper

from src.transformer.train import train_epoch, eval_epoch_losses, debug_snapshot_and_pngs
from src.transformer.metrics import eval_masked_acc_auc, comprehensive_validation_analysis
from src.transformer.io_utils import write_losses_csv
from src.transformer.plots import plot_losses, plot_contrastive_geometry


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


def _resolve_fracs(
    train_frac: float | None,
    val_frac: float,
    test_frac: float,
) -> tuple[float, float, float]:
    """
    Accept:
      - train_frac=None -> infer train = 1 - val - test
      - test_frac can be 0
    Renormalize ONLY if sum is not ~1 AND train_frac was provided explicitly.
    """
    val_frac = float(val_frac)
    test_frac = float(test_frac)

    inferred_train = (train_frac is None)
    if inferred_train:
        train_frac = 1.0 - val_frac - test_frac
    train_frac = float(train_frac)

    for name, v in [("train_frac", train_frac), ("val_frac", val_frac), ("test_frac", test_frac)]:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {v}")

    s = train_frac + val_frac + test_frac
    if s <= 0:
        raise ValueError("train/val/test fracs sum to 0; set at least one > 0")

    # Only renormalize if the user explicitly provided train_frac (i.e., not inferred).
    if abs(s - 1.0) > 1e-6 and (not inferred_train):
        train_frac /= s
        val_frac /= s
        test_frac /= s

    if train_frac <= 0.0:
        raise ValueError("train_frac ended up 0; need non-empty training set.")
    if val_frac <= 0.0:
        raise ValueError("val_frac ended up 0; need non-empty validation set.")
    # test_frac may be 0

    return float(train_frac), float(val_frac), float(test_frac)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--hap", type=str, required=True, help="Path to ONE haplotype .npy (N,L)")
    ap.add_argument("--out_model", type=str, required=True)
    ap.add_argument("--out_losses", type=str, required=True)
    ap.add_argument("--out_plot", type=str, required=True)
    ap.add_argument("--out_debug_dir", type=str, required=True, help="Directory for debug PNGs")
    ap.add_argument("--out_ctr_plot", type=str, default=None, help="Optional: contrastive geometry plot path")
    ap.add_argument("--out_total_plot", type=str, default=None, help="Optional: plot for total losses")

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

    # Train/val/test split overrides
    ap.add_argument("--train_frac", type=float, default=None)
    ap.add_argument("--val_frac", type=float, default=None)
    ap.add_argument("--test_frac", type=float, default=None)

    # Early stopping overrides
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

    return ap


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()

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

    # Splits (defaults keep old behavior: val=0.1, test=0.0, train=rest)
    yaml_train_frac = train_cfg.get("train_frac", None)
    yaml_val_frac = train_cfg.get("val_frac", 0.1)
    yaml_test_frac = train_cfg.get("test_frac", 0.0)

    args.train_frac = _pick(args.train_frac, yaml_train_frac, None)
    args.val_frac = _pick(args.val_frac, yaml_val_frac, 0.1)
    args.test_frac = _pick(args.test_frac, yaml_test_frac, 0.0)

    args.train_frac, args.val_frac, args.test_frac = _resolve_fracs(
        args.train_frac, float(args.val_frac), float(args.test_frac)
    )

    # Early stopping (YAML defaults)
    es_yaml = train_cfg.get("early_stopping", {}) or {}
    es_cfg = EarlyStopConfig(
        enabled=bool(_pick((True if args.early_stop else None), es_yaml.get("enabled"), False)),
        monitor=str(_pick(args.early_monitor, es_yaml.get("monitor"), "val_total_loss")),
        mode=str(_pick(args.early_mode, es_yaml.get("mode"), "min")),
        patience=int(_pick(args.early_patience, es_yaml.get("patience"), 25)),
        min_delta=float(_pick(args.early_min_delta, es_yaml.get("min_delta"), 1e-4)),
        burn_in=int(_pick(args.early_burn_in, es_yaml.get("burn_in"), 0)),
    )
    stopper = EarlyStopper(es_cfg)

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
    print(
        f"[train_transformer_single] window_len={args.window_len} window_mode={args.window_mode} "
        f"fixed_start={args.fixed_start}"
    )
    print(f"[train_transformer_single] masking p_mask_site={args.p_mask_site} mask_id={args.mask_id}")
    print(f"[split_fracs] train={args.train_frac:.3f} val={args.val_frac:.3f} test={args.test_frac:.3f}")
    print(
        f"[train_transformer_single] contrastive={args.contrastive} "
        f"lambda={args.contrastive_lambda} tau={args.contrastive_tau} "
        f"perm_neg={args.permute_negatives} permute_every_k={args.permute_every_k} "
        f"p_mask_site_ctr={args.p_mask_site_ctr}"
    )
    print(
        f"[early_stopping] enabled={es_cfg.enabled} monitor={es_cfg.monitor} mode={es_cfg.mode} "
        f"patience={es_cfg.patience} min_delta={es_cfg.min_delta} burn_in={es_cfg.burn_in}"
    )

    # prior for bias init
    with torch.no_grad():
        pi = float(hap.float().mean().item())
        pi = min(max(pi, 1e-4), 1.0 - 1e-4)
        logit_pi = float(np.log(pi / (1.0 - pi)))
        print(f"[init] pi={pi:.6f} logit_pi={logit_pi:.6f}")

    # ---------------------
    # Dataset + loaders
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

    train_dl, val_dl, test_dl, n_tr, n_va, n_te = make_train_val_test_loaders(
        ds,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        device=device,
        collate_fn=collate_hapbatch,
    )
    print(f"[split_counts] n_train={n_tr} n_val={n_va} n_test={n_te}")

    # ---------------------
    # Model + optimizer
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

    with torch.no_grad():
        model.head.bias.zero_()
        model.head.bias[1] = logit_pi

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    # ---------------------
    # Outputs / tracking
    # ---------------------
    out_debug = Path(args.out_debug_dir)
    out_debug.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    epochs: list[int] = []

    train_mlm_losses: list[float] = []
    val_mlm_losses: list[float] = []
    train_total_losses: list[float] = []
    val_total_losses: list[float] = []

    ctr_pos_cos: list[float] = []
    ctr_neg_cos: list[float] = []
    ctr_perm_neg_cos: list[float] = []

    valid_monitors = {"val_total_loss", "val_mlm_loss", "val_accuracy", "val_auc"}
    if es_cfg.monitor not in valid_monitors:
        raise ValueError(f"early_stopping.monitor must be one of {sorted(valid_monitors)}, got {es_cfg.monitor}")

    # ---------------------
    # Train loop
    # ---------------------
    last_epoch = 0
    for ep in range(1, int(args.epochs) + 1):
        last_epoch = ep

        tr_out = train_epoch(
            model=model,
            loader=train_dl,
            optimizer=opt,
            device=device,
            mask_id=int(args.mask_id),
            p_mask_site=float(args.p_mask_site),
            p_mask_site_ctr=(None if args.p_mask_site_ctr is None else float(args.p_mask_site_ctr)),
            grad_clip=float(args.grad_clip),
            use_contrastive=bool(args.contrastive),
            contrastive_lambda=float(args.contrastive_lambda),
            contrastive_tau=float(args.contrastive_tau),
            use_perm_negatives=bool(args.permute_negatives),
            permute_every_k=int(args.permute_every_k),
        )

        tr_mlm = float(tr_out["mlm_loss"])
        tr_total = float(tr_out["total_loss"])
        tr_ctr = float(tr_out.get("ctr_loss", float("nan")))
        pos_cos = float(tr_out.get("ctr_pos_cos", float("nan")))
        neg_cos = float(tr_out.get("ctr_neg_cos", float("nan")))
        perm_cos = float(tr_out.get("ctr_perm_neg_cos", float("nan")))

        va_out = eval_epoch_losses(
            model=model,
            loader=val_dl,
            device=device,
            mask_id=int(args.mask_id),
            p_mask_site=float(args.p_mask_site),
            class_balance=True,
            use_contrastive=bool(args.contrastive),
            contrastive_lambda=float(args.contrastive_lambda),
            contrastive_tau=float(args.contrastive_tau),
            p_mask_site_ctr=(None if args.p_mask_site_ctr is None else float(args.p_mask_site_ctr)),
            use_perm_negatives=bool(args.permute_negatives),
            permute_every_k=int(args.permute_every_k),
        )
        va_mlm = float(va_out["mlm_loss"])
        va_total = float(va_out["total_loss"])
        va_ctr = float(va_out["ctr_loss"])

        # acc/auc only if needed
        val_acc = float("nan")
        val_auc = float("nan")
        if es_cfg.monitor in ("val_accuracy", "val_auc"):
            m = eval_masked_acc_auc(
                model=model,
                loader=val_dl,
                device=device,
                mask_id=int(args.mask_id),
                p_mask_site=float(args.p_mask_site),
            )
            val_acc = float(m["val_accuracy"])
            val_auc = float(m["val_auc"])

        # -------- logging --------
        if args.contrastive:
            msg = (
                f"[epoch {ep:03d}] "
                f"train_mlm={tr_mlm:.6f} train_total={tr_total:.6f} train_ctr={tr_ctr:.6f} "
                f"val_mlm={va_mlm:.6f} val_total={va_total:.6f} val_ctr={va_ctr:.6f} "
                f"pos_cos={pos_cos:.4f} neg_cos={neg_cos:.4f}"
            )
            if not np.isnan(perm_cos):
                msg += f" perm_neg_cos={perm_cos:.4f}"
            if es_cfg.monitor in ("val_accuracy", "val_auc"):
                msg += f" val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
            print(msg)
        else:
            msg = (
                f"[epoch {ep:03d}] "
                f"train_mlm={tr_mlm:.6f} train_total={tr_total:.6f} "
                f"val_mlm={va_mlm:.6f} val_total={va_total:.6f}"
            )
            if es_cfg.monitor in ("val_accuracy", "val_auc"):
                msg += f" val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
            print(msg)

        # -------- debug snapshots --------
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

        # -------- record row --------
        row: dict[str, Any] = {
            "epoch": ep,
            "train_mlm_loss": tr_mlm,
            "train_total_loss": tr_total,
            "val_mlm_loss": va_mlm,
            "val_total_loss": va_total,
        }
        if args.contrastive:
            row["train_ctr_loss"] = tr_ctr
            row["val_ctr_loss"] = va_ctr
            row["ctr_pos_cos"] = pos_cos
            row["ctr_neg_cos"] = neg_cos
            if not np.isnan(perm_cos):
                row["ctr_perm_neg_cos"] = perm_cos
        if es_cfg.monitor in ("val_accuracy", "val_auc"):
            row["val_accuracy"] = val_acc
            row["val_auc"] = val_auc

        rows.append(row)

        epochs.append(ep)
        train_mlm_losses.append(tr_mlm)
        val_mlm_losses.append(va_mlm)
        train_total_losses.append(tr_total)
        val_total_losses.append(va_total)

        if args.contrastive:
            ctr_pos_cos.append(pos_cos)
            ctr_neg_cos.append(neg_cos)
            ctr_perm_neg_cos.append(perm_cos)

        # -------- early stopping --------
        if es_cfg.enabled:
            if es_cfg.monitor == "val_total_loss":
                curr = va_total
            elif es_cfg.monitor == "val_mlm_loss":
                curr = va_mlm
            elif es_cfg.monitor == "val_accuracy":
                curr = val_acc
            elif es_cfg.monitor == "val_auc":
                curr = val_auc
            else:
                curr = float("nan")

            prev_best_epoch = stopper.best_epoch
            prev_bad = stopper.bad_epochs

            should_stop = stopper.step(curr_metric=float(curr), epoch=int(ep), model=model)

            # nice logs without assuming stopper.is_best exists
            if stopper.best_epoch != prev_best_epoch:
                if not np.isnan(stopper.best_metric):
                    print(f"[early_stopping] new best {es_cfg.monitor}={stopper.best_metric:.6f} at epoch {stopper.best_epoch}")
                else:
                    print(f"[early_stopping] new best at epoch {stopper.best_epoch}")

            if stopper.bad_epochs != prev_bad and ep >= int(es_cfg.burn_in):
                print(f"[early_stopping] no improvement (bad_epochs={stopper.bad_epochs}/{es_cfg.patience})")

            if should_stop:
                if not np.isnan(stopper.best_metric):
                    print(
                        f"[early_stopping] STOP at epoch {ep} "
                        f"(best epoch={stopper.best_epoch}, best {es_cfg.monitor}={stopper.best_metric:.6f})"
                    )
                else:
                    print(f"[early_stopping] STOP at epoch {ep} (best epoch={stopper.best_epoch})")
                break

    # Restore best weights (if early stopping enabled)
    if es_cfg.enabled:
        restored = stopper.restore_best(model)
        if restored:
            if not np.isnan(stopper.best_metric):
                print(
                    f"[early_stopping] restored best model from epoch {stopper.best_epoch} "
                    f"({es_cfg.monitor}={stopper.best_metric:.6f})"
                )
            else:
                print(f"[early_stopping] restored best model from epoch {stopper.best_epoch}")

    # ---------------------
    # Save checkpoint
    # ---------------------
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    ckpt_cfg = {
        "vocab_size": int(args.vocab_size),
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "n_layers": int(args.n_layers),
        "dropout": float(args.dropout),
        "pad_id": args.pad_id,
        "pool": str(args.pool),
        "max_len": int(args.max_len),
        "lr": float(args.lr),
        "epochs_requested": int(args.epochs),
        "epochs_ran": int(last_epoch),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "splits": {
            "train_frac": float(args.train_frac),
            "val_frac": float(args.val_frac),
            "test_frac": float(args.test_frac),
            "n_train": int(n_tr),
            "n_val": int(n_va),
            "n_test": int(n_te),
        },
        "early_stopping": {
            "enabled": bool(es_cfg.enabled),
            "monitor": str(es_cfg.monitor),
            "mode": str(es_cfg.mode),
            "patience": int(es_cfg.patience),
            "min_delta": float(es_cfg.min_delta),
            "burn_in": int(es_cfg.burn_in),
            "best_epoch": int(stopper.best_epoch) if es_cfg.enabled else None,
            "best_metric": (None if (not es_cfg.enabled or np.isnan(stopper.best_metric)) else float(stopper.best_metric)),
        },
        "p_mask_site": float(args.p_mask_site),
        "mask_id": int(args.mask_id),
        "contrastive": bool(args.contrastive),
        "contrastive_lambda": float(args.contrastive_lambda),
        "contrastive_tau": float(args.contrastive_tau),
        "permute_negatives": bool(args.permute_negatives),
        "permute_every_k": int(args.permute_every_k),
        "p_mask_site_ctr": (None if args.p_mask_site_ctr is None else float(args.p_mask_site_ctr)),
        "window_len": int(args.window_len) if args.window_len is not None else None,
        "window_mode": str(args.window_mode),
        "fixed_start": int(args.fixed_start),
    }

    torch.save({"state_dict": model.state_dict(), "config": ckpt_cfg}, out_model)

    # ---------------------
    # CSV + plots
    # ---------------------
    write_losses_csv(Path(args.out_losses), rows)

    plot_losses(Path(args.out_plot), epochs, train_mlm_losses, val_mlm_losses, ylabel="loss", title="MLM Loss")

    out_total_plot = (
        Path(args.out_total_plot)
        if args.out_total_plot is not None
        else Path(args.out_plot).with_name("loss_total.png")
    )
    plot_losses(out_total_plot, epochs, train_total_losses, val_total_losses, ylabel="loss", title="Total Loss (MLM + λ·CTR)")

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
    # Final evaluation: TEST if present else VAL
    # ---------------------
    use_test = (test_dl is not None and int(n_te) > 0)
    eval_loader = test_dl if use_test else val_dl
    split_name = "TEST" if use_test else "VAL"

    # Always write into param_dir/{test_analysis|validation_analysis}
    param_dir = Path(args.out_debug_dir).parent
    analysis_dir = param_dir / ("test_analysis" if use_test else "validation_analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPerforming comprehensive evaluation analysis ({split_name} split)...")
    metrics = comprehensive_validation_analysis(
        model=model,
        loader=eval_loader,
        device=device,
        output_dir=analysis_dir,
        mask_id=int(args.mask_id),
        p_mask_site=float(args.p_mask_site),
        split_name=split_name,
    )

    # Write the EXACT filename Snakemake expects at the param_dir root
    final_metrics = {
        f"final_{split_name.lower()}_accuracy": float(metrics["accuracy"]),
        f"final_{split_name.lower()}_auc": float(metrics["auc"]),
        "total_masked_sites_analyzed": int(metrics["total_masked_sites"]),
        "best_epoch": int(stopper.best_epoch) if es_cfg.enabled else None,
        "best_monitor": str(es_cfg.monitor) if es_cfg.enabled else None,
        "best_metric": (None if (not es_cfg.enabled or np.isnan(stopper.best_metric)) else float(stopper.best_metric)),
    }

    final_metrics_path = param_dir / f"final_{split_name.lower()}_metrics.json"
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Final metrics saved to: {final_metrics_path}")
    print(f"Detailed analysis saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
