#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import yaml

from src.vae.lit_model import LitVAE
from src.vae.model import VAEConfig


# ----------------------------
# IO helpers
# ----------------------------
def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def as_tuple_int(x: Any, name: str) -> Tuple[int, ...]:
    if isinstance(x, tuple):
        return tuple(int(v) for v in x)
    if isinstance(x, list):
        return tuple(int(v) for v in x)
    raise ValueError(f"{name} must be a list/tuple of ints, got {type(x)}")


def make_loader(
    X: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    ds = TensorDataset(X_t)  # yields (x,)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train VAE with PyTorch Lightning (Snakemake-friendly).")
    ap.add_argument("--train", type=Path, required=True, help=".npy (N,L) train array")
    ap.add_argument("--val", type=Path, required=True, help=".npy (N,L) val array")
    ap.add_argument("--target", type=Path, default=None, help="Optional .npy (N,L) target array for eval-only logging")
    ap.add_argument("--hparams", type=Path, required=True, help="YAML with seed/model/training/masking sections")
    ap.add_argument("--outdir", type=Path, required=True)

    # Optional overrides (default: honor YAML)
    ap.add_argument("--accelerator", type=str, default=None)
    ap.add_argument("--devices", type=str, default=None)
    ap.add_argument("--strategy", type=str, default=None)
    ap.add_argument("--precision", type=str, default=None)
    ap.add_argument("--no-progress-bar", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    hp = read_yaml(args.hparams)
    model_hp = hp.get("model", {}) or {}
    train_hp = hp.get("training", {}) or {}
    mask_hp = hp.get("masking", {}) or {}
    seed = int(hp.get("seed", 0))

    pl.seed_everything(seed, workers=True)

    # -------------------------
    # Load arrays
    # -------------------------
    X_train = np.load(args.train)
    X_val = np.load(args.val)

    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError(f"Expected 2D arrays. Got train {X_train.shape}, val {X_val.shape}")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(f"Train/val must have same #features. Got {X_train.shape[1]} vs {X_val.shape[1]}")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Non-finite values found in train/val arrays.")

    input_len = int(X_train.shape[1])

    X_target = None
    if args.target is not None:
        X_target = np.load(args.target)

        if X_target.ndim != 2:
            raise ValueError(f"Expected 2D target array. Got target {X_target.shape}")
        if X_target.shape[1] != input_len:
            raise ValueError(f"Target must have same #features as train/val. Got {X_target.shape[1]} vs {input_len}")
        if not np.isfinite(X_target).all():
            raise ValueError("Non-finite values found in target array.")

    # -------------------------
    # Build cfg (includes masking knobs)
    # -------------------------
    cfg = VAEConfig(
        input_len=input_len,
        latent_dim=int(model_hp.get("latent_dim", 32)),
        hidden_channels=as_tuple_int(model_hp.get("hidden_channels", [32, 64, 128]), "hidden_channels"),
        kernel_size=int(model_hp.get("kernel_size", 9)),
        stride=int(model_hp.get("stride", 2)),
        padding=int(model_hp.get("padding", 4)),
        use_batchnorm=bool(model_hp.get("use_batchnorm", True)),
        lr=float(train_hp.get("lr", 1e-3)),
        beta=float(train_hp.get("beta", 1.0)),
        weight_decay=float(train_hp.get("weight_decay", 0.0)),
        # Masking
        mask_enabled=bool(mask_hp.get("enabled", False)),
        mask_block_len=int(mask_hp.get("block_len", 0)),
        mask_fill_value=str(mask_hp.get("fill_value", "mean")),
        weight_masked=float(mask_hp.get("weight_masked", 1.0)),
        weight_unmasked=float(mask_hp.get("weight_unmasked", 0.0)),
    )

    # -------------------------
    # Training HPs
    # -------------------------
    batch_size = int(train_hp.get("batch_size", 256))
    max_epochs = int(train_hp.get("max_epochs", 50))
    log_every_n_steps = int(train_hp.get("log_every_n_steps", 1))
    num_workers = int(train_hp.get("num_workers", 0))

    accelerator = str(args.accelerator) if args.accelerator is not None else str(train_hp.get("accelerator", "auto"))
    devices: Any = args.devices if args.devices is not None else train_hp.get("devices", "auto")
    strategy = str(args.strategy) if args.strategy is not None else str(train_hp.get("strategy", "auto"))
    precision = str(args.precision) if args.precision is not None else str(train_hp.get("precision", "32-true"))

    # Normalize devices: allow "1" -> 1, keep "auto"
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = make_loader(X_train, batch_size, shuffle=True, num_workers=num_workers)

    val_loader = make_loader(X_val, batch_size, shuffle=False, num_workers=num_workers)
    val_dataloaders = [val_loader]

    if X_target is not None:
        target_loader = make_loader(X_target, batch_size, shuffle=False, num_workers=num_workers)
        val_dataloaders.append(target_loader)

    # -------------------------
    # Write resolved YAML for plotting later (NOW includes masking)
    # -------------------------
    resolved = {
        "seed": seed,
        "data": {
            "train": str(args.train),
            "val": str(args.val),
            "target": (str(args.target) if args.target is not None else None),
            "input_len": input_len,
        },
        "model": {
            "latent_dim": cfg.latent_dim,
            "hidden_channels": list(cfg.hidden_channels),
            "kernel_size": cfg.kernel_size,
            "stride": cfg.stride,
            "padding": cfg.padding,
            "use_batchnorm": cfg.use_batchnorm,
        },
        "masking": {
            "enabled": cfg.mask_enabled,
            "block_len": cfg.mask_block_len,
            "fill_value": cfg.mask_fill_value,
            "weight_masked": cfg.weight_masked,
            "weight_unmasked": cfg.weight_unmasked,
        },
        "training": {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "lr": cfg.lr,
            "beta": cfg.beta,
            "weight_decay": cfg.weight_decay,
            "precision": precision,
            "log_every_n_steps": log_every_n_steps,
            "num_workers": num_workers,
            "accelerator": accelerator,
            "devices": devices,
            "strategy": strategy,
        },
    }
    (outdir / "hparams.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    # -------------------------
    # Module
    # -------------------------
    lit = LitVAE(cfg)

    # -------------------------
    # Callbacks
    # -------------------------
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val/loss_epoch/dataloader_idx_0",  # discovery val
        mode="min",
        save_top_k=1,
        filename="vae-{epoch:03d}",
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    # -------------------------
    # Loggers (TensorBoard + CSV)
    # -------------------------
    tb_logger = TensorBoardLogger(save_dir=str(outdir), name="tb")   # outdir/tb/version_*/events...
    csv_logger = CSVLogger(save_dir=str(outdir), name="logs")        # outdir/logs/version_*/metrics.csv
    logger = [tb_logger, csv_logger]

    # -------------------------
    # Trainer
    # -------------------------
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        callbacks=[ckpt_cb, lr_cb],
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=str(outdir),
        enable_progress_bar=(not args.no_progress_bar),
        enable_model_summary=False,
    )

    # -------------------------
    # Fit
    # -------------------------
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_dataloaders)

    # -------------------------
    # Save summary + stable best.ckpt
    # -------------------------
    summary = {
        "best_model_path": ckpt_cb.best_model_path,
        "best_val_loss": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None,
    }
    write_json(summary, outdir / "train_summary.json")

    best = Path(summary["best_model_path"]) if summary["best_model_path"] else None
    if best is None or not best.exists():
        raise RuntimeError(f"Best checkpoint path invalid: {summary['best_model_path']}")

    best_out = ckpt_dir / "best.ckpt"
    shutil.copy2(best, best_out)

    # -------------------------
    # Snakemake sanity outputs
    # -------------------------
    logs_root = outdir / "logs"
    metrics = list(logs_root.glob("version_*/metrics.csv"))
    if not metrics:
        raise RuntimeError(f"No metrics.csv found under {logs_root}/version_*/metrics.csv")
    (outdir / "logs_ok.txt").write_text(str(metrics[0]) + "\n")

    print("Best checkpoint:", best_out)
    print("TensorBoard logdir:", outdir / "tb")


if __name__ == "__main__":
    main()