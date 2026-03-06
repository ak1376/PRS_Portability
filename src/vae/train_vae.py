#!/usr/bin/env python3
"""
snakemake_scripts/train_vae.py

Minimal Lightning trainer that matches what src/vae/lit_model.py expects:
- cfg has:
  - cfg.input_len, cfg.latent_dim, cfg.hidden_channels, cfg.kernel_size, cfg.stride, cfg.padding, cfg.use_batchnorm
  - cfg.seed
  - cfg.training.{lr,beta,weight_decay,batch_size,max_epochs,log_every_n_steps,num_workers,accelerator,devices,strategy,precision}
  - cfg.masking.{enabled,alpha_masked,n_blocks,allow_overlap,mask_frac,block_len,fill,gaussian_std,constant_value}

It writes:
- outdir/hparams.resolved.yaml
- outdir/train_summary.json
- outdir/checkpoints/best.ckpt
- outdir/logs/version_*/metrics.csv and outdir/tb/version_*/events...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import yaml

from src.vae.lit_model import LitVAE
from src.masking import make_mask_and_apply


# ----------------------------
# utils
# ----------------------------
def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def as_tuple_int(x: Any, name: str) -> tuple[int, ...]:
    if isinstance(x, tuple):
        return tuple(int(v) for v in x)
    if isinstance(x, list):
        return tuple(int(v) for v in x)
    raise ValueError(f"{name} must be a list/tuple of ints, got {type(x)}")


def make_loader(X: np.ndarray, batch_size: int, *, shuffle: bool, num_workers: int) -> DataLoader:
    Xt = torch.from_numpy(X).float()
    ds = TensorDataset(Xt)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def _maybe_int(x: Any) -> Any:
    # allow YAML "1" -> 1, keep "auto" etc.
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return x


# ----------------------------
# Reconstruction saving helpers
# ----------------------------
def _model_recon(lit: LitVAE, x: torch.Tensor) -> torch.Tensor:
    """
    Run LitVAE's underlying model and return recon with shape (B, L).
    """
    lit.eval()
    with torch.no_grad():
        out = lit.model(x)
        # Handle different forward signatures: recon, (recon, mu, logvar), or dict
        if isinstance(out, torch.Tensor):
            recon = out
        elif isinstance(out, (tuple, list)):
            recon = out[0]
        elif isinstance(out, dict):
            recon = out.get("recon", out.get("x_recon"))
        else:
            raise RuntimeError(f"Unexpected model output type: {type(out)}")
    if recon.ndim != 2:
        recon = recon.view(recon.shape[0], -1)
    return recon


def dump_recon_artifacts(
    *,
    outdir: Path,
    lit: LitVAE,
    X: np.ndarray,
    split_name: str,
    n_samples: int,
    seed: int,
    mask_cfg: Dict[str, Any],
) -> None:
    """
    Save reconstruction artifacts for analysis.
    
    Saves recon/{split}_recon.npz with:
      - x_true: ground truth (n_samples, L)
      - x_masked: masked input fed to VAE (n_samples, L)
      - mask: boolean mask where True = masked position (n_samples, L)
      - recon: VAE reconstruction (n_samples, L)
    
    This allows scatterplots of recon vs x_true at masked positions.
    """
    out_recon_dir = outdir / "recon"
    out_recon_dir.mkdir(parents=True, exist_ok=True)

    if X.shape[0] == 0:
        return

    n = min(int(n_samples), X.shape[0])
    x_true = X[:n].astype(np.float32, copy=False)
    device = lit.device

    x_true_t = torch.from_numpy(x_true).to(device=device, dtype=torch.float32)

    # Apply masking using the same masking code as training
    split_offset = {"train": 11_111, "val": 22_222, "target": 33_333}.get(split_name, 0)
    dump_seed = int(seed) + int(split_offset)

    x_masked_t, mask_t, _ = make_mask_and_apply(
        x_true_t,
        enabled=mask_cfg.get("enabled", False),
        n_blocks=mask_cfg.get("n_blocks", 1),
        block_len=mask_cfg.get("block_len"),
        mask_frac=mask_cfg.get("mask_frac"),
        allow_overlap=mask_cfg.get("allow_overlap", True),
        seed=dump_seed,
        fill=mask_cfg.get("fill", "zero"),
        gaussian_std=mask_cfg.get("gaussian_std", 0.1),
        constant_value=mask_cfg.get("constant_value", 0.0),
    )

    # Get reconstruction from masked input
    recon_t = _model_recon(lit, x_masked_t)

    # Convert to numpy
    x_masked = x_masked_t.detach().cpu().numpy().astype(np.float32)
    mask = mask_t.detach().cpu().numpy().astype(np.bool_)
    recon = recon_t.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(
        out_recon_dir / f"{split_name}_recon.npz",
        x_true=x_true,
        x_masked=x_masked,
        mask=mask,
        recon=recon,
    )
    print(f"Saved {split_name} reconstructions to {out_recon_dir / f'{split_name}_recon.npz'}")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train VAE (Lightning) with minimal config plumbing.")
    ap.add_argument("--train", type=Path, required=True, help=".npy (N,L) train array")
    ap.add_argument("--val", type=Path, required=True, help=".npy (N,L) val array")
    ap.add_argument("--target", type=Path, default=None, help="Optional .npy (N,L) target array (2nd val dataloader)")
    ap.add_argument("--hparams", type=Path, required=True, help="YAML with model/training/masking/seed")
    ap.add_argument("--outdir", type=Path, required=True)

    # small optional overrides (keeps script snakemake-friendly)
    ap.add_argument("--no-progress-bar", action="store_true")
    ap.add_argument("--accelerator", type=str, default=None)
    ap.add_argument("--devices", type=str, default=None)
    ap.add_argument("--strategy", type=str, default=None)
    ap.add_argument("--precision", type=str, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--log-every-n-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)

    # reconstruction saving
    ap.add_argument("--save-recon", action="store_true", help="Save reconstruction artifacts after training")
    ap.add_argument("--recon-n", type=int, default=100, help="Number of samples to save for reconstruction analysis")
    ap.add_argument("--recon-splits", type=str, default="val,target", help="Comma-separated splits to save: train,val,target")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    hp = read_yaml(args.hparams)
    model_hp = hp.get("model", {}) or {}
    train_hp = hp.get("training", {}) or {}
    mask_hp = hp.get("masking", {}) or {}
    seed = int(hp.get("seed", 0))

    pl.seed_everything(seed, workers=True)

    # -------------------------
    # load arrays
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

    X_target: Optional[np.ndarray] = None
    if args.target is not None:
        X_target = np.load(args.target)
        if X_target.ndim != 2:
            raise ValueError(f"Expected 2D target array. Got target {X_target.shape}")
        if X_target.shape[1] != input_len:
            raise ValueError(f"Target must have same #features as train/val. Got {X_target.shape[1]} vs {input_len}")
        if not np.isfinite(X_target).all():
            raise ValueError("Non-finite values found in target array.")

    # -------------------------
    # resolve training knobs (with CLI overrides)
    # -------------------------
    batch_size = int(args.batch_size if args.batch_size is not None else train_hp.get("batch_size", 256))
    max_epochs = int(args.max_epochs if args.max_epochs is not None else train_hp.get("max_epochs", 50))
    log_every_n_steps = int(
        args.log_every_n_steps if args.log_every_n_steps is not None else train_hp.get("log_every_n_steps", 1)
    )
    num_workers = int(args.num_workers if args.num_workers is not None else train_hp.get("num_workers", 0))

    accelerator = str(args.accelerator if args.accelerator is not None else train_hp.get("accelerator", "auto"))
    devices: Any = args.devices if args.devices is not None else train_hp.get("devices", "auto")
    strategy = str(args.strategy if args.strategy is not None else train_hp.get("strategy", "auto"))
    precision = str(args.precision if args.precision is not None else train_hp.get("precision", "32-true"))
    
    # Gradient clipping (None means no clipping)
    gradient_clip_val = train_hp.get("gradient_clip_val", None)
    if gradient_clip_val is not None:
        gradient_clip_val = float(gradient_clip_val)

    devices = _maybe_int(devices)

    # -------------------------
    # build cfg for LitVAE
    # -------------------------
    # Handle padding - can be None for auto-calculation in FullyConvVAE1D
    padding_val = model_hp.get("padding", 4)
    if padding_val is not None:
        padding_val = int(padding_val)
    
    cfg = SimpleNamespace(
        # model
        input_len=input_len,
        model_type=str(model_hp.get("model_type", "conv")),  # "conv" or "fully_conv"
        latent_dim=int(model_hp.get("latent_dim", 32)),
        hidden_channels=as_tuple_int(model_hp.get("hidden_channels", [32, 64, 128]), "hidden_channels"),
        kernel_size=int(model_hp.get("kernel_size", 9)),
        stride=int(model_hp.get("stride", 2)),
        padding=padding_val,
        use_batchnorm=bool(model_hp.get("use_batchnorm", False)),
        # seed for deterministic masking
        seed=seed,
        # training (LitVAE will read cfg.training.*)
        training=SimpleNamespace(
            lr=float(train_hp.get("lr", 1e-3)),
            beta=float(train_hp.get("beta", 0.01)),
            weight_decay=float(train_hp.get("weight_decay", 0.0)),
        ),
        # masking (LitVAE will read cfg.masking.*)
        masking=SimpleNamespace(
            enabled=bool(mask_hp.get("enabled", False)),
            alpha_masked=float(mask_hp.get("alpha_masked", 1.0)),
            n_blocks=int(mask_hp.get("n_blocks", 1)),
            allow_overlap=bool(mask_hp.get("allow_overlap", True)),
            mask_frac=mask_hp.get("mask_frac", None),
            block_len=mask_hp.get("block_len", None),
            fill=str(mask_hp.get("fill", mask_hp.get("fill_value", "gaussian"))),
            gaussian_std=float(mask_hp.get("gaussian_std", 0.1)),
            constant_value=float(mask_hp.get("constant_value", 0.0)),
        ),
    )

    # -------------------------
    # dataloaders
    # -------------------------
    train_loader = make_loader(X_train, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(X_val, batch_size, shuffle=False, num_workers=num_workers)
    val_dataloaders = [val_loader]
    if X_target is not None:
        target_loader = make_loader(X_target, batch_size, shuffle=False, num_workers=num_workers)
        val_dataloaders.append(target_loader)

    # -------------------------
    # write resolved config (for plotting/debugging)
    # -------------------------
    resolved = {
        "seed": seed,
        "data": {
            "train": str(args.train),
            "val": str(args.val),
            "target": (str(args.target) if args.target is not None else None),
            "input_len": input_len,
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_target": (int(X_target.shape[0]) if X_target is not None else None),
        },
        "model": {
            "latent_dim": cfg.latent_dim,
            "hidden_channels": list(cfg.hidden_channels),
            "kernel_size": cfg.kernel_size,
            "stride": cfg.stride,
            "padding": cfg.padding,
            "use_batchnorm": cfg.use_batchnorm,
        },
        "training": {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "lr": cfg.training.lr,
            "beta": cfg.training.beta,
            "weight_decay": cfg.training.weight_decay,
            "precision": precision,
            "log_every_n_steps": log_every_n_steps,
            "num_workers": num_workers,
            "accelerator": accelerator,
            "devices": devices,
            "strategy": strategy,
        },
        "masking": {
            "enabled": cfg.masking.enabled,
            "alpha_masked": cfg.masking.alpha_masked,
            "n_blocks": cfg.masking.n_blocks,
            "allow_overlap": cfg.masking.allow_overlap,
            "mask_frac": cfg.masking.mask_frac,
            "block_len": cfg.masking.block_len,
            "fill": cfg.masking.fill,
            "gaussian_std": cfg.masking.gaussian_std,
            "constant_value": cfg.masking.constant_value,
        },
    }
    (outdir / "hparams.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    # -------------------------
    # module
    # -------------------------
    lit = LitVAE(cfg)

    # -------------------------
    # callbacks + loggers
    # -------------------------
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val/loss",  # IMPORTANT: matches the rewritten lit_model.py
        mode="min",
        save_top_k=1,
        filename="vae-{epoch:03d}",
    )

    lr_cb = LearningRateMonitor(logging_interval="epoch")

    tb_logger = TensorBoardLogger(save_dir=str(outdir), name="tb")
    csv_logger = CSVLogger(save_dir=str(outdir), name="logs")
    logger = [tb_logger, csv_logger]

    # -------------------------
    # trainer
    # -------------------------
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        callbacks=[ckpt_cb, lr_cb],
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=str(outdir),
        enable_progress_bar=(not args.no_progress_bar),
        enable_model_summary=False,
    )

    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_dataloaders)

    # -------------------------
    # summary + stable best.ckpt
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

    # sanity output (snakemake-friendly)
    logs_root = outdir / "logs"
    metrics = list(logs_root.glob("version_*/metrics.csv"))
    if not metrics:
        raise RuntimeError(f"No metrics.csv found under {logs_root}/version_*/metrics.csv")
    (outdir / "logs_ok.txt").write_text(str(metrics[0]) + "\n")

    print("Best checkpoint:", best_out)
    print("TensorBoard logdir:", outdir / "tb")

    # -------------------------
    # Save reconstruction artifacts (optional)
    # -------------------------
    if args.save_recon:
        # Load best model for reconstruction
        lit_best = LitVAE.load_from_checkpoint(str(best_out), cfg=cfg)
        lit_best.eval()
        
        # Move to appropriate device
        if torch.cuda.is_available() and accelerator != "cpu":
            lit_best = lit_best.to("cuda")
        
        # Parse which splits to save
        splits_to_save = [s.strip().lower() for s in args.recon_splits.split(",") if s.strip()]
        
        # Prepare mask config dict for dump_recon_artifacts
        mask_cfg_dict = {
            "enabled": bool(mask_hp.get("enabled", False)),
            "n_blocks": int(mask_hp.get("n_blocks", 1)),
            "block_len": mask_hp.get("block_len"),
            "mask_frac": mask_hp.get("mask_frac"),
            "allow_overlap": bool(mask_hp.get("allow_overlap", True)),
            "fill": str(mask_hp.get("fill", mask_hp.get("fill_value", "gaussian"))),
            "gaussian_std": float(mask_hp.get("gaussian_std", 0.1)),
            "constant_value": float(mask_hp.get("constant_value", 0.0)),
        }
        
        for split in splits_to_save:
            if split == "train":
                X = X_train
            elif split == "val":
                X = X_val
            elif split == "target":
                if X_target is None:
                    print(f"Skipping target split (no target data provided)")
                    continue
                X = X_target
            else:
                print(f"Unknown split '{split}', skipping")
                continue
            
            dump_recon_artifacts(
                outdir=outdir,
                lit=lit_best,
                X=X,
                split_name=split,
                n_samples=args.recon_n,
                seed=seed,
                mask_cfg=mask_cfg_dict,
            )


if __name__ == "__main__":
    main()