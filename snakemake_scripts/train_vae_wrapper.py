#!/usr/bin/env python3
# /sietch_colab/akapoor/PRS_Portability/src/vae/training.py
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
# IO helpers
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


def _maybe_int(x: Any) -> Any:
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return x


def _none_if_null(x: Any) -> Any:
    # YAML null -> None (safe), but also treat ""/"None" as None
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"none", "null", ""}:
        return None
    return x


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


def dump_recon_artifacts(
    lit: LitVAE,
    X: np.ndarray,
    outdir: Path,
    split: str,
    n: int,
    mask_cfg: Dict[str, Any],
    seed: int,
) -> None:
    """
    Save x_true, x_masked, mask, recon for `n` samples.
    Uses masking settings from mask_cfg (taken from LitVAE._mask_cfg()).
    Saves to outdir/recon/{split}_recon.npz
    """
    device = next(lit.parameters()).device
    lit.eval()

    # limit samples
    X_sub = X[:n]
    x_true = torch.from_numpy(X_sub).float().to(device)

    # apply masking with same settings as training
    x_in, mask, _ = make_mask_and_apply(
        x_true,
        enabled=mask_cfg["enabled"],
        n_blocks=mask_cfg["n_blocks"],
        block_len=mask_cfg["block_len"],
        mask_frac=mask_cfg["mask_frac"],
        allow_overlap=mask_cfg["allow_overlap"],
        seed=seed,
        fill=mask_cfg["fill"],
        gaussian_std=mask_cfg["gaussian_std"],
        constant_value=mask_cfg["constant_value"],
    )

    with torch.no_grad():
        recon, _, _ = lit(x_in)  # forward returns (recon, mu, logvar)

    # Save to recon/ subdirectory
    recon_dir = outdir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)
    out_path = recon_dir / f"{split}_recon.npz"
    np.savez_compressed(
        out_path,
        x_true=x_true.cpu().numpy(),
        x_masked=x_in.cpu().numpy(),
        mask=mask.cpu().numpy(),
        recon=recon.cpu().numpy(),
    )
    print(f"Saved reconstruction artifacts to {out_path}")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Lightweight VAE training wrapper (Snakemake-friendly).")
    ap.add_argument("--train", type=Path, required=True, help=".npy (N,L) train array")
    ap.add_argument("--val", type=Path, required=True, help=".npy (N,L) val array")
    ap.add_argument("--target", type=Path, default=None, help="Optional .npy (N,L) target array (2nd val dataloader)")
    ap.add_argument("--hparams", type=Path, required=True, help="YAML with model/training/masking/seed sections")
    ap.add_argument("--outdir", type=Path, required=True)

    # optional overrides (nice for quick debugging)
    ap.add_argument("--no-progress-bar", action="store_true")
    ap.add_argument("--accelerator", type=str, default=None)
    ap.add_argument("--devices", type=str, default=None)
    ap.add_argument("--strategy", type=str, default=None)
    ap.add_argument("--precision", type=str, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--log-every-n-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)

    # reconstruction artifacts
    ap.add_argument("--save-recon", action="store_true", help="Save reconstruction artifacts after training")
    ap.add_argument("--recon-n", type=int, default=64, help="Number of samples to include in recon artifacts")
    ap.add_argument("--recon-splits", type=str, default="val", help="Comma-separated list: train,val,target")
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
    # resolve training knobs (+ overrides)
    # -------------------------
    batch_size = int(args.batch_size if args.batch_size is not None else train_hp.get("batch_size", 256))
    max_epochs = int(args.max_epochs if args.max_epochs is not None else train_hp.get("max_epochs", 50))
    log_every_n_steps = int(
        args.log_every_n_steps if args.log_every_n_steps is not None else train_hp.get("log_every_n_steps", 50)
    )
    num_workers = int(args.num_workers if args.num_workers is not None else train_hp.get("num_workers", 0))

    accelerator = str(args.accelerator if args.accelerator is not None else train_hp.get("accelerator", "auto"))
    devices: Any = args.devices if args.devices is not None else train_hp.get("devices", "auto")
    strategy = str(args.strategy if args.strategy is not None else train_hp.get("strategy", "auto"))
    precision = str(args.precision if args.precision is not None else train_hp.get("precision", "32-true"))
    gradient_clip_val = _none_if_null(train_hp.get("gradient_clip_val", None))
    if gradient_clip_val is not None:
        gradient_clip_val = float(gradient_clip_val)

    devices = _maybe_int(devices)

    # -------------------------
    # build cfg for LitVAE
    # IMPORTANT: LitVAE reads:
    #   - cfg.lr / cfg.beta / cfg.weight_decay (flat)
    #   - cfg.masking.* (nested)
    #   - cfg.seed (for deterministic masking)
    # -------------------------
    # Pull masking knobs from YAML (and ensure null -> None)
    mask_frac = _none_if_null(mask_hp.get("mask_frac", None))
    block_len = _none_if_null(mask_hp.get("block_len", None))

    # If user wrote block_len: null, YAML gives None, which is correct.
    # But if block_len was accidentally "0" or 0, treat it as None (otherwise it disables masking logic)
    if block_len == 0:
        block_len = None

    # Handle padding - can be None for auto-calculation in FullyConvVAE1D
    padding_val = model_hp.get("padding", 4)
    if padding_val is not None:
        padding_val = int(padding_val)

    cfg = SimpleNamespace(
        # model (flat)
        input_len=input_len,
        model_type=str(model_hp.get("model_type", "conv")),  # "conv" or "fully_conv"
        latent_dim=int(model_hp.get("latent_dim", 32)),
        hidden_channels=as_tuple_int(model_hp.get("hidden_channels", [32, 64, 128]), "hidden_channels"),
        kernel_size=int(model_hp.get("kernel_size", 9)),
        stride=int(model_hp.get("stride", 2)),
        padding=padding_val,
        use_batchnorm=bool(model_hp.get("use_batchnorm", False)),
        # training (flat: LitVAE uses getattr(cfg, "lr"/"beta"/"weight_decay"))
        lr=float(train_hp.get("lr", 1e-3)),
        beta=float(train_hp.get("beta", 0.01)),
        weight_decay=float(train_hp.get("weight_decay", 0.0)),
        # seed for deterministic masking
        seed=seed,
        # masking (nested: LitVAE._mask_cfg reads cfg.masking.*)
        masking=SimpleNamespace(
            enabled=bool(mask_hp.get("enabled", False)),
            alpha_masked=float(mask_hp.get("alpha_masked", 1.0)),
            n_blocks=int(mask_hp.get("n_blocks", 1)),
            allow_overlap=bool(mask_hp.get("allow_overlap", True)),
            mask_frac=mask_frac,
            block_len=block_len,
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
    # module
    # -------------------------
    lit = LitVAE(cfg)

    # HARD FAIL if masking is enabled but no mask_frac/block_len made it through
    mcfg = lit._mask_cfg()
    if mcfg["enabled"] and (mcfg["mask_frac"] is None and mcfg["block_len"] is None):
        raise RuntimeError(
            "Masking enabled but both mask_frac and block_len are None. "
            "Your YAML masking section is not being propagated correctly."
        )

    # -------------------------
    # write resolved config for debugging/plotting
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
            "model_type": cfg.model_type,
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
        "_lit_mask_cfg": mcfg,  # what LitVAE actually saw
    }
    (outdir / "hparams.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    # -------------------------
    # callbacks + loggers
    # -------------------------
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val/loss",  # MUST match lit_model.py epoch metric key
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

    logs_root = outdir / "logs"
    metrics = list(logs_root.glob("version_*/metrics.csv"))
    if not metrics:
        raise RuntimeError(f"No metrics.csv found under {logs_root}/version_*/metrics.csv")
    (outdir / "logs_ok.txt").write_text(str(metrics[0]) + "\n")

    print("Best checkpoint:", best_out)
    print("TensorBoard logdir:", outdir / "tb")

    # -------------------------
    # optional reconstruction artifacts
    # -------------------------
    if args.save_recon:
        splits = [s.strip() for s in args.recon_splits.split(",")]
        split_data = {"train": X_train, "val": X_val, "target": X_target}
        for split in splits:
            X_split = split_data.get(split)
            if X_split is None:
                print(f"Skipping split '{split}' (no data)")
                continue
            dump_recon_artifacts(
                lit=lit,
                X=X_split,
                outdir=outdir,
                split=split,
                n=args.recon_n,
                mask_cfg=mcfg,
                seed=seed,
            )


if __name__ == "__main__":
    main()