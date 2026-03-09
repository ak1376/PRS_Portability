#!/usr/bin/env python3
# /sietch_colab/akapoor/PRS_Portability/src/vae/train_vae_wrapper.py
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
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

from src.masking import make_mask_and_apply
from src.vae.lit_model import LitVAE


# =========================================================
# IO helpers
# =========================================================

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
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"none", "null", ""}:
        return None
    return x


def _atomic_savez_compressed(out_path: Path, **arrays: Any) -> None:
    """
    Write .npz atomically:
      1) write to temp file in same directory
      2) verify readable
      3) atomically replace destination
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=out_path.stem + ".tmp.",
        suffix=".npz",
        dir=str(out_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        np.savez_compressed(tmp_path, **arrays)

        # Verify archive integrity before promoting it
        with np.load(tmp_path) as chk:
            for k in chk.files:
                _ = chk[k]

        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# =========================================================
# Data helpers
# =========================================================

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


def _load_2d_array(path: Path, name: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for {name}. Got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {name}")
    return arr


# =========================================================
# Reconstruction artifact dumping
# =========================================================

@torch.no_grad()
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
    Save reconstruction artifacts for up to `n` samples.

    Output:
      outdir/recon/{split}_recon.npz

    Keys saved:
      - x_true               : (N, L) true genotype classes
      - x_masked             : (N, L) masked/corrupted input values
      - mask                 : (N, L) boolean mask, True = masked
      - recon                : (N, 3, L) raw logits (kept for backward compatibility)
      - pred                 : (N, L) argmax predicted genotype class
      - prob_0               : (N, L) P(genotype = 0)
      - prob_1               : (N, L) P(genotype = 1)
      - prob_2               : (N, L) P(genotype = 2)
      - used_n_blocks        : scalar
      - used_block_len       : scalar
      - target_mask_frac     : scalar
      - realized_mask_frac   : scalar
    """
    if X.shape[0] == 0:
        return

    lit.eval()
    device = next(lit.parameters()).device

    n_use = min(int(n), int(X.shape[0]))
    X_sub = X[:n_use]
    x_true = torch.from_numpy(X_sub).float().to(device)

    x_in, mask, used_n_blocks, used_block_len, target_mask_frac, realized_mask_frac = make_mask_and_apply(
        x_true,
        enabled=bool(mask_cfg.get("enabled", False)),
        constraint_mode=str(mask_cfg.get("constraint_mode", "frac_and_blocks")),
        n_blocks=mask_cfg.get("n_blocks", None),
        block_len=mask_cfg.get("block_len", None),
        mask_frac=mask_cfg.get("mask_frac", None),
        allow_overlap=bool(mask_cfg.get("allow_overlap", True)),
        seed=int(seed),
        fill=str(mask_cfg.get("fill", "gaussian")),
        gaussian_std=float(mask_cfg.get("gaussian_std", 0.1)),
        constant_value=float(mask_cfg.get("constant_value", 0.0)),
        mask_token_value=float(mask_cfg.get("mask_token_value", -1.0)),
        consistency_tolerance=int(mask_cfg.get("consistency_tolerance", 1)),
    )

    model_in = lit._make_model_input(x_in, mask)
    out = lit(model_in)

    if isinstance(out, torch.Tensor):
        logits = out
    elif isinstance(out, (tuple, list)):
        logits = out[0]
    elif isinstance(out, dict):
        if "logits" in out:
            logits = out["logits"]
        elif "recon" in out:
            logits = out["recon"]
        else:
            raise RuntimeError(f"Unexpected dict output keys from model: {list(out.keys())}")
    else:
        raise RuntimeError(f"Unexpected output type from LitVAE forward: {type(out)}")

    if logits.ndim != 3 or logits.shape[1] != 3:
        raise RuntimeError(f"Expected logits with shape (B, 3, L), got {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)

    recon_dir = outdir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)
    out_path = recon_dir / f"{split}_recon.npz"

    _atomic_savez_compressed(
        out_path,
        x_true=x_true.detach().cpu().numpy().astype(np.int64),
        x_masked=x_in.detach().cpu().numpy().astype(np.float32),
        mask=mask.detach().cpu().numpy().astype(np.bool_),

        # backward-compatible
        recon=logits.detach().cpu().numpy().astype(np.float32),

        # preferred newer format
        pred=pred.detach().cpu().numpy().astype(np.int64),
        prob_0=probs[:, 0, :].detach().cpu().numpy().astype(np.float32),
        prob_1=probs[:, 1, :].detach().cpu().numpy().astype(np.float32),
        prob_2=probs[:, 2, :].detach().cpu().numpy().astype(np.float32),

        # masking metadata
        used_n_blocks=np.array(used_n_blocks, dtype=np.int64),
        used_block_len=np.array(used_block_len, dtype=np.int64),
        target_mask_frac=np.array(target_mask_frac, dtype=np.float32),
        realized_mask_frac=np.array(realized_mask_frac, dtype=np.float32),
    )

    print(f"Saved reconstruction artifacts to {out_path}")


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Lightweight VAE training wrapper (Snakemake-friendly).")
    ap.add_argument("--train", type=Path, required=True, help=".npy (N,L) train array")
    ap.add_argument("--val", type=Path, required=True, help=".npy (N,L) val array")
    ap.add_argument("--target", type=Path, default=None, help="Optional .npy (N,L) target array (2nd val dataloader)")
    ap.add_argument("--no-target-val", action="store_true",
                    help="Do not include target dataset as a validation dataloader")
    ap.add_argument("--hparams", type=Path, required=True, help="YAML with model/training/masking/seed sections")
    ap.add_argument("--outdir", type=Path, required=True)

    # Optional overrides
    ap.add_argument("--no-progress-bar", action="store_true")
    ap.add_argument("--accelerator", type=str, default=None)
    ap.add_argument("--devices", type=str, default=None)
    ap.add_argument("--strategy", type=str, default=None)
    ap.add_argument("--precision", type=str, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--log-every-n-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)

    # Overfit debugging (CLI override of YAML)
    ap.add_argument("--overfit-one-batch", action="store_true",
                    help="Train on one repeated batch only")
    ap.add_argument("--overfit-n", type=int, default=None,
                    help="Number of examples to keep in overfit mode; defaults to batch_size")

    # Optional reconstruction artifacts
    ap.add_argument("--save-recon", action="store_true", help="Save reconstruction artifacts after training")
    ap.add_argument("--recon-n", type=int, default=64, help="Number of samples to include in recon artifacts")
    ap.add_argument("--recon-splits", type=str, default="val", help="Comma-separated list: train,val,target")

    return ap.parse_args()


# =========================================================
# Main
# =========================================================

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
    # Load arrays
    # -------------------------
    X_train = _load_2d_array(args.train, "train")
    X_val = _load_2d_array(args.val, "val")

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(f"Train/val must have same #features. Got {X_train.shape[1]} vs {X_val.shape[1]}")

    input_len = int(X_train.shape[1])

    X_target: Optional[np.ndarray] = None
    if args.target is not None:
        X_target = _load_2d_array(args.target, "target")
        if X_target.shape[1] != input_len:
            raise ValueError(
                f"Target must have same #features as train/val. Got {X_target.shape[1]} vs {input_len}"
            )

    # -------------------------
    # Resolve training knobs
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

    # YAML defaults
    yaml_overfit_one_batch = bool(train_hp.get("overfit_one_batch", False))
    yaml_overfit_n = train_hp.get("overfit_n", None)
    if yaml_overfit_n is not None:
        yaml_overfit_n = int(yaml_overfit_n)

    # CLI overrides YAML if provided
    overfit_one_batch = bool(args.overfit_one_batch or yaml_overfit_one_batch)
    overfit_n = int(args.overfit_n) if args.overfit_n is not None else yaml_overfit_n

    devices = _maybe_int(devices)

    # -------------------------
    # Optional overfit-one-batch mode
    # -------------------------
    if overfit_one_batch:
        n_overfit = int(overfit_n) if overfit_n is not None else batch_size
        if n_overfit <= 0:
            raise ValueError("overfit_n must be positive")
        n_overfit = min(n_overfit, int(X_train.shape[0]))

        X_train = X_train[:n_overfit].copy()
        X_val = X_train.copy()
        if X_target is not None:
            X_target = X_train.copy()

        batch_size = int(X_train.shape[0])

        print(f"[OVERFIT MODE] Using one repeated batch of size {batch_size}")
        print(f"[OVERFIT MODE] train shape:  {X_train.shape}")
        print(f"[OVERFIT MODE] val shape:    {X_val.shape}")
        if X_target is not None:
            print(f"[OVERFIT MODE] target shape: {X_target.shape}")

    # -------------------------
    # Resolve masking knobs
    # -------------------------
    mask_frac = _none_if_null(mask_hp.get("mask_frac", None))
    block_len = _none_if_null(mask_hp.get("block_len", None))
    if block_len == 0:
        block_len = None

    padding_val = model_hp.get("padding", 4)
    if padding_val is not None:
        padding_val = int(padding_val)

    constraint_mode = str(mask_hp.get("constraint_mode", "frac_and_blocks"))

    n_blocks = _none_if_null(mask_hp.get("n_blocks", None))
    if n_blocks is not None:
        n_blocks = int(n_blocks)

    mask_frac = _none_if_null(mask_hp.get("mask_frac", None))
    if mask_frac is not None:
        mask_frac = float(mask_frac)

    block_len = _none_if_null(mask_hp.get("block_len", None))
    if block_len is not None:
        block_len = int(block_len)
    if block_len == 0:
        block_len = None

    # -------------------------
    # Build cfg for LitVAE
    # -------------------------
    cfg = SimpleNamespace(
        input_len=input_len,
        model_type=str(model_hp.get("model_type", "conv")),
        latent_dim=int(model_hp.get("latent_dim", 32)),
        hidden_channels=as_tuple_int(model_hp.get("hidden_channels", [32, 64, 128]), "hidden_channels"),
        kernel_size=int(model_hp.get("kernel_size", 9)),
        stride=int(model_hp.get("stride", 2)),
        padding=padding_val,
        use_batchnorm=bool(model_hp.get("use_batchnorm", False)),
        lr=float(train_hp.get("lr", 1e-3)),
        beta=float(train_hp.get("beta", 0.01)),
        weight_decay=float(train_hp.get("weight_decay", 0.0)),
        seed=seed,
        masking=SimpleNamespace(
            enabled=bool(mask_hp.get("enabled", False)),
            alpha_masked=float(mask_hp.get("alpha_masked", 1.0)),
            constraint_mode=str(mask_hp.get("constraint_mode", "frac_and_blocks")),
            n_blocks=_none_if_null(mask_hp.get("n_blocks", None)),
            allow_overlap=bool(mask_hp.get("allow_overlap", True)),
            mask_frac=mask_frac,
            block_len=block_len,
            fill=str(mask_hp.get("fill", mask_hp.get("fill_value", "gaussian"))),
            gaussian_std=float(mask_hp.get("gaussian_std", 0.1)),
            constant_value=float(mask_hp.get("constant_value", 0.0)),
            mask_token_value=float(mask_hp.get("mask_token_value", -1.0)),
            consistency_tolerance=int(mask_hp.get("consistency_tolerance", 1)),
            use_mask_channel=bool(mask_hp.get("use_mask_channel", False)),
        ),
    )

    # -------------------------
    # Dataloaders
    # -------------------------
    train_loader = make_loader(
        X_train,
        batch_size,
        shuffle=(not overfit_one_batch),
        num_workers=num_workers,
    )
    val_loader = make_loader(X_val, batch_size, shuffle=False, num_workers=num_workers)

    val_dataloaders = [val_loader]
    if X_target is not None and not args.no_target_val:
        target_loader = make_loader(X_target, batch_size, shuffle=False, num_workers=num_workers)
        val_dataloaders.append(target_loader)

    # -------------------------
    # Module
    # -------------------------
    lit = LitVAE(cfg)

    # Validate mask propagation immediately
    mcfg = lit._mask_cfg()
    if mcfg["enabled"]:
        mode = str(mcfg.get("constraint_mode", "frac_and_blocks"))

        if mode == "frac_and_blocks":
            if mcfg["mask_frac"] is None or mcfg["n_blocks"] is None:
                raise RuntimeError(
                    "Masking enabled with constraint_mode='frac_and_blocks' "
                    "but mask_frac or n_blocks is missing."
                )

        elif mode == "frac_and_len":
            if mcfg["mask_frac"] is None or mcfg["block_len"] is None:
                raise RuntimeError(
                    "Masking enabled with constraint_mode='frac_and_len' "
                    "but mask_frac or block_len is missing."
                )

        elif mode == "blocks_and_len":
            if mcfg["n_blocks"] is None or mcfg["block_len"] is None:
                raise RuntimeError(
                    "Masking enabled with constraint_mode='blocks_and_len' "
                    "but n_blocks or block_len is missing."
                )

        else:
            raise RuntimeError(f"Unknown constraint_mode: {mode}")

    # -------------------------
    # Callbacks + loggers
    # -------------------------
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="vae-{epoch:03d}",
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    tb_logger = TensorBoardLogger(save_dir=str(outdir), name="tb")
    csv_logger = CSVLogger(save_dir=str(outdir), name="logs")
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
        gradient_clip_val=gradient_clip_val,
        callbacks=[ckpt_cb, lr_cb],
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=str(outdir),
        enable_progress_bar=(not args.no_progress_bar),
        enable_model_summary=False,
    )

    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_dataloaders)

    # Make sure all ranks finish training before rank 0 touches shared outputs
    trainer.strategy.barrier("post_fit_barrier")

    if not trainer.is_global_zero:
        return

    print(f"[rank0] trainer.global_rank={getattr(trainer, 'global_rank', 'NA')}")
    print(f"[rank0] writing shared outputs under {outdir}")

    # -------------------------
    # Write resolved config (rank 0 only)
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
            "gradient_clip_val": gradient_clip_val,
            "overfit_one_batch": overfit_one_batch,
            "overfit_n": overfit_n,
            "no_target_val": bool(args.no_target_val),
        },
        "masking": {
            "enabled": cfg.masking.enabled,
            "alpha_masked": cfg.masking.alpha_masked,
            "constraint_mode": cfg.masking.constraint_mode,
            "n_blocks": cfg.masking.n_blocks,
            "allow_overlap": cfg.masking.allow_overlap,
            "mask_frac": cfg.masking.mask_frac,
            "block_len": cfg.masking.block_len,
            "fill": cfg.masking.fill,
            "gaussian_std": cfg.masking.gaussian_std,
            "constant_value": cfg.masking.constant_value,
            "mask_token_value": cfg.masking.mask_token_value,
            "consistency_tolerance": cfg.masking.consistency_tolerance,
            "use_mask_channel": cfg.masking.use_mask_channel,
        },
        "_lit_mask_cfg": mcfg,
    }
    (outdir / "hparams.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    # -------------------------
    # Summary + stable best.ckpt
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
    # Optional reconstruction artifacts
    # IMPORTANT: dump from BEST checkpoint, not last in-memory model
    # rank 0 only
    # -------------------------
    if args.save_recon:
        best_lit = LitVAE(cfg)
        ckpt_obj = torch.load(best_out, map_location="cpu")

        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
            state_dict = ckpt_obj["state_dict"]
        else:
            state_dict = ckpt_obj

        best_lit.load_state_dict(state_dict, strict=True)
        best_lit.eval()

        if torch.cuda.is_available() and accelerator != "cpu":
            best_lit = best_lit.to("cuda")

        split_names = [s.strip() for s in args.recon_splits.split(",") if s.strip()]
        split_to_data: dict[str, Optional[np.ndarray]] = {
            "train": X_train,
            "val": X_val,
            "target": X_target,
        }

        for split in split_names:
            X_split = split_to_data.get(split)
            if X_split is None:
                print(f"Skipping split '{split}' (no data)")
                continue

            dump_recon_artifacts(
                lit=best_lit,
                X=X_split,
                outdir=outdir,
                split=split,
                n=args.recon_n,
                mask_cfg=mcfg,
                seed=seed,
            )


if __name__ == "__main__":
    main()