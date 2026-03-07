#!/usr/bin/env python3
"""
src/vae/train_vae.py

Minimal Lightning trainer for genotype-classification VAE.

Expected model behavior:
- input: masked genotype array (N, L), values in {0,1,2} plus mask token (e.g. -1) after masking
- output: logits of shape (B, 3, L)

It writes:
- outdir/hparams.resolved.yaml
- outdir/train_summary.json
- outdir/checkpoints/best.ckpt
- outdir/logs/version_*/metrics.csv
- outdir/tb/version_*/events...
- optionally recon/{split}_recon.npz with predicted genotype classes
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

from src.vae.diagnostics_callback import ReconDiagnosticsCallback
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
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return x


def compute_class_weights(X: np.ndarray) -> torch.Tensor:
    vals = X.astype(np.int64).ravel()
    counts = np.bincount(vals, minlength=3).astype(np.float32)
    counts = np.clip(counts, 1.0, None)

    total = counts.sum()
    weights = total / (3.0 * counts)

    print("Class counts:", counts.tolist())
    print("Class freqs:", (counts / total).tolist())
    print("Class weights:", weights.tolist())

    return torch.tensor(weights, dtype=torch.float32)


# ----------------------------
# prediction-saving helpers
# ----------------------------
@torch.no_grad()
def _model_predict_classes(
    lit: LitVAE,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run LitVAE's underlying model and return:
      - pred_classes: (B, L) integer predicted genotype classes
      - probs:        (B, 3, L) probabilities
    """
    lit.eval()
    out = lit.model(x)

    if isinstance(out, torch.Tensor):
        logits = out
    elif isinstance(out, (tuple, list)):
        logits = out[0]
    elif isinstance(out, dict):
        logits = out.get("logits", out.get("recon", out.get("x_recon")))
    else:
        raise RuntimeError(f"Unexpected model output type: {type(out)}")

    if logits.ndim != 3 or logits.shape[1] != 3:
        raise RuntimeError(f"Expected logits with shape (B,3,L), got {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=1)
    pred_classes = torch.argmax(logits, dim=1)

    return pred_classes, probs


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
    Save prediction artifacts for analysis.

    Saves recon/{split}_recon.npz with:
      - x_true:      ground truth genotype classes (n_samples, L)
      - x_masked:    masked input fed to model (n_samples, L)
      - mask:        boolean mask where True = masked position (n_samples, L)
      - pred:        predicted genotype classes (n_samples, L)
      - prob_0:      predicted P(genotype=0), shape (n_samples, L)
      - prob_1:      predicted P(genotype=1), shape (n_samples, L)
      - prob_2:      predicted P(genotype=2), shape (n_samples, L)
    """
    out_recon_dir = outdir / "recon"
    out_recon_dir.mkdir(parents=True, exist_ok=True)

    if X.shape[0] == 0:
        return

    n = min(int(n_samples), X.shape[0])
    x_true = X[:n].astype(np.float32, copy=False)
    device = lit.device

    x_true_t = torch.from_numpy(x_true).to(device=device, dtype=torch.float32)

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

    pred_t, probs_t = _model_predict_classes(lit, x_masked_t)

    x_masked = x_masked_t.detach().cpu().numpy().astype(np.float32)
    mask = mask_t.detach().cpu().numpy().astype(np.bool_)
    pred = pred_t.detach().cpu().numpy().astype(np.int64)
    probs = probs_t.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(
        out_recon_dir / f"{split_name}_recon.npz",
        x_true=x_true.astype(np.int64),
        x_masked=x_masked,
        mask=mask,
        pred=pred,
        prob_0=probs[:, 0, :],
        prob_1=probs[:, 1, :],
        prob_2=probs[:, 2, :],
    )
    print(f"Saved {split_name} predictions to {out_recon_dir / f'{split_name}_recon.npz'}")


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

    ap.add_argument("--no-progress-bar", action="store_true")
    ap.add_argument("--accelerator", type=str, default=None)
    ap.add_argument("--devices", type=str, default=None)
    ap.add_argument("--strategy", type=str, default=None)
    ap.add_argument("--precision", type=str, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--log-every-n-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)

    ap.add_argument("--save-recon", action="store_true", help="Save prediction artifacts after training")
    ap.add_argument("--recon-n", type=int, default=100, help="Number of samples to save for prediction analysis")
    ap.add_argument("--recon-splits", type=str, default="val,target", help="Comma-separated splits to save: train,val,target")

    ap.add_argument("--overfit-one-batch", action="store_true", help="Train on a single repeated batch for debugging")
    ap.add_argument("--overfit-n", type=int, default=None, help="Number of examples to keep in overfit mode; defaults to batch_size")

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

    valid_vals = {0, 1, 2}
    train_unique = set(np.unique(X_train).tolist())
    val_unique = set(np.unique(X_val).tolist())
    if not train_unique.issubset(valid_vals):
        raise ValueError(f"Train array must contain only genotype classes 0/1/2. Found: {sorted(train_unique)}")
    if not val_unique.issubset(valid_vals):
        raise ValueError(f"Val array must contain only genotype classes 0/1/2. Found: {sorted(val_unique)}")

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
        target_unique = set(np.unique(X_target).tolist())
        if not target_unique.issubset(valid_vals):
            raise ValueError(f"Target array must contain only genotype classes 0/1/2. Found: {sorted(target_unique)}")

    # -------------------------
    # resolve training knobs
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

    gradient_clip_val = train_hp.get("gradient_clip_val", None)
    if gradient_clip_val is not None:
        gradient_clip_val = float(gradient_clip_val)

    devices = _maybe_int(devices)

    # -------------------------
    # optional overfit-one-batch mode
    # -------------------------
    if args.overfit_one_batch:
        n_overfit = int(args.overfit_n) if args.overfit_n is not None else batch_size
        if n_overfit <= 0:
            raise ValueError("--overfit-n must be positive")
        if n_overfit > X_train.shape[0]:
            n_overfit = X_train.shape[0]

        X_train = X_train[:n_overfit].copy()
        X_val = X_train.copy()
        if X_target is not None:
            X_target = X_train.copy()

        batch_size = X_train.shape[0]

        print(f"[OVERFIT MODE] Using one repeated batch of size {batch_size}")
        print(f"[OVERFIT MODE] Train shape: {X_train.shape}")
        print(f"[OVERFIT MODE] Val shape:   {X_val.shape}")
        if X_target is not None:
            print(f"[OVERFIT MODE] Target shape:{X_target.shape}")

    # -------------------------
    # build cfg for LitVAE
    # -------------------------
    padding_val = model_hp.get("padding", 4)
    if padding_val is not None:
        padding_val = int(padding_val)

    cfg = SimpleNamespace(
        input_len=input_len,
        model_type=str(model_hp.get("model_type", "conv")),
        latent_dim=int(model_hp.get("latent_dim", 32)),
        hidden_channels=as_tuple_int(model_hp.get("hidden_channels", [32, 64, 128]), "hidden_channels"),
        kernel_size=int(model_hp.get("kernel_size", 9)),
        stride=int(model_hp.get("stride", 2)),
        padding=padding_val,
        use_batchnorm=bool(model_hp.get("use_batchnorm", False)),
        seed=seed,
        training=SimpleNamespace(
            lr=float(train_hp.get("lr", 1e-3)),
            beta=float(train_hp.get("beta", 0.01)),
            weight_decay=float(train_hp.get("weight_decay", 0.0)),
        ),
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
    train_loader = make_loader(
        X_train,
        batch_size,
        shuffle=(not args.overfit_one_batch),
        num_workers=num_workers,
    )
    val_loader = make_loader(X_val, batch_size, shuffle=False, num_workers=num_workers)

    val_dataloaders = [val_loader]
    if X_target is not None:
        target_loader = make_loader(X_target, batch_size, shuffle=False, num_workers=num_workers)
        val_dataloaders.append(target_loader)

    # -------------------------
    # write resolved config
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
            "overfit_one_batch": bool(args.overfit_one_batch),
            "overfit_n": (int(args.overfit_n) if args.overfit_n is not None else None),
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

    class_weights = compute_class_weights(X_train)
    lit.set_class_weights(class_weights)
    print("Using class weights:", class_weights.tolist())

    # -------------------------
    # callbacks + loggers
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

    diag_cb = ReconDiagnosticsCallback(
        X_val_diag=X_val[: min(64, len(X_val))],
        outdir=outdir / "diagnostics_live",
        mask_cfg=mask_cfg_dict,
        seed=seed + 999999,
    )

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
        callbacks=[ckpt_cb, lr_cb, diag_cb],
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
    # save prediction artifacts (optional)
    # -------------------------
    if args.save_recon:
        lit_best = LitVAE(cfg)
        lit_best.set_class_weights(class_weights)
        obj = torch.load(best_out, map_location="cpu")
        state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        lit_best.load_state_dict(state, strict=True)
        lit_best.eval()

        if torch.cuda.is_available() and accelerator != "cpu":
            lit_best = lit_best.to("cuda")

        splits_to_save = [s.strip().lower() for s in args.recon_splits.split(",") if s.strip()]

        for split in splits_to_save:
            if split == "train":
                X = X_train
            elif split == "val":
                X = X_val
            elif split == "target":
                if X_target is None:
                    print("Skipping target split (no target data provided)")
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