#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from src.masking import make_mask_and_apply


def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _load_2d_array(path: Path, name: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for {name}. Got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {name}")
    return arr


def _none_if_null(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"none", "null", ""}:
        return None
    return x


def _save_pair(
    x_masked: torch.Tensor,
    mask: torch.Tensor,
    masked_path: Path,
    mask_path: Path,
) -> None:
    masked_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(masked_path, x_masked.detach().cpu().numpy().astype(np.float32))
    np.save(mask_path, mask.detach().cpu().numpy().astype(np.bool_))


def _make_one(
    x_np: np.ndarray,
    *,
    mask_cfg: Dict[str, Any],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, float, float]:
    x = torch.from_numpy(x_np).float()

    x_in, mask, used_n_blocks, used_block_len, target_mask_frac, realized_mask_frac = make_mask_and_apply(
        x,
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
    return x_in, mask, used_n_blocks, used_block_len, target_mask_frac, realized_mask_frac


def _resolve_mask_cfg(mask_hp: Dict[str, Any]) -> Dict[str, Any]:
    mask_frac = _none_if_null(mask_hp.get("mask_frac", None))
    if mask_frac is not None:
        mask_frac = float(mask_frac)

    block_len = _none_if_null(mask_hp.get("block_len", None))
    if block_len is not None:
        block_len = int(block_len)
    if block_len == 0:
        block_len = None

    n_blocks = _none_if_null(mask_hp.get("n_blocks", None))
    if n_blocks is not None:
        n_blocks = int(n_blocks)

    mask_cfg = {
        "enabled": bool(mask_hp.get("enabled", False)),
        "alpha_masked": float(mask_hp.get("alpha_masked", 1.0)),
        "constraint_mode": str(mask_hp.get("constraint_mode", "frac_and_blocks")),
        "n_blocks": n_blocks,
        "allow_overlap": bool(mask_hp.get("allow_overlap", True)),
        "mask_frac": mask_frac,
        "block_len": block_len,
        "fill": str(mask_hp.get("fill", mask_hp.get("fill_value", "gaussian"))),
        "gaussian_std": float(mask_hp.get("gaussian_std", 0.1)),
        "constant_value": float(mask_hp.get("constant_value", 0.0)),
        "mask_token_value": float(mask_hp.get("mask_token_value", -1.0)),
        "consistency_tolerance": int(mask_hp.get("consistency_tolerance", 1)),
        "use_mask_channel": bool(mask_hp.get("use_mask_channel", False)),
    }
    return mask_cfg


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Precompute masked inputs for VAE training/validation.")
    ap.add_argument("--train", type=Path, default=None, help="Train genotype .npy (N,L)")
    ap.add_argument("--val", type=Path, default=None, help="Validation genotype .npy (N,L)")
    ap.add_argument("--target", type=Path, default=None, help="Optional target genotype .npy (N,L)")
    ap.add_argument("--hparams", type=Path, required=True, help="Generated VAE YAML")
    ap.add_argument("--outdir", type=Path, required=True, help="Output masked_inputs directory")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval-only", action="store_true", help="Write val/target masks only")
    mode.add_argument("--train-epoch", type=int, default=None, help="Write train mask pair for exactly one epoch")
    mode.add_argument("--write-metadata-only", action="store_true", help="Write mask_metadata.yaml only")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    hp = read_yaml(args.hparams)
    train_hp = hp.get("training", {}) or {}
    mask_hp = hp.get("masking", {}) or {}
    seed = int(hp.get("seed", 0))
    max_epochs = int(train_hp.get("max_epochs", 50))
    mask_cfg = _resolve_mask_cfg(mask_hp)

    print(f"[precompute_vae_masks] outdir={outdir}")
    print(f"[precompute_vae_masks] seed={seed}")
    print(f"[precompute_vae_masks] max_epochs={max_epochs}")
    print(f"[precompute_vae_masks] mask_cfg={mask_cfg}")

    if args.eval_only:
        if args.val is None:
            raise ValueError("--eval-only requires --val")
        X_val = _load_2d_array(args.val, "val")
        print(f"[precompute_vae_masks] val shape={X_val.shape}")

        x_val_masked, val_mask, *_ = _make_one(
            X_val,
            mask_cfg=mask_cfg,
            seed=seed + 1_000_000,
        )
        _save_pair(
            x_val_masked,
            val_mask,
            outdir / "val_masked.npy",
            outdir / "val_mask.npy",
        )
        print("[precompute_vae_masks] wrote val masks")

        if args.target is not None:
            X_target = _load_2d_array(args.target, "target")
            if X_target.shape[1] != X_val.shape[1]:
                raise ValueError(f"Val/target feature mismatch: {X_val.shape[1]} vs {X_target.shape[1]}")
            print(f"[precompute_vae_masks] target shape={X_target.shape}")

            x_target_masked, target_mask, *_ = _make_one(
                X_target,
                mask_cfg=mask_cfg,
                seed=seed + 2_000_000,
            )
            _save_pair(
                x_target_masked,
                target_mask,
                outdir / "target_masked.npy",
                outdir / "target_mask.npy",
            )
            print("[precompute_vae_masks] wrote target masks")

        return

    if args.train_epoch is not None:
        if args.train is None:
            raise ValueError("--train-epoch requires --train")
        epoch = int(args.train_epoch)
        if epoch < 0:
            raise ValueError("--train-epoch must be >= 0")
        if epoch >= max_epochs:
            raise ValueError(f"--train-epoch {epoch} >= max_epochs {max_epochs}")

        X_train = _load_2d_array(args.train, "train")
        print(f"[precompute_vae_masks] train shape={X_train.shape}")
        print(f"[precompute_vae_masks] writing epoch {epoch}")

        epoch_seed = seed + 10_000 * epoch
        x_train_masked, train_mask, *_ = _make_one(
            X_train,
            mask_cfg=mask_cfg,
            seed=epoch_seed,
        )
        _save_pair(
            x_train_masked,
            train_mask,
            outdir / f"train_masked_epoch{epoch}.npy",
            outdir / f"train_mask_epoch{epoch}.npy",
        )
        print(f"[precompute_vae_masks] wrote epoch {epoch}")
        return

    if args.write_metadata_only:
        if args.train is None or args.val is None:
            raise ValueError("--write-metadata-only requires --train and --val")

        X_train = _load_2d_array(args.train, "train")
        X_val = _load_2d_array(args.val, "val")
        X_target: Optional[np.ndarray] = None
        if args.target is not None:
            X_target = _load_2d_array(args.target, "target")

        if X_val.shape[1] != X_train.shape[1]:
            raise ValueError(f"Train/val feature mismatch: {X_train.shape[1]} vs {X_val.shape[1]}")
        if X_target is not None and X_target.shape[1] != X_train.shape[1]:
            raise ValueError(f"Train/target feature mismatch: {X_train.shape[1]} vs {X_target.shape[1]}")

        metadata = {
            "seed": seed,
            "max_epochs": max_epochs,
            "train_shape": list(X_train.shape),
            "val_shape": list(X_val.shape),
            "target_shape": (list(X_target.shape) if X_target is not None else None),
            "mask_cfg": mask_cfg,
        }
        (outdir / "mask_metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
        print("[precompute_vae_masks] wrote mask_metadata.yaml")
        return

    raise RuntimeError("No mode selected")
    

if __name__ == "__main__":
    main()