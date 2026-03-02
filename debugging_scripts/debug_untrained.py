#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

from src.vae.lit_model import LitVAE
from src.vae.model import VAEConfig


# ----------------------------
# IO helpers
# ----------------------------
def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _as_tuple_int(x: Any, name: str) -> Tuple[int, ...]:
    if isinstance(x, tuple):
        return tuple(int(v) for v in x)
    if isinstance(x, list):
        return tuple(int(v) for v in x)
    raise ValueError(f"{name} must be list/tuple of ints, got {type(x)}")


def build_cfg_from_yaml(yaml_path: Path, input_len: int) -> VAEConfig:
    """
    Supports either:
      - hparams.resolved.yaml (your resolved format)
      - model_hyperparams/vae.yaml (seed/model/training sections)
    """
    d = read_yaml(yaml_path)

    # resolved format?
    if "model" in d and "training" in d and "data" in d:
        model = d.get("model", {}) or {}
        training = d.get("training", {}) or {}
        return VAEConfig(
            input_len=input_len,
            latent_dim=int(model.get("latent_dim", 32)),
            hidden_channels=tuple(int(v) for v in model.get("hidden_channels", [32, 64, 128])),
            kernel_size=int(model.get("kernel_size", 9)),
            stride=int(model.get("stride", 2)),
            padding=int(model.get("padding", 4)),
            use_batchnorm=bool(model.get("use_batchnorm", False)),
            lr=float(training.get("lr", 1e-3)),
            beta=float(training.get("beta", 0.01)),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )

    # non-resolved format (your main YAML)
    model = d.get("model", {}) or {}
    training = d.get("training", {}) or {}
    return VAEConfig(
        input_len=input_len,
        latent_dim=int(model.get("latent_dim", 32)),
        hidden_channels=_as_tuple_int(model.get("hidden_channels", [32, 64, 128]), "hidden_channels"),
        kernel_size=int(model.get("kernel_size", 9)),
        stride=int(model.get("stride", 2)),
        padding=int(model.get("padding", 4)),
        use_batchnorm=bool(model.get("use_batchnorm", False)),
        lr=float(training.get("lr", 1e-3)),
        beta=float(training.get("beta", 0.01)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


# ----------------------------
# Metrics + baselines
# ----------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def maf_baseline_from_train(X_train: np.ndarray) -> np.ndarray:
    """
    Baseline predicts each SNP as 2 * p where p is allele frequency in CEU_train.
    Assumes X in {0,1,2} (or float versions thereof).
    """
    # p = mean(X/2)
    p = np.mean(X_train / 2.0, axis=0)  # shape (L,)
    pred = 2.0 * p  # shape (L,)
    return pred.astype(np.float32)


@torch.no_grad()
def recon_with_untrained_model(lit: LitVAE, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """
    Runs forward pass through an UNTRAINED model to get reconstructions.
    """
    lit.eval().to(device)
    Xt = torch.from_numpy(X.astype(np.float32))

    outs = []
    for i in range(0, Xt.shape[0], batch_size):
        xb = Xt[i : i + batch_size].to(device)
        # avoid autocast surprises for debugging
        with torch.cuda.amp.autocast(enabled=False):
            out = lit.model(xb)  # ConvVAE1D returns (recon, mu, logvar)
        recon = out[0] if isinstance(out, (tuple, list)) else out
        outs.append(recon.detach().cpu())

    return torch.cat(outs, dim=0).numpy()


def write_table(
    outpath: Path,
    *,
    ceu_train: np.ndarray,
    ceu_val: np.ndarray,
    yri_target: np.ndarray,
    recon_train: np.ndarray,
    recon_val: np.ndarray,
    recon_target: np.ndarray,
    maf_pred: np.ndarray,
) -> None:
    """
    Writes:
    split    metric  model   maf_baseline
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    def add(split: str, X: np.ndarray, R: np.ndarray) -> None:
        # model
        rows.append((split, "mse", mse(R, X), mse(np.broadcast_to(maf_pred, X.shape), X)))
        rows.append((split, "mae", mae(R, X), mae(np.broadcast_to(maf_pred, X.shape), X)))

    add("CEU_train", ceu_train, recon_train)
    add("CEU_val", ceu_val, recon_val)
    add("YRI_target", yri_target, recon_target)

    lines = ["split\tmetric\tmodel\tmaf_baseline"]
    for s, m, v_model, v_maf in rows:
        lines.append(f"{s}\t{m}\t{v_model:.6g}\t{v_maf:.6g}")

    outpath.write_text("\n".join(lines) + "\n")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Debug: recon error of an UNTRAINED VAE + MAF baseline")
    ap.add_argument("--ceu-train", type=Path, required=True)
    ap.add_argument("--ceu-val", type=Path, required=True)
    ap.add_argument("--yri-target", type=Path, required=True)

    ap.add_argument("--hparams", type=Path, required=True, help="vae.yaml or hparams.resolved.yaml")
    ap.add_argument("--out", type=Path, required=True, help="output TSV")

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    X_train = np.load(args.ceu_train).astype(np.float32)
    X_val = np.load(args.ceu_val).astype(np.float32)
    X_targ = np.load(args.yri_target).astype(np.float32)

    if X_train.ndim != 2 or X_val.ndim != 2 or X_targ.ndim != 2:
        raise ValueError(f"Expected 2D arrays. Got {X_train.shape}, {X_val.shape}, {X_targ.shape}")
    if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_targ.shape[1]:
        raise ValueError("All splits must have same number of SNPs (same input_len).")

    input_len = int(X_train.shape[1])

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    )

    # Build UNTRAINED model
    cfg = build_cfg_from_yaml(args.hparams, input_len=input_len)
    lit = LitVAE(cfg)

    # Reconstructions (untrained)
    R_train = recon_with_untrained_model(lit, X_train, device=device, batch_size=args.batch_size)
    R_val = recon_with_untrained_model(lit, X_val, device=device, batch_size=args.batch_size)
    R_targ = recon_with_untrained_model(lit, X_targ, device=device, batch_size=args.batch_size)

    # MAF baseline from CEU_train
    maf_pred = maf_baseline_from_train(X_train)

    # Write table
    write_table(
        args.out,
        ceu_train=X_train,
        ceu_val=X_val,
        yri_target=X_targ,
        recon_train=R_train,
        recon_val=R_val,
        recon_target=R_targ,
        maf_pred=maf_pred,
    )

    print("Wrote:", args.out)
    print("Device:", device)


if __name__ == "__main__":
    main()