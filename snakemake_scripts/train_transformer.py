#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# --- ensure "import src.*" works when called by Snakemake ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml  # NEW

from src.transformer.data import HapPairDataset, collate_happairbatch
from src.transformer.model import HapMaskTransformer
from src.transformer.train import train_epoch


def write_losses_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        raise ValueError("No loss rows to write.")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_losses(path: Path, epochs: list[int], losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _maybe_load_cfg(cfg_path: str | None) -> dict:
    if cfg_path is None:
        return {}
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"--config not found: {cfg_path}")
    return yaml.safe_load(p.read_text()) or {}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--hap1", type=str, required=True, help="Path to hap1 .npy (N,L)")
    ap.add_argument("--hap2", type=str, required=True, help="Path to hap2 .npy (N,L)")

    ap.add_argument("--out_model", type=str, required=True)
    ap.add_argument("--out_losses", type=str, required=True)
    ap.add_argument("--out_plot", type=str, required=True)

    # YAML config (optional but recommended)
    ap.add_argument("--config", type=str, default=None, help="transformer_model_config.yaml")

    # Model hparams (allow YAML defaults by using None here)
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--vocab_size", type=int, default=None)

    # Training hparams (allow YAML defaults)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)

    ap.add_argument("--p_mask_site", type=float, default=None)
    ap.add_argument("--mask_both_prob", type=float, default=None)
    ap.add_argument("--mask_id", type=int, default=None)

    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)

    # Windowing (allow YAML defaults)
    ap.add_argument("--window_len", type=int, default=None)
    ap.add_argument("--window_mode", type=str, default=None, choices=["random", "first", "middle", "fixed"])
    ap.add_argument("--fixed_start", type=int, default=None)

    # Mixed precision toggle
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")

    args = ap.parse_args()

    cfg = _maybe_load_cfg(args.config)

    # ---- pull defaults from YAML if CLI didn't provide ----
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg  = cfg.get("data", {})

    def pick(val, fallback):
        return fallback if val is None else val

    args.d_model   = pick(args.d_model,   model_cfg.get("d_model", 128))
    args.n_heads   = pick(args.n_heads,   model_cfg.get("n_heads", 8))
    args.n_layers  = pick(args.n_layers,  model_cfg.get("n_layers", 6))
    args.dropout   = pick(args.dropout,   model_cfg.get("dropout", 0.1))
    args.vocab_size= pick(args.vocab_size,model_cfg.get("vocab_size", 3))

    args.epochs    = pick(args.epochs,    train_cfg.get("epochs", 10))
    args.batch_size= pick(args.batch_size,train_cfg.get("batch_size", 32))
    args.lr        = pick(args.lr,        train_cfg.get("lr", 3e-4))

    args.p_mask_site    = pick(args.p_mask_site,    train_cfg.get("p_mask_site", 0.15))
    args.mask_both_prob = pick(args.mask_both_prob, train_cfg.get("mask_both_prob", 1.0))
    args.mask_id        = pick(args.mask_id,        train_cfg.get("mask_id", 2))

    args.num_workers = pick(args.num_workers, train_cfg.get("num_workers", 2))

    args.window_len  = pick(args.window_len,  data_cfg.get("window_len", 1024))
    args.window_mode = pick(args.window_mode, data_cfg.get("window_mode", "random"))
    args.fixed_start = pick(args.fixed_start, data_cfg.get("fixed_start", 0))

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load haplotypes from .npy -> torch.long
    hap1 = torch.from_numpy(np.load(args.hap1)).long()
    hap2 = torch.from_numpy(np.load(args.hap2)).long()

    N, L_total = hap1.shape
    print(f"[train_transformer] hap1 shape: {hap1.shape}")
    print(f"[train_transformer] window_len={args.window_len} (sequence length used by model)")

    # Dataset RNG (so "random windows" are reproducible)
    g = torch.Generator()
    g.manual_seed(args.seed)

    ds = HapPairDataset(
        hap1, hap2,
        pad_id=None,
        window_len=int(args.window_len) if args.window_len is not None else None,
        window_mode=str(args.window_mode),
        fixed_start=int(args.fixed_start),
        rng=g,
    )

    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_happairbatch,  # REQUIRED for HapPairBatch
        persistent_workers=(int(args.num_workers) > 0),
    )

    model = HapMaskTransformer(
        vocab_size=int(args.vocab_size),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    # Train + log
    loss_rows: list[dict] = []
    epoch_list: list[int] = []
    loss_list: list[float] = []

    for ep in range(1, int(args.epochs) + 1):
        if use_amp:
            # Your train_epoch currently does backward() internally.
            # Easiest path: keep train_epoch FP32, but amp still helps inside model forward
            # if your model uses autocast internally. If not, you can ignore --amp.
            pass

        loss = train_epoch(
            model=model,
            loader=dl,
            optimizer=opt,
            device=device,
            mask_id=int(args.mask_id),
            p_mask_site=float(args.p_mask_site),
            mask_both_prob=float(args.mask_both_prob),
            grad_clip=1.0,
        )

        print(f"[epoch {ep:03d}] train_loss={loss:.6f}")
        epoch_list.append(ep)
        loss_list.append(float(loss))
        loss_rows.append({"epoch": ep, "train_loss": float(loss)})

    # Save outputs
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": int(args.vocab_size),
                "d_model": int(args.d_model),
                "n_heads": int(args.n_heads),
                "n_layers": int(args.n_layers),
                "dropout": float(args.dropout),
                "lr": float(args.lr),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "p_mask_site": float(args.p_mask_site),
                "mask_both_prob": float(args.mask_both_prob),
                "mask_id": int(args.mask_id),
                "seed": int(args.seed),
                "window_len": int(args.window_len) if args.window_len is not None else None,
                "window_mode": str(args.window_mode),
                "fixed_start": int(args.fixed_start),
            },
        },
        out_model,
    )

    write_losses_csv(Path(args.out_losses), loss_rows)
    plot_losses(Path(args.out_plot), epoch_list, loss_list)


if __name__ == "__main__":
    main()
