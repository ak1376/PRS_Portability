#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformer.data_class import HapDataset, collate_hapbatch
from src.transformer.model import HapMaskTransformer
from src.transformer.train import train_epoch, eval_epoch, debug_snapshot_and_pngs


def write_losses_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_losses(path: Path, epochs: list[int], train_losses: list[float], val_losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
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


def _pick(cli_val, yaml_val, default):
    return default if (cli_val is None and yaml_val is None) else (yaml_val if cli_val is None else cli_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hap", type=str, required=True, help="Path to ONE haplotype .npy (N,L)")
    ap.add_argument("--out_model", type=str, required=True)
    ap.add_argument("--out_losses", type=str, required=True)
    ap.add_argument("--out_plot", type=str, required=True)
    ap.add_argument("--out_debug_dir", type=str, required=True, help="Directory for debug PNGs")

    ap.add_argument("--config", type=str, default=None)

    # Model
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--vocab_size", type=int, default=None)
    ap.add_argument("--pad_id", type=int, default=None)

    # Training
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--p_mask_site", type=float, default=None)
    ap.add_argument("--mask_id", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug_every", type=int, default=None)
    ap.add_argument("--debug_n_show", type=int, default=None)
    ap.add_argument("--debug_max_sites", type=int, default=None)

    # Windowing
    ap.add_argument("--window_len", type=int, default=None)
    ap.add_argument("--window_mode", type=str, default=None, choices=["random", "first", "middle", "fixed"])
    ap.add_argument("--fixed_start", type=int, default=None)

    args = ap.parse_args()
    cfg = _maybe_load_cfg(args.config)
    model_cfg = cfg.get("model", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    args.d_model = _pick(args.d_model, model_cfg.get("d_model"), 128)
    args.n_heads = _pick(args.n_heads, model_cfg.get("n_heads"), 8)
    args.n_layers = _pick(args.n_layers, model_cfg.get("n_layers"), 6)
    args.dropout = _pick(args.dropout, model_cfg.get("dropout"), 0.1)
    args.vocab_size = _pick(args.vocab_size, model_cfg.get("vocab_size"), 3)
    args.pad_id = _pick(args.pad_id, model_cfg.get("pad_id"), None)

    args.epochs = _pick(args.epochs, train_cfg.get("epochs"), 10)
    args.batch_size = _pick(args.batch_size, train_cfg.get("batch_size"), 32)
    args.lr = _pick(args.lr, train_cfg.get("lr"), 3e-4)
    args.p_mask_site = _pick(args.p_mask_site, train_cfg.get("p_mask_site"), 0.15)
    args.mask_id = _pick(args.mask_id, train_cfg.get("mask_id"), 2)
    args.num_workers = _pick(args.num_workers, train_cfg.get("num_workers"), 2)
    args.debug_every = _pick(args.debug_every, train_cfg.get("debug_every"), 5)
    args.debug_n_show = _pick(args.debug_n_show, train_cfg.get("debug_n_show"), 2)
    args.debug_max_sites = _pick(args.debug_max_sites, train_cfg.get("debug_max_sites"), 256)

    args.window_len = _pick(args.window_len, data_cfg.get("window_len"), 1024)
    args.window_mode = _pick(args.window_mode, data_cfg.get("window_mode"), "random")
    args.fixed_start = _pick(args.fixed_start, data_cfg.get("fixed_start"), 0)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hap = torch.from_numpy(np.load(args.hap)).long()
    print(f"[train_transformer_single] hap shape: {tuple(hap.shape)}")
    print(f"[train_transformer_single] window_len={args.window_len} window_mode={args.window_mode}")

    # prior for bias init
    with torch.no_grad():
        pi = float(hap.float().mean().item())
        pi = min(max(pi, 1e-4), 1.0 - 1e-4)
        logit_pi = float(np.log(pi / (1.0 - pi)))
        print(f"[init] pi={pi:.6f} logit_pi={logit_pi:.6f}")

    g = torch.Generator().manual_seed(int(args.seed))
    ds = HapDataset(
        hap_all=hap,
        pad_id=args.pad_id,
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
        collate_fn=collate_hapbatch,
        persistent_workers=(int(args.num_workers) > 0),
    )

    model = HapMaskTransformer(
        vocab_size=int(args.vocab_size),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        pad_id=args.pad_id,
    ).to(device)

    # init bias to class prior
    with torch.no_grad():
        model.head.bias.zero_()
        model.head.bias[1] = logit_pi

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    out_debug = Path(args.out_debug_dir)
    out_debug.mkdir(parents=True, exist_ok=True)

    rows = []
    epochs = []
    train_losses = []
    val_losses = []

    for ep in range(1, int(args.epochs) + 1):
        tr = train_epoch(
            model=model,
            loader=dl,
            optimizer=opt,
            device=device,
            mask_id=int(args.mask_id),
            p_mask_site=float(args.p_mask_site),
            grad_clip=1.0,
        )
        # quick val on one batch (cheap sanity) or make a real val split if you want
        va = eval_epoch(
            model=model,
            loader=dl,
            device=device,
            mask_id=int(args.mask_id),
            p_mask_site=float(args.p_mask_site),
        )

        print(f"[epoch {ep:03d}] train_loss={tr:.6f} val_loss={va:.6f}")

        if int(args.debug_every) > 0 and (ep % int(args.debug_every) == 0):
            batch = next(iter(dl))
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

        rows.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        epochs.append(ep)
        train_losses.append(tr)
        val_losses.append(va)

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
                "pad_id": args.pad_id,
                "lr": float(args.lr),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "p_mask_site": float(args.p_mask_site),
                "mask_id": int(args.mask_id),
                "seed": int(args.seed),
                "window_len": int(args.window_len) if args.window_len is not None else None,
                "window_mode": str(args.window_mode),
                "fixed_start": int(args.fixed_start),
            },
        },
        out_model,
    )

    write_losses_csv(Path(args.out_losses), rows)
    plot_losses(Path(args.out_plot), epochs, train_losses, val_losses)


if __name__ == "__main__":
    main()
