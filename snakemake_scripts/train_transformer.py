#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

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


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def write_losses_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write rows with a union-of-keys header (robust if some rows have extra columns).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")

    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_losses(path: Path, epochs: list[int], train_mlm: list[float], val_mlm: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, train_mlm, label="train_mlm")
    plt.plot(epochs, val_mlm, label="val_mlm")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_contrastive_geometry(
    path: Path,
    epochs: list[int],
    pos_cos: list[float],
    neg_cos: list[float],
    perm_neg_cos: list[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs, pos_cos, label="pos_cos (z1·z2)")
    plt.plot(epochs, neg_cos, label="neg_cos (z1·z2_shuf)")
    if perm_neg_cos is not None:
        plt.plot(epochs, perm_neg_cos, label="perm_neg_cos (z1·zperm)")
    plt.xlabel("epoch")
    plt.ylabel("cosine")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--hap", type=str, required=True, help="Path to ONE haplotype .npy (N,L)")
    ap.add_argument("--out_model", type=str, required=True)
    ap.add_argument("--out_losses", type=str, required=True)
    ap.add_argument("--out_plot", type=str, required=True)
    ap.add_argument("--out_debug_dir", type=str, required=True, help="Directory for debug PNGs")
    ap.add_argument("--out_ctr_plot", type=str, default=None, help="Optional: contrastive geometry plot path")

    ap.add_argument("--config", type=str, default=None)

    # Model overrides (optional)
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--vocab_size", type=int, default=None)
    ap.add_argument("--pad_id", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--pool", type=str, default=None)

    # Training overrides (optional)
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

    # Masking overrides (optional)
    ap.add_argument("--p_mask_site", type=float, default=None)
    ap.add_argument("--mask_id", type=int, default=None)

    # Contrastive (optional)
    ap.add_argument("--contrastive", action="store_true", help="Enable contrastive auxiliary loss")
    ap.add_argument("--contrastive_lambda", type=float, default=None)
    ap.add_argument("--contrastive_tau", type=float, default=None)
    ap.add_argument("--permute_every_k", type=int, default=None)
    ap.add_argument("--no_perm_negatives", action="store_true")
    ap.add_argument("--p_mask_site_ctr", type=float, default=None, help="Masking rate for contrastive views (default: p_mask_site)")

    # Windowing overrides (optional)
    ap.add_argument("--window_len", type=int, default=None)
    ap.add_argument("--window_mode", type=str, default=None, choices=["random", "first", "middle", "fixed"])
    ap.add_argument("--fixed_start", type=int, default=None)

    args = ap.parse_args()

    # ---------------------
    # Load YAML
    # ---------------------
    cfg = _maybe_load_cfg(args.config)

    # Support both:
    #   - flat YAML: {model:..., training:..., masking:..., data:...}
    #   - base YAML: {base: {model:..., training:..., masking:..., data:...}}
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

    # p_mask_site_ctr can live either in CLI or YAML
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
    print(f"[train_transformer_single] window_len={args.window_len} window_mode={args.window_mode} fixed_start={args.fixed_start}")
    print(f"[train_transformer_single] masking p_mask_site={args.p_mask_site} mask_id={args.mask_id}")
    print(
        f"[train_transformer_single] contrastive={args.contrastive} "
        f"lambda={args.contrastive_lambda} tau={args.contrastive_tau} "
        f"perm_neg={args.permute_negatives} permute_every_k={args.permute_every_k} "
        f"p_mask_site_ctr={args.p_mask_site_ctr}"
    )

    # prior for bias init
    with torch.no_grad():
        pi = float(hap.float().mean().item())
        pi = min(max(pi, 1e-4), 1.0 - 1e-4)
        logit_pi = float(np.log(pi / (1.0 - pi)))
        print(f"[init] pi={pi:.6f} logit_pi={logit_pi:.6f}")

    # ---------------------
    # Dataset + loader
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

    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_hapbatch,
        persistent_workers=(int(args.num_workers) > 0),
    )

    # ---------------------
    # Model
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

    # init bias to class prior
    with torch.no_grad():
        model.head.bias.zero_()
        model.head.bias[1] = logit_pi

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    # ---------------------
    # Outputs
    # ---------------------
    out_debug = Path(args.out_debug_dir)
    out_debug.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    epochs: list[int] = []
    train_mlm_losses: list[float] = []
    val_losses: list[float] = []

    # contrastive tracking
    train_ctr_losses: list[float] = []
    ctr_pos_cos: list[float] = []
    ctr_neg_cos: list[float] = []
    ctr_perm_neg_cos: list[float] = []

    # ---------------------
    # Train loop
    # ---------------------
    for ep in range(1, int(args.epochs) + 1):
        tr_out = train_epoch(
            model=model,
            loader=dl,
            optimizer=opt,
            device=device,
            mask_id=int(args.mask_id),
            p_mask_site=float(args.p_mask_site),
            p_mask_site_ctr=(None if args.p_mask_site_ctr is None else float(args.p_mask_site_ctr)),
            grad_clip=float(args.grad_clip),
            # contrastive knobs
            use_contrastive=bool(args.contrastive),
            contrastive_lambda=float(args.contrastive_lambda),
            contrastive_tau=float(args.contrastive_tau),
            use_perm_negatives=bool(args.permute_negatives),
            permute_every_k=int(args.permute_every_k),
        )

        tr_mlm = float(tr_out["mlm_loss"])
        tr_ctr = float(tr_out.get("ctr_loss", float("nan")))
        pos_cos = float(tr_out.get("ctr_pos_cos", float("nan")))
        neg_cos = float(tr_out.get("ctr_neg_cos", float("nan")))
        perm_cos = float(tr_out.get("ctr_perm_neg_cos", float("nan")))

        va = float(
            eval_epoch(
                model=model,
                loader=dl,
                device=device,
                mask_id=int(args.mask_id),
                p_mask_site=float(args.p_mask_site),
            )
        )

        if args.contrastive:
            msg = (
                f"[epoch {ep:03d}] train_mlm={tr_mlm:.6f} train_ctr={tr_ctr:.6f} "
                f"pos_cos={pos_cos:.4f} neg_cos={neg_cos:.4f}"
            )
            if not np.isnan(perm_cos):
                msg += f" perm_neg_cos={perm_cos:.4f}"
            msg += f" val_mlm={va:.6f}"
            print(msg)
        else:
            print(f"[epoch {ep:03d}] train_mlm={tr_mlm:.6f} val_mlm={va:.6f}")

        # Debug snapshots
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

        row: dict[str, Any] = {"epoch": ep, "train_mlm_loss": tr_mlm, "val_mlm_loss": va}
        if args.contrastive:
            row["train_ctr_loss"] = tr_ctr
            row["ctr_pos_cos"] = pos_cos
            row["ctr_neg_cos"] = neg_cos
            if not np.isnan(perm_cos):
                row["ctr_perm_neg_cos"] = perm_cos
        rows.append(row)

        epochs.append(ep)
        train_mlm_losses.append(tr_mlm)
        val_losses.append(va)

        if args.contrastive:
            train_ctr_losses.append(tr_ctr)
            ctr_pos_cos.append(pos_cos)
            ctr_neg_cos.append(neg_cos)
            ctr_perm_neg_cos.append(perm_cos)

    # ---------------------
    # Save checkpoint
    # ---------------------
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                # model
                "vocab_size": int(args.vocab_size),
                "d_model": int(args.d_model),
                "n_heads": int(args.n_heads),
                "n_layers": int(args.n_layers),
                "dropout": float(args.dropout),
                "pad_id": args.pad_id,
                "pool": str(args.pool),
                "max_len": int(args.max_len),
                # training
                "lr": float(args.lr),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
                "weight_decay": float(args.weight_decay),
                "grad_clip": float(args.grad_clip),
                # masking
                "p_mask_site": float(args.p_mask_site),
                "mask_id": int(args.mask_id),
                # contrastive
                "contrastive": bool(args.contrastive),
                "contrastive_lambda": float(args.contrastive_lambda),
                "contrastive_tau": float(args.contrastive_tau),
                "permute_negatives": bool(args.permute_negatives),
                "permute_every_k": int(args.permute_every_k),
                "p_mask_site_ctr": (None if args.p_mask_site_ctr is None else float(args.p_mask_site_ctr)),
                # windowing
                "window_len": int(args.window_len) if args.window_len is not None else None,
                "window_mode": str(args.window_mode),
                "fixed_start": int(args.fixed_start),
            },
        },
        out_model,
    )

    # ---------------------
    # Write CSV + plots
    # ---------------------
    write_losses_csv(Path(args.out_losses), rows)
    plot_losses(Path(args.out_plot), epochs, train_mlm_losses, val_losses)

    if args.contrastive:
        out_ctr_plot = (
            Path(args.out_ctr_plot)
            if args.out_ctr_plot is not None
            else Path(args.out_plot).with_name("contrastive_geometry.png")
        )
        perm_ok = not np.all(np.isnan(np.asarray(ctr_perm_neg_cos, dtype=float)))
        plot_contrastive_geometry(
            out_ctr_plot,
            epochs,
            ctr_pos_cos,
            ctr_neg_cos,
            ctr_perm_neg_cos if perm_ok else None,
        )


if __name__ == "__main__":
    main()
