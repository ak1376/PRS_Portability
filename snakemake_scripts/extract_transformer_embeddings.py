#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pickle

import numpy as np
import torch
import yaml

# Make src/ importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformer.train import load_checkpoint_model


def _load_cfg(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--config not found: {path}")
    return yaml.safe_load(p.read_text()) or {}


def _choose_windows(L_total: int, window_len: int) -> list[tuple[int, int]]:
    if window_len <= 0:
        raise ValueError("window_len must be > 0")
    if window_len >= L_total:
        return [(0, L_total)]

    starts = list(range(0, L_total, window_len))
    windows = []
    for s in starts:
        e = s + window_len
        if e <= L_total:
            windows.append((s, e))
        else:
            windows.append((L_total - window_len, L_total))
            break
    windows = sorted(set(windows))
    return windows


def _maybe_get_labels(meta_path: str | None, N: int):
    if meta_path is None:
        return None
    p = Path(meta_path)
    if not p.exists():
        return None
    try:
        meta = pickle.load(open(p, "rb"))
    except Exception:
        return None

    # Try common patterns
    if isinstance(meta, dict):
        for k in ("pop", "population", "pops", "labels"):
            if k in meta:
                lab = np.asarray(meta[k])
                if lab.shape[0] == N:
                    return lab
    # If it's a dataframe-like
    try:
        import pandas as pd  # optional
        if hasattr(meta, "columns") and hasattr(meta, "__len__"):
            for k in ("pop", "population"):
                if k in meta.columns and len(meta) == N:
                    return np.asarray(meta[k])
    except Exception:
        pass
    return None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    # Snakefile compatibility args (some unused but accepted)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--geno", type=str, required=False)
    ap.add_argument("--meta", type=str, required=False)
    ap.add_argument("--outdir", type=str, required=False)

    ap.add_argument("--hap1", type=str, required=True)
    ap.add_argument("--hap2", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)

    ap.add_argument("--output-embeddings", type=str, required=True)
    ap.add_argument("--output-pca", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    window_len = int(data_cfg.get("window_len", 1024))
    # We’re doing full-genome by averaging across windows; mode doesn’t matter here.

    batch_size = int(args.batch_size or train_cfg.get("batch_size", 32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    # Load haplotypes
    hap1_np = np.load(args.hap1)
    hap2_np = np.load(args.hap2)
    if hap1_np.shape != hap2_np.shape:
        raise ValueError(f"hap1 shape {hap1_np.shape} != hap2 shape {hap2_np.shape}")
    N, L_total = hap1_np.shape

    # Load model
    model, _ckpt = load_checkpoint_model(args.model, device=str(device))
    model.eval()

    windows = _choose_windows(L_total, window_len)
    print(f"[extract_transformer_embeddings] N={N} L_total={L_total} window_len={window_len} n_windows={len(windows)} batch_size={batch_size}")

    z_sum = None
    n_win = 0

    for wi, (s, e) in enumerate(windows, start=1):
        h1w = torch.from_numpy(hap1_np[:, s:e]).long()
        h2w = torch.from_numpy(hap2_np[:, s:e]).long()

        z_chunks = []
        for i0 in range(0, N, batch_size):
            i1 = min(N, i0 + batch_size)
            b1 = h1w[i0:i1].to(device, non_blocking=True)
            b2 = h2w[i0:i1].to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _hap1_logits, _hap2_logits, z = model(b1, b2, pad_mask=None)
            else:
                _hap1_logits, _hap2_logits, z = model(b1, b2, pad_mask=None)

            z_chunks.append(z.detach().cpu())

        z_w = torch.cat(z_chunks, dim=0).numpy().astype(np.float32)  # (N, d)

        if z_sum is None:
            z_sum = z_w
        else:
            z_sum += z_w
        n_win += 1

        if wi % 10 == 0 or wi == 1 or wi == len(windows):
            print(f"[extract_transformer_embeddings] window {wi}/{len(windows)} s={s} e={e}")

    z_mean = z_sum / max(n_win, 1)

    out_z = Path(args.output_embeddings)
    out_z.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_z, z_mean)
    print(f"[extract_transformer_embeddings] wrote embeddings: {z_mean.shape} -> {out_z}")

    # PCA plot
    out_pca = Path(args.output_pca)
    out_pca.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = _maybe_get_labels(args.meta, N)

    Z = z_mean - z_mean.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pcs = U[:, :2] * S[:2]

    plt.figure()
    if labels is None:
        plt.scatter(pcs[:, 0], pcs[:, 1], s=6)
    else:
        uniq = {v: i for i, v in enumerate(sorted(set(labels.tolist())))}
        c = np.array([uniq[v] for v in labels])
        plt.scatter(pcs[:, 0], pcs[:, 1], c=c, s=6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_pca, dpi=200)
    plt.close()
    print(f"[extract_transformer_embeddings] wrote PCA -> {out_pca}")


if __name__ == "__main__":
    main()
