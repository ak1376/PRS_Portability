#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


# -------------------------
# Helpers
# -------------------------

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).mean()) * np.sqrt((y * y).mean()))
    if denom <= 0:
        return float("nan")
    return float((x * y).mean() / denom)

def compute_r2(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    r^2 between two 0/1 vectors across individuals.
    Handles monomorphic sites by returning 0.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    vx = x.var()
    vy = y.var()
    if vx < eps or vy < eps:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(r * r)

def make_pairs(L: int, n_pairs: int, max_dist: int, seed: int) -> np.ndarray:
    """
    Sample pairs (i,j) with j = i + d where d ~ Uniform[1..max_dist].
    Ensures d < L.
    Returns array shape (n_pairs, 2).
    """
    rng = np.random.default_rng(seed)
    max_dist = int(min(max_dist, L - 1))
    if max_dist <= 0:
        raise ValueError(f"max_dist must be >=1 and < L. Got max_dist={max_dist}, L={L}.")

    d = rng.integers(1, max_dist + 1, size=n_pairs)  # [1..max_dist]
    i = rng.integers(0, L - d, size=n_pairs)         # i in [0..L-d-1]
    j = i + d
    return np.stack([i, j], axis=1).astype(np.int64)

def bin_means(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin y by x according to bin edges.
    Returns (bin_midpoints, mean_y, stderr_y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mids = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(len(mids), np.nan, dtype=float)
    se   = np.full(len(mids), np.nan, dtype=float)

    for k in range(len(mids)):
        lo, hi = bins[k], bins[k+1]
        m = (x >= lo) & (x < hi) if k < len(mids)-1 else (x >= lo) & (x <= hi)
        vals = y[m]
        if vals.size == 0:
            continue
        mean[k] = float(vals.mean())
        se[k]   = float(vals.std(ddof=1) / math.sqrt(max(vals.size, 1))) if vals.size > 1 else 0.0
    return mids, mean, se


# -------------------------
# Main
# -------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default=None, help="Optional: config.yaml used to train (to instantiate model).")
    ap.add_argument("--hap", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--window_len", type=int, default=512)
    ap.add_argument("--fixed_start", type=int, default=0)

    ap.add_argument("--n_pairs", type=int, default=5000)
    ap.add_argument("--max_dist", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--mask_id", type=int, default=2)
    ap.add_argument("--pad_id", type=int, default=-1, help="If you use PAD, set it; otherwise leave -1.")

    ap.add_argument("--n_ind", type=int, default=256, help="Subsample individuals for speed.")
    ap.add_argument("--batch_ind", type=int, default=256, help="Batch size of individuals through the model.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load hap and take window ----
    hap_full = np.load(args.hap)  # (N, Ltot)
    N, Ltot = hap_full.shape
    start = int(args.fixed_start)
    wlen  = int(args.window_len)
    if start < 0 or start + wlen > Ltot:
        raise ValueError(f"window out of bounds: start={start}, len={wlen}, Ltot={Ltot}")

    hap = hap_full[:, start:start+wlen].astype(np.int64)  # (N, L)
    L = hap.shape[1]

    # optional: subsample individuals
    rng = np.random.default_rng(int(args.seed))
    n_ind = int(min(args.n_ind, N))
    ind_idx = rng.choice(N, size=n_ind, replace=False)
    hap = hap[ind_idx]  # (n_ind, L)

    print(f"[ld_condmask] Using fixed window: start={start} len={wlen}")
    print(f"[ld_condmask] hap shape used for analysis: (N={hap.shape[0]}, L={hap.shape[1]})")

    # ---- build model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # You likely saved state_dict directly; handle both formats
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Instantiate from config if provided, else infer minimal from state_dict shapes
    if args.config is not None:
        cfg = load_config(Path(args.config))
        # expect your config structure like:
        # model: { vocab_size, d_model, n_heads, n_layers, dropout, max_len, pad_id, pool }
        mcfg = cfg.get("model", cfg)  # tolerate either layout
        vocab_size = int(mcfg.get("vocab_size", 3))
        d_model    = int(mcfg.get("d_model", mcfg.get("embed_dim", 128)))
        n_heads    = int(mcfg.get("n_heads", 8))
        n_layers   = int(mcfg.get("n_layers", 4))
        dropout    = float(mcfg.get("dropout", 0.1))
        max_len    = int(mcfg.get("max_len", 50000))
        pad_id     = None if int(args.pad_id) < 0 else int(args.pad_id)
        pool       = str(mcfg.get("pool", "mean"))
    else:
        # infer d_model and vocab_size from token embedding weight
        tok_w = state_dict["token_emb.weight"]
        vocab_size = int(tok_w.shape[0])
        d_model = int(tok_w.shape[1])
        # try to infer encoder depth
        n_layers = len({k.split(".")[2] for k in state_dict.keys() if k.startswith("encoder.layers.")})
        # heads/dropout not inferable reliably; set common defaults (must match training though)
        n_heads = 8
        dropout = 0.1
        max_len = 50000
        pad_id = None if int(args.pad_id) < 0 else int(args.pad_id)
        pool = "mean"
        print("[ld_condmask] WARNING: no --config provided; some hyperparams inferred/guessed. Prefer passing --config.")

    # IMPORTANT: import your model class
    # Adjust this import if your class lives somewhere else.
    from src.transformer.model import HapMaskTransformer

    model = HapMaskTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        max_len=max_len,
        pad_id=pad_id,
        pool=pool,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    # ---- sample pairs ----
    pairs = make_pairs(L=L, n_pairs=int(args.n_pairs), max_dist=int(args.max_dist), seed=int(args.seed))
    dists = (pairs[:, 1] - pairs[:, 0]).astype(np.int64)

    # ---- precompute r^2 for pairs (cheap) ----
    r2 = np.empty(len(pairs), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        r2[k] = compute_r2(hap[:, i], hap[:, j])

    # ---- compute Δ(i|j) via conditional masking (expensive) ----
    mask_id = int(args.mask_id)
    hap_torch = torch.from_numpy(hap).long()  # (N, L)

    batch_ind = int(args.batch_ind)
    delta = np.empty(len(pairs), dtype=np.float32)

    # We compute per-pair:
    #   run model on batch with i masked -> p1 at i
    #   run model on batch with i and j masked -> p1' at i
    #   delta = mean(|p1 - p1'|)
    #
    # This is O(n_pairs) forward passes * 2. Keep n_pairs modest or n_ind small.
    for k, (i, j) in enumerate(pairs):
        # iterate individuals in chunks to fit GPU memory
        ds = []
        for s in range(0, hap_torch.size(0), batch_ind):
            xb = hap_torch[s:s+batch_ind].to(device)  # (B, L)

            x_i = xb.clone()
            x_i[:, i] = mask_id
            logits_i, _ = model(x_i)
            p1_i = torch.softmax(logits_i[:, i, :], dim=-1)[:, 1]  # (B,)

            x_ij = x_i.clone()
            x_ij[:, j] = mask_id
            logits_ij, _ = model(x_ij)
            p1_ij = torch.softmax(logits_ij[:, i, :], dim=-1)[:, 1]

            ds.append(torch.abs(p1_i - p1_ij).mean().item())
        delta[k] = float(np.mean(ds))

        if (k + 1) % 500 == 0:
            print(f"[ld_condmask] pairs done: {k+1}/{len(pairs)}")

    # ---- save table ----
    out_npz = out_dir / "ld_condmask_results.npz"
    np.savez_compressed(
        out_npz,
        pairs=pairs,
        dist=dists,
        r2=r2,
        delta=delta,
        window_start=start,
        window_len=wlen,
        n_ind=hap.shape[0],
        mask_id=mask_id,
    )
    print(f"[ld_condmask] wrote: {out_npz}")

    # ---- plots ----
    r = pearsonr(r2, delta)

    # scatter: Δ vs r2
    plt.figure(figsize=(8, 6))
    plt.scatter(r2, delta, s=8, alpha=0.25)
    plt.xlabel("LD r² (across individuals)")
    plt.ylabel("Δ(i|j) = mean |P1(mask i) - P1(mask i,j)|")
    plt.title(f"Conditional masking effect vs LD | pearson={r:.3f} | pairs={len(pairs)}")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_vs_r2.png", dpi=200)
    plt.close()

    # binned trend
    bins = np.linspace(0.0, 1.0, 21)
    mids, mean_d, se_d = bin_means(r2, delta, bins=bins)
    plt.figure(figsize=(8, 6))
    plt.errorbar(mids, mean_d, yerr=se_d, marker="o", linestyle="-", capsize=3)
    plt.xlabel("r² bin (midpoint)")
    plt.ylabel("mean Δ(i|j)")
    plt.title("Binned trend: does conditional effect increase with LD?")
    plt.tight_layout()
    plt.savefig(out_dir / "binned_delta_vs_r2.png", dpi=200)
    plt.close()

    # diagnostic: Δ vs distance
    plt.figure(figsize=(8, 6))
    plt.scatter(dists, delta, s=8, alpha=0.25)
    plt.xlabel("distance (sites)")
    plt.ylabel("Δ(i|j)")
    plt.title("Δ vs distance (diagnostic)")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_vs_distance.png", dpi=200)
    plt.close()

    print("[ld_condmask] done.")


if __name__ == "__main__":
    main()
