#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import math
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------
# Utilities
# ---------------------------

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]
    cfg = ckpt.get("config", {})

    # Import your single-hap model
    from src.transformer.model import HapMaskTransformer

    model = HapMaskTransformer(
        vocab_size=int(cfg.get("vocab_size", 3)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 8)),
        n_layers=int(cfg.get("n_layers", 6)),
        dropout=float(cfg.get("dropout", 0.1)),
        pad_id=cfg.get("pad_id", None),
        pool=str(cfg.get("pool", "mean")),
        max_len=int(cfg.get("max_len", 50_000)),
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def corr2(x: np.ndarray, y: np.ndarray) -> float:
    """
    r^2 between two 0/1 vectors across individuals.
    Returns 0.0 if variance is zero or numeric issues.
    """
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x -= x.mean()
    y -= y.mean()
    vx = np.dot(x, x)
    vy = np.dot(y, y)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    r = float(np.dot(x, y) / math.sqrt(vx * vy))
    return r * r


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = u.astype(np.float64, copy=False)
    v = v.astype(np.float64, copy=False)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu <= 1e-12 or nv <= 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a = a - a.mean()
    b = b - b.mean()
    da = np.dot(a, a)
    db = np.dot(b, b)
    if da <= 1e-12 or db <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / math.sqrt(da * db))


def make_pairs(L: int, n_pairs: int, max_dist: int | None, seed: int = 0):
    rng = np.random.default_rng(seed)

    # distances must satisfy 1 <= d <= L-1
    max_d = (L - 1) if (max_dist is None) else min(int(max_dist), L - 1)
    if max_d < 1:
        raise ValueError(f"L={L} too small for pair sampling.")

    # sample distances uniformly from [1, max_d]
    d = rng.integers(1, max_d + 1, size=n_pairs)

    # for each distance, choose i in [0, L-d)
    i = np.empty(n_pairs, dtype=np.int64)
    for dist in np.unique(d):
        mask = (d == dist)
        i[mask] = rng.integers(0, L - dist, size=mask.sum())

    j = i + d
    return np.stack([i, j], axis=1)  # (n_pairs, 2)


# ---------------------------
# Main
# ---------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to trained model.pt checkpoint")
    ap.add_argument("--hap", required=True, help="Path to hap .npy (N,L_total) with 0/1 tokens (no MASKs)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # windowing for consistent SNP identity
    ap.add_argument("--window_len", type=int, default=None, help="If set, evaluate on this fixed window length")
    ap.add_argument("--fixed_start", type=int, default=0, help="Window start if window_len is set")

    # pair sampling
    ap.add_argument("--n_pairs", type=int, default=20000, help="How many SNP pairs to sample")
    ap.add_argument("--max_dist", type=int, default=512, help="Max distance between SNP pairs (in sites). Set 0 to disable.")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Load hap
    hap_all = np.load(args.hap)
    if hap_all.ndim != 2:
        raise ValueError(f"--hap must be 2D (N,L). Got {hap_all.shape}")
    N, L_total = hap_all.shape

    # Apply fixed window if requested (IMPORTANT for SNP identity!)
    if args.window_len is not None:
        s = int(args.fixed_start)
        e = s + int(args.window_len)
        if e > L_total:
            raise ValueError(f"window exceeds length: start={s} len={args.window_len} L_total={L_total}")
        hap_all = hap_all[:, s:e]
        print(f"[ld_vs_cosine] Using fixed window: start={s} len={args.window_len}")
    else:
        print("[ld_vs_cosine] Using full hap length (no windowing)")

    N, L = hap_all.shape
    print(f"[ld_vs_cosine] hap shape used for analysis: (N={N}, L={L})")

    # Load model
    model, ckpt = load_model(args.ckpt, device)

    # Build DataLoader over individuals
    hap_tensor = torch.from_numpy(hap_all).long()
    ds = torch.utils.data.TensorDataset(hap_tensor)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Compute E = mean over individuals of H[:,i,:]
    # We support two paths:
    #  (A) if you added return_site_embeddings=True and it returns E
    #  (B) fallback: get H and average ourselves
    E_sum = None
    E_count = 0

    for (hap_batch,) in dl:
        hap_batch = hap_batch.to(device)  # (B,L)

        # --- Try "return_site_embeddings" if your model supports it ---
        try:
            logits, z, E_batch = model(hap_batch, return_site_embeddings=True)
            # E_batch: (L,d)
            Eb = E_batch.detach().cpu()
        except TypeError:
            # fallback: request H then average over batch
            logits, z, H = model(hap_batch, return_site_features=True)
            # H: (B,L,d) -> mean over B => (L,d)
            Eb = H.detach().cpu().mean(dim=0)

        if E_sum is None:
            E_sum = Eb.clone()
        else:
            E_sum += Eb
        E_count += 1

    assert E_sum is not None
    E = (E_sum / float(E_count)).numpy()  # (L,d)
    print(f"[ld_vs_cosine] Built E: shape={E.shape} from {E_count} batches")

    # Sample pairs
    max_dist = None if int(args.max_dist) <= 0 else int(args.max_dist)
    pairs = make_pairs(L=L, n_pairs=int(args.n_pairs), max_dist=max_dist, seed=int(args.seed))

    # Compute r2 and cosine for pairs
    r2s = np.zeros(len(pairs), dtype=np.float64)
    coss = np.zeros(len(pairs), dtype=np.float64)
    dists = np.zeros(len(pairs), dtype=np.int64)

    for k, (i, j) in enumerate(pairs):
        xi = hap_all[:, i]
        xj = hap_all[:, j]
        r2s[k] = corr2(xi, xj)
        coss[k] = cosine(E[i], E[j])
        dists[k] = j - i

    # Save CSV
    out_csv = out_dir / "ld_vs_cosine_pairs.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "dist_sites", "r2", "cosine"])
        for (i, j), d, r2, cs in zip(pairs, dists, r2s, coss):
            w.writerow([int(i), int(j), int(d), float(r2), float(cs)])
    print(f"[ld_vs_cosine] Wrote {out_csv}")

    # Correlation summary
    r = pearson(r2s, coss)
    with (out_dir / "summary.txt").open("w") as f:
        f.write(f"N={N} L={L}\n")
        f.write(f"n_pairs={len(pairs)} max_dist={args.max_dist}\n")
        f.write(f"pearson(r2, cosine)={r:.6f}\n")
    print(f"[ld_vs_cosine] pearson(r2, cosine)={r:.4f}")

    # Plot scatter
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(r2s, coss, s=6, alpha=0.25)
    plt.xlabel("LD r² (across individuals)")
    plt.ylabel("cosine(E[i], E[j])")
    plt.title(f"LD vs Embedding similarity | pearson={r:.3f} | pairs={len(pairs)}")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_ld_vs_cosine.png", dpi=220)
    plt.close()

    # Plot binned by r2
    nb = 25
    bins = np.linspace(0.0, 1.0, nb + 1)
    bin_idx = np.digitize(r2s, bins) - 1
    x_mid = 0.5 * (bins[:-1] + bins[1:])
    y_mean = np.full(nb, np.nan)
    y_sem = np.full(nb, np.nan)

    for b in range(nb):
        m = bin_idx == b
        if np.sum(m) >= 10:
            vals = coss[m]
            y_mean[b] = vals.mean()
            y_sem[b] = vals.std(ddof=1) / math.sqrt(len(vals))

    plt.figure(figsize=(6.5, 4.5))
    plt.errorbar(x_mid, y_mean, yerr=y_sem, fmt="o-", capsize=3)
    plt.xlabel("r² bin (midpoint)")
    plt.ylabel("mean cosine(E)")
    plt.title("Binned trend: does cosine increase with LD?")
    plt.tight_layout()
    plt.savefig(out_dir / "binned_cosine_by_r2.png", dpi=220)
    plt.close()

    # Plot cosine vs distance (to diagnose “distance-only” effects)
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(dists, coss, s=6, alpha=0.25)
    plt.xlabel("distance (sites)")
    plt.ylabel("cosine(E[i], E[j])")
    plt.title("Cosine vs distance (diagnostic)")
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_vs_distance.png", dpi=220)
    plt.close()

    # ------------------------------------------------------------
    # Distance-stratified cosine vs LD (KEY DIAGNOSTIC)
    # ------------------------------------------------------------

    def bin_means(x, y, bins):
        x = np.asarray(x)
        y = np.asarray(y)
        mids = 0.5 * (bins[:-1] + bins[1:])
        mean = np.full(len(mids), np.nan)
        sem  = np.full(len(mids), np.nan)
        n    = np.zeros(len(mids), dtype=int)

        for k in range(len(mids)):
            lo, hi = bins[k], bins[k+1]
            m = (x >= lo) & (x < hi) if k < len(mids)-1 else (x >= lo) & (x <= hi)
            vals = y[m]
            n[k] = vals.size
            if vals.size >= 5:
                mean[k] = vals.mean()
                sem[k]  = vals.std(ddof=1) / math.sqrt(vals.size)
        return mids, mean, sem, n


    # distance bins (adjust to taste)
    dist_bins = [(1,20), (20,50), (50,100), (100,200), (200, args.max_dist)]

    r2_bins = np.linspace(0.0, 1.0, 21)

    for dlo, dhi in dist_bins:
        m = (
            (dists >= dlo) &
            (dists < dhi) &
            np.isfinite(r2s) &
            np.isfinite(coss)
        )

        if m.sum() < 100:
            continue

        r = pearson(r2s[m], coss[m])

        mids, mean_c, sem_c, nbin = bin_means(r2s[m], coss[m], r2_bins)

        plt.figure(figsize=(6.5, 4.5))
        plt.errorbar(mids, mean_c, yerr=sem_c, fmt="o-", capsize=3)
        plt.xlabel("LD r² (across individuals)")
        plt.ylabel("mean cosine(E)")
        plt.title(
            f"cosine vs r² | dist∈[{dlo},{dhi}) | "
            f"pearson={r:.3f} | pairs={m.sum()}"
        )
        plt.tight_layout()
        plt.savefig(
            out_dir / f"binned_cosine_vs_r2_dist_{dlo}_{dhi}.png",
            dpi=220
        )
        plt.close()



if __name__ == "__main__":
    main()
