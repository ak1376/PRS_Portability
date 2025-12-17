#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math

import numpy as np
import torch
import yaml
import pandas as pd  # NEW

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- make sure we can import src.* no matter where we run from ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -------------------------
# Helpers
# -------------------------

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 1.0
    return float(1.0 - (np.dot(a, b) / (na * nb)))

def hamming_distance01(x: np.ndarray, y: np.ndarray) -> float:
    # x,y: (L,) with 0/1
    return float(np.mean(x != y))  # normalized [0,1]

def maf_weights(hap: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # hap: (N,L) 0/1
    p = hap.mean(axis=0)
    return (1.0 / np.sqrt(p * (1.0 - p) + eps)).astype(np.float32)

def weighted_hamming01(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    m = (x != y).astype(np.float32)
    num = float(np.sum(w * m))
    den = float(np.sum(w)) + 1e-12
    return num / den

def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = (np.sqrt((rx * rx).mean()) * np.sqrt((ry * ry).mean()))
    if denom <= 0:
        return float("nan")
    return float((rx * ry).mean() / denom)

def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).mean()) * np.sqrt((y * y).mean()))
    if denom <= 0:
        return float("nan")
    return float((x * y).mean() / denom)

def sample_pairs_from_index_list(idxs: np.ndarray, n_pairs: int, seed: int) -> np.ndarray:
    """
    Sample pairs (i,j) within a subset of indices (e.g., within one population).
    idxs are indices into the *current* arrays hap/E/pop (0..n_ind-1).
    Returns (n_pairs, 2) with i<j.
    """
    rng = np.random.default_rng(seed)
    m = idxs.size
    if m < 2:
        return np.zeros((0, 2), dtype=np.int64)

    a = rng.integers(0, m, size=n_pairs)
    b = rng.integers(0, m, size=n_pairs)
    ok = a != b
    a, b = a[ok], b[ok]

    while a.size < n_pairs:
        aa = rng.integers(0, m, size=n_pairs)
        bb = rng.integers(0, m, size=n_pairs)
        ok2 = aa != bb
        a = np.concatenate([a, aa[ok2]])
        b = np.concatenate([b, bb[ok2]])

    a = a[:n_pairs]
    b = b[:n_pairs]

    i = idxs[a]
    j = idxs[b]
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    return np.stack([lo, hi], axis=1).astype(np.int64)

def bin_means(x: np.ndarray, y: np.ndarray, bins: np.ndarray):
    mids = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(len(mids), np.nan, dtype=float)
    se   = np.full(len(mids), np.nan, dtype=float)
    nbin = np.zeros(len(mids), dtype=int)

    for k in range(len(mids)):
        lo, hi = bins[k], bins[k+1]
        m = (x >= lo) & (x < hi) if k < len(mids)-1 else (x >= lo) & (x <= hi)
        vals = y[m]
        nbin[k] = int(vals.size)
        if vals.size == 0:
            continue
        mean[k] = float(vals.mean())
        se[k]   = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mids, mean, se, nbin


# -------------------------
# Main
# -------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--hap", type=str, required=True, help="hap matrix (N, Ltot) with 0/1 alleles")
    ap.add_argument("--hap_meta", type=str, required=True, help="hap_meta.pkl with columns incl. hap_index and population")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--window_len", type=int, default=512)
    ap.add_argument("--fixed_start", type=int, default=0)

    ap.add_argument("--n_ind", type=int, default=256, help="how many haplotypes (rows) to subsample")
    ap.add_argument("--batch_ind", type=int, default=256)
    ap.add_argument("--n_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_weighted_hamming", action="store_true",
                    help="Use MAF-weighted Hamming instead of plain Hamming.")
    ap.add_argument("--cosine_metric", choices=["cosdist", "cossim"], default="cosdist",
                    help="Compare against cosine distance (1-cos) or cosine similarity.")
    
    ap.add_argument("--hap_id", type=int, default=1,
                help="Which haplotype this file corresponds to (0 or 1). Use 1 for hap1.npy, 0 for hap0.npy.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load hap and slice window ----
    hap_full = np.load(args.hap)  # (N, Ltot)
    N, Ltot = hap_full.shape
    start = int(args.fixed_start)
    wlen = int(args.window_len)
    if start < 0 or start + wlen > Ltot:
        raise ValueError(f"window out of bounds: start={start}, len={wlen}, Ltot={Ltot}")
    hap_full = hap_full[:, start:start+wlen].astype(np.int64)  # (N, L)

    # ---- load hap_meta.pkl and align to hap row order via hap_index ----
    meta = pd.read_pickle(args.hap_meta)

    req = {"individual_id", "population", "hap_id"}
    missing = req - set(meta.columns)
    if missing:
        raise ValueError(f"hap_meta.pkl missing columns: {sorted(missing)}")

    # Filter meta to match which hap matrix you passed (hap0 vs hap1)
    hid = int(args.hap_id)
    if hid not in (0, 1):
        raise ValueError("--hap_id must be 0 or 1")

    meta = meta.loc[meta["hap_id"] == hid].copy()

    # Now meta should have one row per individual (N rows)
    # Align to hap rows by individual_id order (robust)
    meta = meta.sort_values("individual_id")
    if meta.shape[0] != N:
        raise ValueError(f"After filtering hap_id=={hid}, meta has {meta.shape[0]} rows but hap has N={N} rows.")

    if meta["individual_id"].iloc[0] != 0 or meta["individual_id"].iloc[-1] != N - 1:
        # Not necessarily fatal, but usually indicates mismatch
        print(f"[warn] individual_id range is {meta['individual_id'].min()}..{meta['individual_id'].max()} but expected 0..{N-1}")

    pop_full = meta["population"].to_numpy()


    # ---- subsample haplotypes (rows) ----
    rng = np.random.default_rng(int(args.seed))
    n_ind = int(min(args.n_ind, N))
    idx = rng.choice(N, size=n_ind, replace=False)

    hap = hap_full[idx]      # (n_ind, L)
    pop = pop_full[idx]      # (n_ind,)

    print(f"[hap_vs_emb_pop] window_start={start} window_len={wlen} hap shape=(N={hap.shape[0]}, L={hap.shape[1]})")
    uniq, cnt = np.unique(pop, return_counts=True)
    print("[hap_vs_emb_pop] pop counts in subsample:", dict(zip(uniq.tolist(), cnt.tolist())))

    # weights if requested (computed on whole subsample)
    w = maf_weights(hap) if args.use_weighted_hamming else None

    # ---- build model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    cfg = load_config(Path(args.config))
    mcfg = cfg.get("model", cfg)

    vocab_size = int(mcfg.get("vocab_size", 3))
    d_model    = int(mcfg.get("d_model", mcfg.get("embed_dim", 128)))
    n_heads    = int(mcfg.get("n_heads", 8))
    n_layers   = int(mcfg.get("n_layers", 4))
    dropout    = float(mcfg.get("dropout", 0.1))
    max_len    = int(mcfg.get("max_len", 50000))
    pool       = str(mcfg.get("pool", "mean"))
    pad_id_cfg = mcfg.get("pad_id", None)
    pad_id = None if pad_id_cfg is None else int(pad_id_cfg)

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
    model.eval().to(device)

    # ---- compute pooled embeddings for each haplotype ----
    hap_t = torch.from_numpy(hap).long()
    batch = int(args.batch_ind)

    embs = []
    for s in range(0, hap_t.size(0), batch):
        xb = hap_t[s:s+batch].to(device)  # (B, L)
        _logits, aux = model(xb)
        if not torch.is_tensor(aux) or aux.ndim != 2:
            raise RuntimeError(f"Expected pooled aux (B,d); got type={type(aux)} shape={getattr(aux,'shape',None)}")
        embs.append(aux.detach().cpu().numpy().astype(np.float32))

    E = np.concatenate(embs, axis=0)  # (n_ind, d)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    # ---- compute per-pop distances (within-pop pairs) ----
    metric_name = "cosine_distance" if args.cosine_metric == "cosdist" else "cosine_similarity"
    ham_name = "weighted_hamming" if args.use_weighted_hamming else "hamming"

    bins = np.linspace(0.0, 1.0, 21)

    results = {}
    for p in np.unique(pop):
        idxs = np.where(pop == p)[0]
        if idxs.size < 2:
            continue
        pairs = sample_pairs_from_index_list(idxs, n_pairs=int(args.n_pairs), seed=int(args.seed) + (hash(str(p)) % 100000))

        ham = np.empty(len(pairs), dtype=np.float32)
        cosv = np.empty(len(pairs), dtype=np.float32)

        for k, (i, j) in enumerate(pairs):
            if w is None:
                ham[k] = hamming_distance01(hap[i], hap[j])
            else:
                ham[k] = weighted_hamming01(hap[i], hap[j], w)

            cdist = cosine_distance(E[i], E[j])
            cosv[k] = (1.0 - cdist) if args.cosine_metric == "cossim" else cdist

        sp = spearmanr(ham, cosv)
        pr = pearsonr(ham, cosv)
        results[str(p)] = {"ham": ham, "cos": cosv, "spearman": sp, "pearson": pr}

        print(f"[hap_vs_emb_pop] {p}: Spearman={sp:.4f}  Pearson={pr:.4f}  pairs={len(pairs)}")

    # ---- save npz ----
    npz_kwargs = {
        "pops": np.array(list(results.keys()), dtype=object),
        "ham_metric": ham_name,
        "cos_metric": metric_name,
        "window_start": start,
        "window_len": wlen,
        "n_ind": hap.shape[0],
        "embed_dim": E.shape[1],
        "seed": int(args.seed),
    }
    for p, d in results.items():
        npz_kwargs[f"ham_{p}"] = d["ham"]
        npz_kwargs[f"cos_{p}"] = d["cos"]
        npz_kwargs[f"spearman_{p}"] = np.array(d["spearman"])
        npz_kwargs[f"pearson_{p}"] = np.array(d["pearson"])

    np.savez_compressed(out_dir / "haplotype_similarity_vs_embedding_by_pop.npz", **npz_kwargs)

    # ---- plots ----
    # Scatter (colored by pop)
    plt.figure(figsize=(7, 6))
    for p, d in results.items():
        plt.scatter(d["ham"], d["cos"], s=8, alpha=0.20, label=p)
    plt.xlabel(f"{ham_name} (fraction mismatched)")
    plt.ylabel(metric_name)
    plt.title(f"{ham_name} vs {metric_name} (within-pop pairs)")
    plt.legend(frameon=True, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_hamming_vs_cosine_by_pop.png", dpi=200)
    plt.close()

    # Binned trends per pop
    plt.figure(figsize=(7, 6))
    for p, d in results.items():
        mids, mean, se, nbin = bin_means(d["ham"], d["cos"], bins=bins)
        plt.errorbar(mids, mean, yerr=se, marker="o", linestyle="-", capsize=3, label=p)
    plt.xlabel(f"{ham_name} bin midpoint")
    plt.ylabel(metric_name)
    plt.title("Binned trend by population (within-pop pairs)")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "binned_hamming_vs_cosine_by_pop.png", dpi=200)
    plt.close()

    print(f"[hap_vs_emb_pop] wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
