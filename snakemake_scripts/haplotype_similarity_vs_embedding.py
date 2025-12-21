#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math

import numpy as np
import torch
import yaml
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    return float(np.mean(x != y))

def maf_weights(hap: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = hap.mean(axis=0)
    return (1.0 / np.sqrt(p * (1.0 - p) + eps)).astype(np.float32)

def weighted_hamming01(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    m = (x != y).astype(np.float32)
    return float(np.sum(w * m)) / (float(np.sum(w)) + 1e-12)

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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--hap", type=str, required=True)
    ap.add_argument("--hap_meta", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--window_len", type=int, default=512)
    ap.add_argument("--fixed_start", type=int, default=0)

    ap.add_argument("--n_ind", type=int, default=256)
    ap.add_argument("--batch_ind", type=int, default=256)
    ap.add_argument("--n_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_weighted_hamming", action="store_true")
    ap.add_argument("--cosine_metric", choices=["cosdist", "cossim"], default="cosdist")

    ap.add_argument("--hap_id", type=int, default=1)

    # NEW: exact-mcount strat settings
    ap.add_argument("--top_k_mcount", type=int, default=8,
                    help="Show the top-K most frequent mismatch counts (per pop) in stratified plot.")
    ap.add_argument("--min_pairs_per_mcount", type=int, default=200,
                    help="Only include mismatch-count strata with at least this many pairs.")
    # fallback binning if weighted
    ap.add_argument("--strata_bins", type=int, default=10,
                    help="If --use_weighted_hamming, stratify using this many bins in [0,1].")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load hap and slice window ----
    hap_full = np.load(args.hap)
    N, Ltot = hap_full.shape
    start = int(args.fixed_start)
    wlen = int(args.window_len)
    if start < 0 or start + wlen > Ltot:
        raise ValueError(f"window out of bounds: start={start}, len={wlen}, Ltot={Ltot}")
    hap_full = hap_full[:, start:start+wlen].astype(np.int64)  # (N,L)
    L = hap_full.shape[1]

    # ---- meta align ----
    meta = pd.read_pickle(args.hap_meta)
    req = {"individual_id", "population", "hap_id"}
    missing = req - set(meta.columns)
    if missing:
        raise ValueError(f"hap_meta.pkl missing columns: {sorted(missing)}")

    hid = int(args.hap_id)
    meta = meta.loc[meta["hap_id"] == hid].copy()
    meta = meta.sort_values("individual_id")
    if meta.shape[0] != N:
        raise ValueError(f"After filtering hap_id=={hid}, meta has {meta.shape[0]} rows but hap has N={N} rows.")

    pop_full = meta["population"].to_numpy()

    # ---- subsample ----
    rng = np.random.default_rng(int(args.seed))
    n_ind = int(min(args.n_ind, N))
    idx = rng.choice(N, size=n_ind, replace=False)
    hap = hap_full[idx]
    pop = pop_full[idx]

    print(f"[hap_vs_emb_pop] window_start={start} window_len={wlen} hap shape=(N={hap.shape[0]}, L={hap.shape[1]})")
    uniq, cnt = np.unique(pop, return_counts=True)
    print("[hap_vs_emb_pop] pop counts in subsample:", dict(zip(uniq.tolist(), cnt.tolist())))

    # weights only if requested
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

    # ---- pooled z embeddings ----
    hap_t = torch.from_numpy(hap).long()
    batch = int(args.batch_ind)

    embs = []
    for s in range(0, hap_t.size(0), batch):
        xb = hap_t[s:s+batch].to(device)
        _logits, z = model(xb)
        embs.append(z.detach().cpu().numpy().astype(np.float32))

    Z = np.concatenate(embs, axis=0)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    # ---- compute per-pop ----
    metric_name = "cosine_distance" if args.cosine_metric == "cosdist" else "cosine_similarity"
    ham_name = "weighted_hamming" if args.use_weighted_hamming else "hamming"

    bins_trend = np.linspace(0.0, 1.0, 21)
    results = {}
    strat_rows = []

    # weighted fallback bins
    bins_weighted = np.linspace(0.0, 1.0, int(args.strata_bins) + 1)
    wlabs = []
    for i in range(len(bins_weighted) - 1):
        lo, hi = bins_weighted[i], bins_weighted[i+1]
        wlabs.append(f"[{lo:.2f},{hi:.2f})" if i < len(bins_weighted) - 2 else f"[{lo:.2f},{hi:.2f}]")

    for p in np.unique(pop):
        idxs = np.where(pop == p)[0]
        if idxs.size < 2:
            continue

        pairs = sample_pairs_from_index_list(idxs, n_pairs=int(args.n_pairs),
                                            seed=int(args.seed) + (hash(str(p)) % 100000))

        ham = np.empty(len(pairs), dtype=np.float32)
        cosv = np.empty(len(pairs), dtype=np.float32)

        # NEW: exact mismatch counts (only meaningful for plain Hamming)
        mcount = np.empty(len(pairs), dtype=np.int32)

        # fallback: weighted stratum index
        wstratum = np.empty(len(pairs), dtype=np.int32)

        for k, (i, j) in enumerate(pairs):
            mism = (hap[i] != hap[j])
            mcount[k] = int(mism.sum())

            if w is None:
                h = float(mcount[k]) / float(L)
            else:
                h = weighted_hamming01(hap[i], hap[j], w)
                # weighted stratum assignment
                si = int(np.digitize(h, bins_weighted, right=False) - 1)
                si = max(0, min(si, len(wlabs) - 1))
                wstratum[k] = si

            ham[k] = h

            cdist = cosine_distance(Z[i], Z[j])
            cosv[k] = (1.0 - cdist) if args.cosine_metric == "cossim" else cdist

        sp = spearmanr(ham, cosv)
        pr = pearsonr(ham, cosv)

        results[str(p)] = {
            "ham": ham,
            "cos": cosv,
            "mcount": mcount,
            "wstratum": wstratum,
            "spearman": sp,
            "pearson": pr,
        }

        print(f"[hap_vs_emb_pop] {p}: Spearman={sp:.4f}  Pearson={pr:.4f}  pairs={len(pairs)}")

        # ---- strat summary rows ----
        if w is None:
            # exact mismatch-count strata
            vals, counts = np.unique(mcount, return_counts=True)
            # keep those with enough pairs, then take top-k by frequency
            keep = counts >= int(args.min_pairs_per_mcount)
            vals, counts = vals[keep], counts[keep]
            if vals.size > 0:
                order = np.argsort(-counts)  # descending
                vals = vals[order][: int(args.top_k_mcount)]
                for mc in vals:
                    m = (mcount == mc)
                    strat_rows.append({
                        "population": str(p),
                        "stratum_type": "mismatch_count",
                        "mcount": int(mc),
                        "ham_mean": float(ham[m].mean()),
                        "n_pairs": int(m.sum()),
                        "cos_mean": float(cosv[m].mean()),
                        "cos_std": float(cosv[m].std(ddof=1)) if m.sum() > 1 else 0.0,
                        "cos_q25": float(np.quantile(cosv[m], 0.25)),
                        "cos_q50": float(np.quantile(cosv[m], 0.50)),
                        "cos_q75": float(np.quantile(cosv[m], 0.75)),
                    })
        else:
            # weighted bins (fallback)
            for si, lab in enumerate(wlabs):
                m = (wstratum == si)
                if int(m.sum()) == 0:
                    continue
                strat_rows.append({
                    "population": str(p),
                    "stratum_type": "weighted_bin",
                    "mcount": -1,
                    "ham_mean": float(ham[m].mean()),
                    "n_pairs": int(m.sum()),
                    "cos_mean": float(cosv[m].mean()),
                    "cos_std": float(cosv[m].std(ddof=1)) if m.sum() > 1 else 0.0,
                    "cos_q25": float(np.quantile(cosv[m], 0.25)),
                    "cos_q50": float(np.quantile(cosv[m], 0.50)),
                    "cos_q75": float(np.quantile(cosv[m], 0.75)),
                    "weighted_bin": lab,
                })

    # ---- save npz ----
    npz_kwargs = {
        "pops": np.array(list(results.keys()), dtype=object),
        "ham_metric": ham_name,
        "cos_metric": metric_name,
        "window_start": start,
        "window_len": wlen,
        "n_ind": hap.shape[0],
        "embed_dim": Z.shape[1],
        "seed": int(args.seed),
        "L": int(L),
    }
    for p, d in results.items():
        npz_kwargs[f"ham_{p}"] = d["ham"]
        npz_kwargs[f"cos_{p}"] = d["cos"]
        npz_kwargs[f"mcount_{p}"] = d["mcount"]
        np.savez_compressed(out_dir / "haplotype_similarity_vs_embedding_by_pop.npz", **npz_kwargs)

    # ---- existing plots ----
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

    plt.figure(figsize=(7, 6))
    for p, d in results.items():
        mids, mean, se, nbin = bin_means(d["ham"], d["cos"], bins=bins_trend)
        plt.errorbar(mids, mean, yerr=se, marker="o", linestyle="-", capsize=3, label=p)
    plt.xlabel(f"{ham_name} bin midpoint")
    plt.ylabel(metric_name)
    plt.title("Binned trend by population (within-pop pairs)")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "binned_hamming_vs_cosine_by_pop.png", dpi=200)
    plt.close()

    # ---- NEW plot: stratify by exact mismatch count (plain) ----
    pops = list(results.keys())
    if len(pops) > 0:
        fig, axes = plt.subplots(nrows=len(pops), ncols=1, figsize=(12, 3.2 * len(pops)), squeeze=False)
        for r, p in enumerate(pops):
            ax = axes[r, 0]
            d = results[p]

            if args.use_weighted_hamming:
                ax.text(0.5, 0.5, f"{p}: weighted hamming -> use binned strata (not exact mcount)",
                        ha="center", va="center")
                ax.set_axis_off()
                continue

            mcount = d["mcount"]
            cosv = d["cos"]

            vals, counts = np.unique(mcount, return_counts=True)
            keep = counts >= int(args.min_pairs_per_mcount)
            vals, counts = vals[keep], counts[keep]
            if vals.size == 0:
                ax.text(0.5, 0.5, f"{p}: no mcount strata with n >= {args.min_pairs_per_mcount}",
                        ha="center", va="center")
                ax.set_axis_off()
                continue

            order = np.argsort(-counts)
            vals = vals[order][: int(args.top_k_mcount)]

            data = [cosv[mcount == mc] for mc in vals]
            labels = [f"m={int(mc)}\n(n={(mcount==mc).sum()})" for mc in vals]

            ax.boxplot(data, showfliers=False)
            ax.set_title(f"{p}: {metric_name} stratified by exact mismatch count (top-K frequent)")
            ax.set_xlabel("mismatch count m (exact)")
            ax.set_ylabel(metric_name)
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=0)

        plt.tight_layout()
        plt.savefig(out_dir / "stratified_cosine_by_mcount_by_pop.png", dpi=200)
        plt.close(fig)

    # ---- write CSV summary ----
    if len(strat_rows) > 0:
        df = pd.DataFrame(strat_rows)
        df.to_csv(out_dir / "stratified_by_mcount_summary.csv", index=False)

    print(f"[hap_vs_emb_pop] wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
