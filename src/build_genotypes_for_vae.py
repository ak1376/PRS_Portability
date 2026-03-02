# src/build_genotypes_for_vae.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import tskit
import yaml


# -------------------------
# Public API
# -------------------------

@dataclass(frozen=True)
class BuildGenotypesArgs:
    tree: Path
    phenotype: Path
    outdir: Path

    # optional config file (used only to find maf_threshold under cfg["data"]["maf_threshold"])
    config: Optional[Path] = None
    maf_threshold: Optional[float] = None

    # SNP subsetting
    subset_snps: int = 5000
    subset_bp: Optional[float] = None
    subset_mode: str = "first"   # "first" | "middle" | "random"
    subset_seed: int = 0

    # Train/val split
    split_mode: str = "cross_pop"  # "random" | "within_pop" | "discovery_only" | "cross_pop"
    val_frac: float = 0.2
    split_seed: int = 0
    discovery_pop: str = "CEU"


def build_genotypes_for_vae(a: BuildGenotypesArgs) -> Dict[str, Any]:
    """
    Heavy-hitter entrypoint: loads TS + phenotype, filters sites, subsets window,
    aligns meta, creates train/val split, and writes outputs to a.outdir.

    Outputs written in outdir:
      - all_individuals.npy (N,L) float32 in {0,1,2}
      - hap1.npy, hap2.npy (N,L) float32 in {0,1}
      - meta.pkl (aligned to genotype rows)
      - hap_meta.pkl
      - snp_index.npy (indices in filtered-site space)
      - ts_individual_ids.npy (tskit individual IDs aligned to rows)
      - variant_positions_bp.npy, variant_site_ids.npy
      - genotype_site_stats.txt, site_filter_report.txt
      - train_idx.npy, val_idx.npy
      - train.npy, val.npy  (convenience slices of all_individuals)
    """
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    maf_from_yaml = _load_maf_from_config(a.config)
    maf_threshold = (
        a.maf_threshold
        if a.maf_threshold is not None
        else (maf_from_yaml if maf_from_yaml is not None else 0.0)
    )

    print(f"[build_genotypes_for_vae] maf_threshold={maf_threshold} (monomorphic always removed)")
    print(f"[build_genotypes_for_vae] loading ts: {a.tree}")
    ts = tskit.load(str(a.tree))

    # Basic TS site stats (raw)
    stats = compute_site_stats(ts)
    stats_txt = (
        f"Tree sequence: {a.tree}\n"
        f"Number of individuals_or_samples: {stats['num_individuals_or_samples']}\n"
        f"Total sites: {stats['num_sites_total']}\n"
        f"Segregating sites: {stats['num_segregating_sites']}\n"
        f"Multiallelic sites: {stats['num_multiallelic_sites']}\n"
        f"Biallelic segregating sites: {stats['num_biallelic_sites']}\n"
    )
    (outdir / "genotype_site_stats.txt").write_text(stats_txt)

    # Extract/filter to biallelic + maf
    hap1, hap2, G, kept_ind_ids, filt, kept_positions_bp, kept_site_ids = _extract_haps_and_diploid(
        ts, maf_threshold=float(maf_threshold)
    )
    (outdir / "site_filter_report.txt").write_text(
        "\n".join([f"{k}: {v}" for k, v in filt.items()]) + "\n"
    )

    # Load + align phenotype/meta
    pheno = pd.read_pickle(a.phenotype)
    pheno = _align_pheno_to_kept_inds(ts, pheno, kept_ind_ids, G.shape[0])

    # Validate phenotype columns we rely on
    for col in ["individual_id", "population", "phenotype"]:
        if col not in pheno.columns:
            raise ValueError(
                f"phenotype.pkl must include column '{col}'. Columns found: {list(pheno.columns)}"
            )

    # Subset window (AFTER site filtering)
    num_sites = G.shape[1]
    start, end = _choose_subset_indices(
        kept_positions_bp=kept_positions_bp,
        num_sites=num_sites,
        subset_snps=a.subset_snps,
        subset_bp=a.subset_bp,
        subset_mode=a.subset_mode,
        subset_seed=a.subset_seed,
    )

    if end <= start:
        raise RuntimeError(f"Subset window invalid: start={start}, end={end}, num_sites={num_sites}")

    G_subset = G[:, start:end]
    hap1_subset = hap1[:, start:end]
    hap2_subset = None if hap2 is None else hap2[:, start:end]
    kept_positions_subset = kept_positions_bp[start:end]
    kept_site_ids_subset = kept_site_ids[start:end]
    snp_idx = np.arange(start, end, dtype=np.int64)

    # Save core arrays
    np.save(outdir / "all_individuals.npy", G_subset.astype(np.float32))
    np.save(outdir / "hap1.npy", hap1_subset.astype(np.float32))
    if hap2_subset is not None:
        np.save(outdir / "hap2.npy", hap2_subset.astype(np.float32))
    else:
        # predictable artifact even in haploid mode
        np.save(outdir / "hap2.npy", np.zeros_like(hap1_subset, dtype=np.float32))

    np.save(outdir / "snp_index.npy", snp_idx)
    np.save(outdir / "variant_positions_bp.npy", kept_positions_subset.astype(np.float64))
    np.save(outdir / "variant_site_ids.npy", kept_site_ids_subset.astype(np.int32))
    np.save(outdir / "ts_individual_ids.npy", kept_ind_ids.astype(np.int64))

    # Meta aligned to genotype rows
    meta = pheno[["individual_id", "population", "phenotype"]].copy()
    meta.to_pickle(outdir / "meta.pkl")

    hap_meta = _build_hap_meta(meta, has_hap2=True)
    hap_meta.to_pickle(outdir / "hap_meta.pkl")

    # -------------------------
    # Train/val split + convenience arrays
    # -------------------------
    train_idx, val_idx = _make_train_val_split(
        meta=meta,
        split_mode=a.split_mode,
        val_frac=float(a.val_frac),
        seed=int(a.split_seed),
        discovery_pop=str(a.discovery_pop),
    )

    # enforce non-empty splits (fallback to random split if needed)
    if train_idx.size == 0 or val_idx.size == 0:
        print("[build_genotypes_for_vae] WARNING: empty split detected; falling back to random split.")
        train_idx, val_idx = _make_train_val_split(
            meta=meta,
            split_mode="random",
            val_frac=float(a.val_frac),
            seed=int(a.split_seed),
            discovery_pop=str(a.discovery_pop),
        )

    np.save(outdir / "train_idx.npy", train_idx.astype(np.int64))
    np.save(outdir / "val_idx.npy", val_idx.astype(np.int64))

    np.save(outdir / "train.npy", G_subset[train_idx].astype(np.float32))
    np.save(outdir / "val.npy",   G_subset[val_idx].astype(np.float32))

    summary = {
        "outdir": str(outdir),
        "maf_threshold": float(maf_threshold),
        "subset_start": int(start),
        "subset_end": int(end),
        "num_inds": int(G_subset.shape[0]),
        "num_snps": int(G_subset.shape[1]),
        "split_mode": str(a.split_mode),
        "val_frac": float(a.val_frac),
        "split_seed": int(a.split_seed),
        "discovery_pop": str(a.discovery_pop),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "pop_counts": meta["population"].astype(str).value_counts().to_dict(),
    }
    return summary


# -------------------------
# Small helpers
# -------------------------

def _load_maf_from_config(config_path: Optional[Path]) -> Optional[float]:
    if config_path is None:
        return None
    cfg = yaml.safe_load(Path(config_path).read_text())
    data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    maf = data.get("maf_threshold", None)
    return None if maf is None else float(maf)


def _choose_subset_indices(
    *,
    kept_positions_bp: np.ndarray,
    num_sites: int,
    subset_snps: int,
    subset_bp: Optional[float],
    subset_mode: str,
    subset_seed: int,
) -> Tuple[int, int]:
    if subset_bp is not None:
        start, end = _choose_bp_window(
            kept_positions_bp, subset_bp=float(subset_bp), subset_mode=subset_mode, seed=subset_seed
        )
    else:
        start, end = _choose_contiguous_block(num_sites, subset_snps, subset_mode, seed=subset_seed)
    return start, end


def _align_pheno_to_kept_inds(
    ts: tskit.TreeSequence,
    pheno: pd.DataFrame,
    kept_ind_ids: np.ndarray,
    n_rows_expected: int,
) -> pd.DataFrame:
    # Sort by individual_id if present (just for determinism)
    if "individual_id" in pheno.columns:
        pheno = pheno.sort_values("individual_id").reset_index(drop=True)

    if ts.num_individuals > 0:
        if "individual_id" not in pheno.columns:
            raise ValueError("phenotype.pkl must have an 'individual_id' column when ts has individuals.")
        pheno_indexed = pheno.set_index("individual_id", drop=False)
        missing = [i for i in kept_ind_ids.tolist() if i not in pheno_indexed.index]
        if missing:
            raise ValueError(f"Some kept tskit individual IDs missing from phenotype.pkl: {missing[:10]} ...")
        pheno = pheno_indexed.loc[kept_ind_ids].reset_index(drop=True)

    if n_rows_expected != len(pheno):
        raise ValueError(
            f"Genotype rows ({n_rows_expected}) and phenotype rows ({len(pheno)}) do not match after alignment."
        )
    return pheno


def _build_hap_meta(meta: pd.DataFrame, has_hap2: bool) -> pd.DataFrame:
    if not has_hap2:
        hm = meta.assign(hap_id=0).copy()
        hm["hap_index"] = np.arange(len(hm))
        return hm
    hm = pd.concat([meta.assign(hap_id=0), meta.assign(hap_id=1)], ignore_index=True)
    hm["hap_index"] = np.arange(len(hm))
    return hm


def _make_train_val_split(
    *,
    meta: pd.DataFrame,
    split_mode: str,
    val_frac: float,
    seed: int,
    discovery_pop: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns train_idx, val_idx indices into rows of meta / G_subset.

    Modes:
      - random: split across all individuals
      - within_pop: stratified split within each population (keeps pop proportions)
      - discovery_only: split only within discovery_pop, BUT indices are still into full array
        (i.e. training+val are subsets of the full individuals, containing only discovery pop)
      - cross_pop: train = discovery_pop, val = everyone else (ignores val_frac)
    """
    if split_mode not in {"random", "within_pop", "discovery_only", "cross_pop"}:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    if split_mode != "cross_pop":
        if not (0.0 < val_frac < 1.0):
            raise ValueError(f"val_frac must be in (0,1). Got {val_frac}")

    n = len(meta)
    if n < 2:
        raise ValueError("Need at least 2 individuals to split.")

    rng = np.random.default_rng(seed)
    pops = meta["population"].astype(str).to_numpy()
    all_idx = np.arange(n, dtype=np.int64)

    def split_indices(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(idx, dtype=np.int64).copy()
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * idx.size)))
        if idx.size - n_val < 1:
            n_val = idx.size - 1
        val = np.sort(idx[:n_val])
        train = np.sort(idx[n_val:])
        return train, val

    if split_mode == "random":
        return split_indices(all_idx)

    if split_mode == "within_pop":
        train_parts: List[np.ndarray] = []
        val_parts: List[np.ndarray] = []
        for u in np.unique(pops):
            idx_u = all_idx[pops == u]
            if idx_u.size < 2:
                # can't split this pop; push to train and rely on other pops for val
                train_parts.append(idx_u)
                continue
            tr, va = split_indices(idx_u)
            train_parts.append(tr)
            val_parts.append(va)

        train = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=np.int64)
        val = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=np.int64)

        # If we failed to produce both sides, fallback random
        if train.size == 0 or val.size == 0:
            return split_indices(all_idx)
        return train, val

    if split_mode == "discovery_only":
        idx_d = all_idx[pops == discovery_pop]
        if idx_d.size < 2:
            raise ValueError(f"discovery_only requires >=2 individuals in {discovery_pop}. Got {idx_d.size}.")
        return split_indices(idx_d)

    if split_mode == "cross_pop":
        idx_train = np.sort(all_idx[pops == discovery_pop])
        idx_val = np.sort(all_idx[pops != discovery_pop])
        if idx_train.size < 1 or idx_val.size < 1:
            raise ValueError(
                f"cross_pop requires >=1 in discovery ({discovery_pop}) and >=1 in other. "
                f"Got train={idx_train.size}, val={idx_val.size}."
            )
        # Shuffle within each to avoid any ordering artifacts
        idx_train = idx_train.copy(); rng.shuffle(idx_train)
        idx_val = idx_val.copy(); rng.shuffle(idx_val)
        return np.sort(idx_train), np.sort(idx_val)

    raise RuntimeError("unreachable")


def compute_site_stats(ts: tskit.TreeSequence) -> dict:
    """
    Basic genotype / site statistics from a TreeSequence.
    """
    G_hap = ts.genotype_matrix()  # (sites, samples)
    num_sites_total = ts.num_sites
    num_inds = ts.num_individuals if ts.num_individuals > 0 else ts.num_samples

    segregating = (G_hap.max(axis=1) > 0)
    multiallelic = (G_hap.max(axis=1) > 1)

    return {
        "num_individuals_or_samples": int(num_inds),
        "num_sites_total": int(num_sites_total),
        "num_segregating_sites": int(segregating.sum()),
        "num_multiallelic_sites": int(multiallelic.sum()),
        "num_biallelic_sites": int((segregating & ~multiallelic).sum()),
    }


def _choose_contiguous_block(num_sites: int, subset_snps: int, subset_mode: str, seed: int = 0) -> Tuple[int, int]:
    if subset_snps is None or subset_snps >= num_sites:
        return 0, num_sites

    if subset_mode == "first":
        start = 0
    elif subset_mode == "middle":
        start = max((num_sites - subset_snps) // 2, 0)
    elif subset_mode == "random":
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, num_sites - subset_snps + 1))
    else:
        raise ValueError(f"Unknown subset_mode: {subset_mode}")

    end = start + subset_snps
    return start, end


def _choose_bp_window(
    positions_bp: np.ndarray,
    subset_bp: float,
    subset_mode: str,
    seed: int = 0,
) -> Tuple[int, int]:
    if positions_bp.size == 0:
        return 0, 0
    if subset_bp is None or subset_bp <= 0:
        return 0, positions_bp.size

    pos = positions_bp.astype(np.float64)
    max_start_idx = np.searchsorted(pos, pos[-1] - subset_bp, side="right") - 1
    max_start_idx = int(max(0, min(max_start_idx, pos.size - 1)))

    if subset_mode == "first":
        start_idx = 0
    elif subset_mode == "middle":
        mid_bp = 0.5 * (pos[0] + pos[-1])
        start_bp = mid_bp - 0.5 * subset_bp
        start_idx = int(np.searchsorted(pos, start_bp, side="left"))
        start_idx = int(max(0, min(start_idx, max_start_idx)))
    elif subset_mode == "random":
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, max_start_idx + 1)) if max_start_idx > 0 else 0
    else:
        raise ValueError(f"Unknown subset_mode: {subset_mode}")

    end_bp = pos[start_idx] + subset_bp
    end_idx = int(np.searchsorted(pos, end_bp, side="right"))

    if end_idx <= start_idx and pos.size > 0:
        end_idx = min(start_idx + 1, pos.size)

    return start_idx, end_idx


def _maf_filter_mask_from_haps(
    G_hap_biallelic: np.ndarray,
    maf_threshold: float,
) -> Tuple[np.ndarray, dict]:
    """
    Removes monomorphic sites and (optionally) sites with MAF < maf_threshold.
    G_hap_biallelic: (sites, samples) with alleles in {0,1}, no missing.
    """
    if G_hap_biallelic.size == 0:
        keep = np.zeros((0,), dtype=bool)
        return keep, {
            "num_sites_in": 0,
            "num_haplotypes": 0,
            "maf_threshold": float(maf_threshold) if maf_threshold is not None else None,
            "min_allele_count_implied": None,
            "num_monomorphic_removed": 0,
            "num_maf_removed": 0,
            "num_sites_out": 0,
        }

    n_haps = int(G_hap_biallelic.shape[1])
    p = G_hap_biallelic.mean(axis=1)
    maf = np.minimum(p, 1.0 - p)

    keep = maf > 0.0
    num_mono_removed = int((maf == 0.0).sum())

    num_maf_removed = 0
    min_ac = None
    if maf_threshold is not None and maf_threshold > 0.0:
        min_ac = int(math.ceil(maf_threshold * n_haps))
        before = int(keep.sum())
        keep &= (maf >= maf_threshold)
        after = int(keep.sum())
        num_maf_removed = before - after

    info = {
        "num_sites_in": int(G_hap_biallelic.shape[0]),
        "num_haplotypes": n_haps,
        "maf_threshold": float(maf_threshold) if maf_threshold is not None else None,
        "min_allele_count_implied": min_ac,
        "num_monomorphic_removed": num_mono_removed,
        "num_maf_removed": int(num_maf_removed),
        "num_sites_out": int(keep.sum()),
    }
    return keep, info


def _extract_haps_and_diploid(
    ts: tskit.TreeSequence,
    maf_threshold: float,
):
    """
    Returns:
      - hap1: (num_valid_inds, num_kept_sites) float32 in {0,1}
      - hap2: (num_valid_inds, num_kept_sites) float32 in {0,1} or None
      - dip:  (num_valid_inds, num_kept_sites) float32 in {0,1,2}
      - kept_ind_ids: (num_valid_inds,) int64, the *tskit individual IDs* kept
      - filter_report: dict with biallelic/monomorphic/maf filtering counts
      - kept_positions_bp: (num_kept_sites,) float64, site positions in bp
      - kept_site_ids: (num_kept_sites,) int32, tskit site IDs
    """
    G_hap = ts.genotype_matrix()  # (sites, samples)
    filter_report: Dict[str, object] = {}

    site_positions_bp_all = ts.tables.sites.position.astype(np.float64)
    site_ids_all = np.arange(ts.num_sites, dtype=np.int32)

    # biallelic + non-missing
    biallelic = (G_hap.max(axis=1) <= 1)
    if (G_hap.min(axis=1) < 0).any():
        biallelic &= (G_hap.min(axis=1) >= 0)

    filter_report["num_sites_raw"] = int(G_hap.shape[0])
    filter_report["num_sites_after_biallelic_nonmissing"] = int(biallelic.sum())

    G_hap = G_hap[biallelic, :]
    site_positions_bp = site_positions_bp_all[biallelic]
    site_ids = site_ids_all[biallelic]

    keep_maf, maf_info = _maf_filter_mask_from_haps(G_hap.astype(np.float32), maf_threshold=float(maf_threshold))
    filter_report.update(maf_info)

    G_hap = G_hap[keep_maf, :]
    site_positions_bp = site_positions_bp[keep_maf]
    site_ids = site_ids[keep_maf]

    # If TS has no individuals, treat samples as haploid rows
    if ts.num_individuals == 0:
        hap1 = G_hap.T.astype(np.float32)
        dip = hap1.copy()
        kept_ind_ids = np.arange(hap1.shape[0], dtype=np.int64)
        return hap1, None, dip, kept_ind_ids, filter_report, site_positions_bp, site_ids

    # Map nodes -> sample columns
    samples = ts.samples()
    node_to_col = np.full(ts.num_nodes, -1, dtype=np.int32)
    node_to_col[samples] = np.arange(samples.size, dtype=np.int32)

    inds = list(ts.individuals())
    nodes2 = np.full((len(inds), 2), -1, dtype=np.int32)
    kept_ind_ids_all = np.full((len(inds),), -1, dtype=np.int64)

    non_diploid_ids: List[int] = []
    for i, ind in enumerate(inds):
        kept_ind_ids_all[i] = ind.id
        if len(ind.nodes) != 2:
            non_diploid_ids.append(ind.id)
            continue
        nodes2[i, 0] = ind.nodes[0]
        nodes2[i, 1] = ind.nodes[1]

    if non_diploid_ids:
        raise ValueError(
            "Found individuals that do not have exactly 2 nodes (diploid requirement). "
            f"First few IDs: {non_diploid_ids[:10]}"
        )

    cols2 = node_to_col[nodes2]
    valid = (cols2[:, 0] >= 0) & (cols2[:, 1] >= 0)
    cols2 = cols2[valid]
    kept_ind_ids = kept_ind_ids_all[valid]

    hap1 = G_hap[:, cols2[:, 0]].T.astype(np.float32)
    hap2 = G_hap[:, cols2[:, 1]].T.astype(np.float32)
    dip = (hap1 + hap2).astype(np.float32)

    return hap1, hap2, dip, kept_ind_ids, filter_report, site_positions_bp, site_ids