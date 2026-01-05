#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tskit
import yaml


def compute_site_stats(ts: tskit.TreeSequence) -> dict:
    """
    Compute basic genotype / site statistics from a TreeSequence.
    Note: these are based on raw genotype_matrix() (before biallelic/MAF filters).
    """
    G_hap = ts.genotype_matrix()  # (sites, samples), where samples = haploid sample nodes

    num_sites_total = ts.num_sites
    # NOTE: ts.num_samples is haploid sample nodes, not diploid individuals
    num_inds = ts.num_individuals if ts.num_individuals > 0 else ts.num_samples

    segregating = (G_hap.max(axis=1) > 0)  # any non-0 allele seen
    multiallelic = (G_hap.max(axis=1) > 1)

    stats = {
        "num_individuals_or_samples": int(num_inds),
        "num_sites_total": int(num_sites_total),
        "num_segregating_sites": int(segregating.sum()),
        "num_multiallelic_sites": int(multiallelic.sum()),
        "num_biallelic_sites": int((segregating & ~multiallelic).sum()),
    }
    return stats


def _choose_contiguous_block(num_sites: int, subset_snps: int, subset_mode: str, seed: int = 0) -> tuple[int, int]:
    """
    Choose a contiguous block in *variant index space* (after filtering).
    Returns (start, end) indices suitable for slicing [:, start:end].
    """
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
) -> tuple[int, int]:
    """
    Choose a contiguous window in *bp space* using the filtered variant positions.

    Returns (start_idx, end_idx) indices over variants such that variants in
    positions_bp[start_idx:end_idx] fall within a span of ~subset_bp bp.
    """
    if positions_bp.size == 0:
        return 0, 0
    if subset_bp is None or subset_bp <= 0:
        return 0, positions_bp.size

    pos = positions_bp.astype(np.float64)
    # Determine which start indices can produce a window of length subset_bp
    max_start_idx = np.searchsorted(pos, pos[-1] - subset_bp, side="right") - 1
    max_start_idx = int(max(0, min(max_start_idx, pos.size - 1)))

    if subset_mode == "first":
        start_idx = 0
    elif subset_mode == "middle":
        # pick start so that window is centered in bp space
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

    # Guarantee at least one SNP if possible
    if end_idx <= start_idx and pos.size > 0:
        end_idx = min(start_idx + 1, pos.size)

    return start_idx, end_idx


def _maf_filter_mask_from_haps(
    G_hap_biallelic: np.ndarray,
    maf_threshold: float,
) -> tuple[np.ndarray, dict]:
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

    p = G_hap_biallelic.mean(axis=1)  # allele-1 frequency across hap samples
    maf = np.minimum(p, 1.0 - p)

    # Always remove monomorphic
    keep = maf > 0.0
    num_mono_removed = int((maf == 0.0).sum())

    # Optional MAF threshold beyond monomorphic
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
    # Genotypes: (sites, samples), allele labels 0/1/2...
    G_hap = ts.genotype_matrix()

    filter_report: dict[str, object] = {}

    # Site metadata arrays (aligned with first axis of genotype_matrix())
    # Sites are ordered by position in tskit.
    site_positions_bp_all = ts.tables.sites.position.astype(np.float64)  # (num_sites,)
    site_ids_all = np.arange(ts.num_sites, dtype=np.int32)              # (num_sites,)

    # --- filter to biallelic sites (and drop missing) ---
    biallelic = (G_hap.max(axis=1) <= 1)
    if (G_hap.min(axis=1) < 0).any():
        biallelic &= (G_hap.min(axis=1) >= 0)

    filter_report["num_sites_raw"] = int(G_hap.shape[0])
    filter_report["num_sites_after_biallelic_nonmissing"] = int(biallelic.sum())

    # apply biallelic mask to genotypes and to site metadata
    G_hap = G_hap[biallelic, :]
    site_positions_bp = site_positions_bp_all[biallelic]
    site_ids = site_ids_all[biallelic]

    # Now G_hap should be only {0,1} and no missing; apply monomorphic + optional MAF
    keep_maf, maf_info = _maf_filter_mask_from_haps(G_hap.astype(np.float32), maf_threshold=float(maf_threshold))
    filter_report.update(maf_info)

    G_hap = G_hap[keep_maf, :]
    site_positions_bp = site_positions_bp[keep_maf]
    site_ids = site_ids[keep_maf]

    # If no individuals exist, we can't form diploids cleanly
    if ts.num_individuals == 0:
        hap1 = G_hap.T.astype(np.float32)  # (samples, sites)
        dip = hap1.copy()
        kept_ind_ids = np.arange(hap1.shape[0], dtype=np.int64)
        return hap1, None, dip, kept_ind_ids, filter_report, site_positions_bp, site_ids

    # Map sample node id -> column in G_hap
    samples = ts.samples()
    node_to_col = np.full(ts.num_nodes, -1, dtype=np.int32)
    node_to_col[samples] = np.arange(samples.size, dtype=np.int32)

    # For each individual, require exactly 2 nodes (diploid assumption)
    inds = list(ts.individuals())
    nodes2 = np.full((len(inds), 2), -1, dtype=np.int32)
    kept_ind_ids_all = np.full((len(inds),), -1, dtype=np.int64)

    non_diploid_ids: list[int] = []
    for i, ind in enumerate(inds):
        kept_ind_ids_all[i] = ind.id
        if len(ind.nodes) != 2:
            non_diploid_ids.append(ind.id)
            continue
        nodes2[i, 0] = ind.nodes[0]
        nodes2[i, 1] = ind.nodes[1]

    if len(non_diploid_ids) > 0:
        # Hard fail: better than silently taking the first two nodes
        raise ValueError(
            "Found individuals that do not have exactly 2 nodes (diploid requirement). "
            f"First few IDs: {non_diploid_ids[:10]}"
        )

    cols2 = node_to_col[nodes2]  # (num_inds, 2)
    valid = (cols2[:, 0] >= 0) & (cols2[:, 1] >= 0)
    cols2 = cols2[valid]
    kept_ind_ids = kept_ind_ids_all[valid]

    # Extract haplotypes: (sites, inds) -> transpose to (inds, sites)
    hap1 = G_hap[:, cols2[:, 0]].T.astype(np.float32)
    hap2 = G_hap[:, cols2[:, 1]].T.astype(np.float32)
    dip = (hap1 + hap2).astype(np.float32)

    return hap1, hap2, dip, kept_ind_ids, filter_report, site_positions_bp, site_ids


def _load_maf_from_config(config_path: str | None) -> float | None:
    if config_path is None:
        return None
    cfg = yaml.safe_load(Path(config_path).read_text())
    data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    maf = data.get("maf_threshold", None)
    if maf is None:
        return None
    return float(maf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True, help=".trees file with many CEU/YRI individuals")
    p.add_argument("--phenotype", required=True, help="phenotype.pkl with individual_id, population")
    p.add_argument("--outdir", required=True)

    # Optional YAML config
    p.add_argument("--config", default=None, help="YAML config with data.maf_threshold (optional)")

    # Optional CLI override (wins over YAML if provided)
    p.add_argument(
        "--maf-threshold",
        type=float,
        default=None,
        help=(
            "Minor allele frequency threshold. 0 disables MAF filtering, but monomorphic sites are always removed. "
            "If not set, tries to read from --config (data.maf_threshold)."
        ),
    )

    # Subsetting option A: by SNP count (index space)
    p.add_argument("--subset-snps", type=int, default=5000, help="Number of SNPs to keep (index-based).")
    # Subsetting option B: by bp span (position space)
    p.add_argument(
        "--subset-bp",
        type=float,
        default=None,
        help="If set, subset by bp span using filtered variant positions (overrides --subset-snps). Example: 50000.",
    )

    p.add_argument(
        "--subset-mode",
        choices=["first", "middle", "random"],
        default="first",
        help="Where to choose the contiguous window from (first/middle/random).",
    )
    p.add_argument("--subset-seed", type=int, default=0, help="Seed used only when subset-mode=random.")

    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve maf_threshold: CLI > YAML > default(0.0)
    maf_from_yaml = _load_maf_from_config(args.config)
    maf_threshold = (
        args.maf_threshold
        if args.maf_threshold is not None
        else (maf_from_yaml if maf_from_yaml is not None else 0.0)
    )

    print(f"[build_genotypes_for_vae] Using maf_threshold={maf_threshold} (monomorphic sites always removed)")
    print(f"[build_genotypes_for_vae] Loading tree sequence from {args.tree}")
    ts = tskit.load(args.tree)

    # -------------------------------
    # Compute and report site stats
    # -------------------------------
    stats = compute_site_stats(ts)
    stats_txt = (
        f"Tree sequence: {args.tree}\n"
        f"Number of individuals_or_samples: {stats['num_individuals_or_samples']}\n"
        f"Total sites: {stats['num_sites_total']}\n"
        f"Segregating sites: {stats['num_segregating_sites']}\n"
        f"Multiallelic sites: {stats['num_multiallelic_sites']}\n"
        f"Biallelic segregating sites: {stats['num_biallelic_sites']}\n"
    )
    print("[build_genotypes_for_vae] Site / genotype summary (raw TS):")
    print(stats_txt)

    stats_path = outdir / "genotype_site_stats.txt"
    stats_path.write_text(stats_txt)
    print(f"[build_genotypes_for_vae] Wrote genotype/site stats to {stats_path}")

    # -------------------------------
    # Build hap1/hap2 + diploid dosage
    # -------------------------------
    print("[build_genotypes_for_vae] Extracting haplotypes and diploid genotypes (biallelic + monomorphic removed + optional MAF)")
    hap1, hap2, G, kept_ind_ids, filt, kept_positions_bp, kept_site_ids = _extract_haps_and_diploid(
        ts, maf_threshold=maf_threshold
    )

    num_inds, num_sites = G.shape
    print("[build_genotypes_for_vae] Site filtering report:")
    print(f"  raw sites: {filt.get('num_sites_raw')}")
    print(f"  after biallelic+nonmissing: {filt.get('num_sites_after_biallelic_nonmissing')}")
    print(f"  haplotypes used for MAF: {filt.get('num_haplotypes')}")
    print(f"  maf_threshold: {filt.get('maf_threshold')}")
    print(f"  min allele count implied: {filt.get('min_allele_count_implied')}")
    print(f"  monomorphic removed: {filt.get('num_monomorphic_removed')}")
    print(f"  maf removed (<{maf_threshold}): {filt.get('num_maf_removed')}")
    print(f"  final kept sites: {filt.get('num_sites_out')}")

    print(f"[build_genotypes_for_vae] Diploid genotype matrix shape: {G.shape}")
    if hap2 is None:
        print(f"[build_genotypes_for_vae] No individuals in ts; using haplotypes-as-individuals. hap1 shape={hap1.shape}")
    else:
        print(f"[build_genotypes_for_vae] hap1 shape={hap1.shape}, hap2 shape={hap2.shape}")

    (outdir / "site_filter_report.txt").write_text("\n".join([f"{k}: {v}" for k, v in filt.items()]) + "\n")

    # -------------------------------
    # Load phenotype/meta and align rows
    # -------------------------------
    print(f"[build_genotypes_for_vae] Loading phenotype/meta from {args.phenotype}")
    pheno = pd.read_pickle(args.phenotype)

    if "individual_id" in pheno.columns:
        pheno = pheno.sort_values("individual_id").reset_index(drop=True)

    if ts.num_individuals > 0:
        if "individual_id" not in pheno.columns:
            raise ValueError("phenotype.pkl must have an 'individual_id' column when ts has individuals.")
        pheno_indexed = pheno.set_index("individual_id", drop=False)

        missing = [i for i in kept_ind_ids.tolist() if i not in pheno_indexed.index]
        if len(missing) > 0:
            raise ValueError(f"Some kept tskit individual IDs are missing from phenotype.pkl: {missing[:10]} ...")

        pheno = pheno_indexed.loc[kept_ind_ids].reset_index(drop=True)

    if G.shape[0] != len(pheno):
        raise ValueError(
            f"Genotype rows ({G.shape[0]}) and phenotype rows ({len(pheno)}) do not match "
            f"after alignment. Check individual_id mapping."
        )

    # -------------------------------
    # Subsetting window (apply to ALL matrices + positions/site IDs)
    # -------------------------------
    if args.subset_bp is not None:
        start, end = _choose_bp_window(
            kept_positions_bp, subset_bp=float(args.subset_bp), subset_mode=args.subset_mode, seed=args.subset_seed
        )
        print(
            f"[build_genotypes_for_vae] Subsetting by bp span ({args.subset_mode}): "
            f"sites [{start}:{end}) out of {num_sites} (bp span ~{args.subset_bp})"
        )
    else:
        start, end = _choose_contiguous_block(num_sites, args.subset_snps, args.subset_mode, seed=args.subset_seed)
        if (start, end) != (0, num_sites):
            print(
                f"[build_genotypes_for_vae] Subsetting SNPs (index-contiguous, {args.subset_mode}): "
                f"sites [{start}:{end}) out of {num_sites}"
            )
        else:
            print("[build_genotypes_for_vae] Not subsetting SNPs (using all sites)")

    G_subset = G[:, start:end]
    hap1_subset = hap1[:, start:end]
    hap2_subset = None if hap2 is None else hap2[:, start:end]
    kept_positions_subset = kept_positions_bp[start:end]
    kept_site_ids_subset = kept_site_ids[start:end]

    # `snp_idx` is now explicitly "indices within filtered matrix"
    snp_idx = np.arange(start, end, dtype=np.int64)

    # -------------------------------
    # Save outputs
    # -------------------------------
    geno_path = outdir / "all_individuals.npy"
    hap1_path = outdir / "hap1.npy"
    hap2_path = outdir / "hap2.npy"  # will only be written if hap2 exists
    meta_path = outdir / "meta.pkl"
    snp_idx_path = outdir / "snp_index.npy"
    ts_ids_path = outdir / "ts_individual_ids.npy"
    hap_meta_path = outdir / "hap_meta.pkl"

    positions_path = outdir / "variant_positions_bp.npy"
    site_ids_path = outdir / "variant_site_ids.npy"

    print(f"[build_genotypes_for_vae] Saving diploid genotype matrix to {geno_path}")
    np.save(geno_path, G_subset.astype(np.float32))

    print(f"[build_genotypes_for_vae] Saving hap1 to {hap1_path}")
    np.save(hap1_path, hap1_subset.astype(np.float32))

    if hap2_subset is not None:
        print(f"[build_genotypes_for_vae] Saving hap2 to {hap2_path}")
        np.save(hap2_path, hap2_subset.astype(np.float32))
    else:
        print("[build_genotypes_for_vae] No hap2 available; not writing hap2.npy")

    print(f"[build_genotypes_for_vae] Saving SNP indices to {snp_idx_path}")
    np.save(snp_idx_path, snp_idx)

    print(f"[build_genotypes_for_vae] Saving variant positions (bp) to {positions_path}")
    np.save(positions_path, kept_positions_subset.astype(np.float64))

    print(f"[build_genotypes_for_vae] Saving variant site IDs to {site_ids_path}")
    np.save(site_ids_path, kept_site_ids_subset.astype(np.int32))

    print(f"[build_genotypes_for_vae] Saving kept tskit individual IDs to {ts_ids_path}")
    np.save(ts_ids_path, kept_ind_ids.astype(np.int64))

    meta = pheno[["individual_id", "population"]].copy()
    print(f"[build_genotypes_for_vae] Saving meta to {meta_path}")
    meta.to_pickle(meta_path)

    # -------------------------------
    # Store population labels for each haplotype
    # -------------------------------
    if hap2_subset is None:
        hap_meta = meta.assign(hap_id=0).copy()
        hap_meta["hap_index"] = np.arange(len(hap_meta))
    else:
        hap_meta = pd.concat(
            [
                meta.assign(hap_id=0),
                meta.assign(hap_id=1),
            ],
            ignore_index=True,
        )
        hap_meta["hap_index"] = np.arange(len(hap_meta))

    print(f"[build_genotypes_for_vae] Saving haplotype meta to {hap_meta_path}")
    hap_meta.to_pickle(hap_meta_path)

    print("[build_genotypes_for_vae] Done.")


if __name__ == "__main__":
    main()
