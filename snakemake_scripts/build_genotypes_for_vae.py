#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tskit


def compute_site_stats(ts: tskit.TreeSequence) -> dict:
    """
    Compute basic genotype / site statistics from a TreeSequence.
    """
    G_hap = ts.genotype_matrix()  # (sites, samples)

    num_sites_total = ts.num_sites
    num_inds = ts.num_individuals if ts.num_individuals > 0 else ts.num_samples

    segregating = (G_hap.max(axis=1) > 0)
    multiallelic = (G_hap.max(axis=1) > 1)

    stats = {
        "num_individuals": int(num_inds),
        "num_sites_total": int(num_sites_total),
        "num_segregating_sites": int(segregating.sum()),
        "num_multiallelic_sites": int(multiallelic.sum()),
        "num_biallelic_sites": int((segregating & ~multiallelic).sum()),
    }
    return stats


def _choose_contiguous_block(num_sites: int, subset_snps: int, subset_mode: str, seed: int = 0):
    """
    Return (start, end) indices for a contiguous SNP block.
    """
    if subset_snps is None or subset_snps >= num_sites:
        return 0, num_sites

    if subset_mode == "first":
        start = 0
    elif subset_mode == "middle":
        start = max((num_sites - subset_snps) // 2, 0)
    elif subset_mode == "random":
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, num_sites - subset_snps))
    else:
        raise ValueError(f"Unknown subset_mode: {subset_mode}")

    end = start + subset_snps
    return start, end


def _extract_haps_and_diploid(ts: tskit.TreeSequence):
    """
    Returns:
      - hap1: (num_valid_inds, num_biallelic_sites) float32 in {0,1}
      - hap2: (num_valid_inds, num_biallelic_sites) float32 in {0,1}
      - dip:  (num_valid_inds, num_biallelic_sites) float32 in {0,1,2}
      - kept_ind_ids: (num_valid_inds,) int64, the *tskit individual IDs* kept

    If ts has no individuals, treats each sample as "haplotype-individual":
      - hap1 is the sample matrix
      - hap2 is None
      - dip is same as hap1
      - kept_ind_ids is np.arange(num_samples)
    """
    G_hap = ts.genotype_matrix()  # (sites, samples), allele labels 0/1/2...

    # --- filter to biallelic sites (and drop missing) ---
    biallelic = (G_hap.max(axis=1) <= 1)
    if (G_hap.min(axis=1) < 0).any():
        biallelic &= (G_hap.min(axis=1) >= 0)
    G_hap = G_hap[biallelic, :]  # (biallelic_sites, samples)

    # If no individuals exist, we can't form diploids cleanly
    if ts.num_individuals == 0:
        hap1 = G_hap.T.astype(np.float32)  # (samples, sites)
        dip = hap1.copy()
        kept_ind_ids = np.arange(hap1.shape[0], dtype=np.int64)
        return hap1, None, dip, kept_ind_ids

    # Map sample node id -> column in G_hap
    samples = ts.samples()
    node_to_col = np.full(ts.num_nodes, -1, dtype=np.int32)
    node_to_col[samples] = np.arange(samples.size, dtype=np.int32)

    # For each individual, pick first 2 nodes (diploid assumption)
    inds = list(ts.individuals())
    nodes2 = np.full((len(inds), 2), -1, dtype=np.int32)
    kept_ind_ids_all = np.full((len(inds),), -1, dtype=np.int64)

    for i, ind in enumerate(inds):
        kept_ind_ids_all[i] = ind.id
        if len(ind.nodes) >= 2:
            nodes2[i, 0] = ind.nodes[0]
            nodes2[i, 1] = ind.nodes[1]

    cols2 = node_to_col[nodes2]  # (num_inds, 2)
    valid = (cols2[:, 0] >= 0) & (cols2[:, 1] >= 0)
    cols2 = cols2[valid]
    kept_ind_ids = kept_ind_ids_all[valid]

    # Extract haplotypes: (sites, inds) -> transpose to (inds, sites)
    hap1 = G_hap[:, cols2[:, 0]].T.astype(np.float32)
    hap2 = G_hap[:, cols2[:, 1]].T.astype(np.float32)
    dip = (hap1 + hap2).astype(np.float32)

    return hap1, hap2, dip, kept_ind_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True, help=".trees file with many CEU/YRI individuals")
    p.add_argument("--phenotype", required=True, help="phenotype.pkl with individual_id, population")
    p.add_argument("--outdir", required=True)

    p.add_argument("--subset-snps", type=int, default=5000,
                   help="Number of contiguous SNPs to keep (default: 5000).")
    p.add_argument("--subset-mode", choices=["first", "middle", "random"],
                   default="first",
                   help="Where to choose the contiguous block of SNPs from.")
    p.add_argument("--subset-seed", type=int, default=0,
                   help="Seed used only when subset-mode=random (default: 0).")

    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[build_genotypes_for_vae] Loading tree sequence from {args.tree}")
    ts = tskit.load(args.tree)

    # -------------------------------
    # Compute and report site stats
    # -------------------------------
    stats = compute_site_stats(ts)
    stats_txt = (
        f"Tree sequence: {args.tree}\n"
        f"Number of individuals: {stats['num_individuals']}\n"
        f"Total sites: {stats['num_sites_total']}\n"
        f"Segregating sites: {stats['num_segregating_sites']}\n"
        f"Multiallelic sites: {stats['num_multiallelic_sites']}\n"
        f"Biallelic segregating sites: {stats['num_biallelic_sites']}\n"
    )
    print("[build_genotypes_for_vae] Site / genotype summary:")
    print(stats_txt)

    stats_path = outdir / "genotype_site_stats.txt"
    stats_path.write_text(stats_txt)
    print(f"[build_genotypes_for_vae] Wrote genotype/site stats to {stats_path}")

    # -------------------------------
    # Build hap1/hap2 + diploid dosage
    # -------------------------------
    print("[build_genotypes_for_vae] Extracting haplotypes and diploid genotypes (biallelic only)")
    hap1, hap2, G, kept_ind_ids = _extract_haps_and_diploid(ts)

    num_inds, num_sites = G.shape
    print(f"[build_genotypes_for_vae] After biallelic filtering: {num_sites} sites retained")
    print(f"[build_genotypes_for_vae] Diploid genotype matrix shape: {G.shape}")
    if hap2 is None:
        print(f"[build_genotypes_for_vae] No individuals in ts; using haplotypes-as-individuals. hap1 shape={hap1.shape}")
    else:
        print(f"[build_genotypes_for_vae] hap1 shape={hap1.shape}, hap2 shape={hap2.shape}")

    # -------------------------------
    # Load phenotype/meta and align rows
    # -------------------------------
    print(f"[build_genotypes_for_vae] Loading phenotype/meta from {args.phenotype}")
    pheno = pd.read_pickle(args.phenotype)

    # Ensure sorted by individual_id
    if "individual_id" in pheno.columns:
        pheno = pheno.sort_values("individual_id").reset_index(drop=True)

    # If we filtered individuals (e.g. missing nodes), we need to subset phenotype too.
    # Assumption: pheno["individual_id"] matches tskit individual IDs.
    if ts.num_individuals > 0:
        if "individual_id" not in pheno.columns:
            raise ValueError("phenotype.pkl must have an 'individual_id' column when ts has individuals.")
        # subset and re-order phenotype to match kept_ind_ids
        pheno_indexed = pheno.set_index("individual_id", drop=False)
        missing = [i for i in kept_ind_ids.tolist() if i not in pheno_indexed.index]
        if len(missing) > 0:
            raise ValueError(
                f"Some kept tskit individual IDs are missing from phenotype.pkl: {missing[:10]} ..."
            )
        pheno = pheno_indexed.loc[kept_ind_ids].reset_index(drop=True)

    if G.shape[0] != len(pheno):
        raise ValueError(
            f"Genotype rows ({G.shape[0]}) and phenotype rows ({len(pheno)}) do not match "
            f"after alignment. Check individual_id mapping."
        )

    # -------------------------------
    # Contiguous SNP subsetting (apply to ALL matrices)
    # -------------------------------
    start, end = _choose_contiguous_block(num_sites, args.subset_snps, args.subset_mode, seed=args.subset_seed)

    if (start, end) != (0, num_sites):
        print(
            f"[build_genotypes_for_vae] Subsetting SNPs (contiguous, {args.subset_mode}): "
            f"using sites [{start}:{end}) out of {num_sites}"
        )
    else:
        print("[build_genotypes_for_vae] Not subsetting SNPs (using all sites)")

    G_subset = G[:, start:end]
    hap1_subset = hap1[:, start:end]
    hap2_subset = None if hap2 is None else hap2[:, start:end]
    snp_idx = np.arange(start, end, dtype=np.int64)

    # -------------------------------
    # Save outputs
    # -------------------------------
    geno_path = outdir / "all_individuals.npy"
    hap1_path = outdir / "hap1.npy"
    hap2_path = outdir / "hap2.npy"
    meta_path = outdir / "meta.pkl"
    snp_idx_path = outdir / "snp_index.npy"
    ts_ids_path = outdir / "ts_individual_ids.npy"

    print(f"[build_genotypes_for_vae] Saving diploid genotype matrix to {geno_path}")
    np.save(geno_path, G_subset.astype(np.float32))

    print(f"[build_genotypes_for_vae] Saving hap1 to {hap1_path}")
    np.save(hap1_path, hap1_subset.astype(np.float32))

    if hap2_subset is not None:
        print(f"[build_genotypes_for_vae] Saving hap2 to {hap2_path}")
        np.save(hap2_path, hap2_subset.astype(np.float32))
    else:
        # keep the file contract if your Snakefile expects it
        print(f"[build_genotypes_for_vae] No hap2 available; writing empty placeholder to {hap2_path}")
        np.save(hap2_path, np.zeros((hap1_subset.shape[0], hap1_subset.shape[1]), dtype=np.float32))

    print(f"[build_genotypes_for_vae] Saving SNP indices to {snp_idx_path}")
    np.save(snp_idx_path, snp_idx)

    print(f"[build_genotypes_for_vae] Saving kept tskit individual IDs to {ts_ids_path}")
    np.save(ts_ids_path, kept_ind_ids.astype(np.int64))

    # Keep useful columns for plotting
    meta = pheno[["individual_id", "population"]].copy()
    print(f"[build_genotypes_for_vae] Saving meta to {meta_path}")
    meta.to_pickle(meta_path)

    print("[build_genotypes_for_vae] Done.")


if __name__ == "__main__":
    main()
