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


def _individual_genotype_matrix(ts: tskit.TreeSequence) -> np.ndarray:
    """
    Return (num_individuals, num_biallelic_sites) diploid genotype matrix with entries 0/1/2,
    filtering multiallelic sites explicitly. Vectorized (no per-individual loop).

    If no individuals are defined, returns (num_samples, num_biallelic_sites) haplotypes-as-individuals.
    """

    # Haploid genotype matrix: (sites, samples). Entries are allele labels (0,1,2,...).
    G_hap = ts.genotype_matrix()

    # ---- Filter multiallelic sites explicitly ----
    # Keep only sites where allele labels are in {0,1} for all samples (i.e., max <= 1).
    # Also drop sites with missing genotypes (-1), if any.
    biallelic = (G_hap.max(axis=1) <= 1)
    if (G_hap.min(axis=1) < 0).any():  # missing data safety
        biallelic &= (G_hap.min(axis=1) >= 0)

    G_hap = G_hap[biallelic, :]  # (biallelic_sites, samples)

    # If no individuals are defined, treat each haplotype sample as its own "individual"
    if ts.num_individuals == 0:
        return G_hap.T.astype(np.float32)  # (samples, sites)

    # ---- Vectorize diploid construction ----
    samples = ts.samples()  # node IDs for columns of G_hap
    # Build node_id -> column_index map (O(num_nodes) memory, O(1) lookup)
    node_to_col = np.full(ts.num_nodes, -1, dtype=np.int32)
    node_to_col[samples] = np.arange(samples.size, dtype=np.int32)

    # Collect each individual's first two nodes (diploid assumption)
    inds = list(ts.individuals())
    nodes2 = np.full((len(inds), 2), -1, dtype=np.int32)
    for i, ind in enumerate(inds):
        if len(ind.nodes) >= 2:
            nodes2[i, 0] = ind.nodes[0]
            nodes2[i, 1] = ind.nodes[1]

    cols2 = node_to_col[nodes2]  # (num_inds, 2) column indices into G_hap

    # Optional: keep only individuals that have two *sample* nodes (mapped cols >= 0)
    valid_inds = (cols2[:, 0] >= 0) & (cols2[:, 1] >= 0)
    cols2 = cols2[valid_inds]

    # Diploid dosage: (sites, num_valid_inds) then transpose to (inds, sites)
    G_ind = (G_hap[:, cols2[:, 0]] + G_hap[:, cols2[:, 1]]).T.astype(np.float32)

    return G_ind


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True, help=".trees file with many CEU/YRI individuals")
    p.add_argument("--phenotype", required=True, help="phenotype.pkl with individual_id, population")
    p.add_argument("--outdir", required=True)
    # optional: allow command-line control of subset size
    p.add_argument("--subset-snps", type=int, default=5000,
                   help="Number of contiguous SNPs to keep for VAE (default: 5000).")
    p.add_argument("--subset-mode", choices=["first", "middle", "random"],
                   default="first",
                   help="Where to choose the contiguous block of SNPs from.")
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
    with open(stats_path, "w") as f:
        f.write(stats_txt)

    print(f"[build_genotypes_for_vae] Wrote genotype/site stats to {stats_path}")




    print("[build_genotypes_for_vae] Building individual genotype matrix")
    G = _individual_genotype_matrix(ts)  # (num_inds, num_sites)
    print(
    f"[build_genotypes_for_vae] After biallelic filtering: "
    f"{G.shape[1]} sites retained"
    )
    num_inds, num_sites = G.shape
    print(f"[build_genotypes_for_vae] Genotype matrix shape: {G.shape}")

    print(f"[build_genotypes_for_vae] Loading phenotype/meta from {args.phenotype}")
    pheno = pd.read_pickle(args.phenotype)

    # Ensure sorted by individual_id (this matches how you built phenotype_df)
    if "individual_id" in pheno.columns:
        pheno = pheno.sort_values("individual_id").reset_index(drop=True)

    if G.shape[0] != len(pheno):
        raise ValueError(
            f"Genotype rows ({G.shape[0]}) and phenotype rows ({len(pheno)}) "
            "do not match. Check that individual_id ordering is consistent."
        )

    # -------------------------------
    # Contiguous SNP subsetting here
    # -------------------------------
    subset_snps = args.subset_snps
    subset_mode = args.subset_mode

    if subset_snps is not None and subset_snps < num_sites:
        if subset_mode == "first":
            start = 0
        elif subset_mode == "middle":
            start = max((num_sites - subset_snps) // 2, 0)
        elif subset_mode == "random":
            rng = np.random.default_rng(0)  # fixed seed for reproducibility
            start = int(rng.integers(0, num_sites - subset_snps))
        else:
            raise ValueError(f"Unknown subset_mode: {subset_mode}")

        end = start + subset_snps
        print(
            f"[build_genotypes_for_vae] Subsetting SNPs (contiguous, {subset_mode}): "
            f"using sites [{start}:{end}) out of {num_sites}"
        )

        G_subset = G[:, start:end]
        snp_idx = np.arange(start, end, dtype=int)
    else:
        print("[build_genotypes_for_vae] Not subsetting SNPs (using all sites)")
        G_subset = G
        snp_idx = np.arange(num_sites, dtype=int)

    # -------------------------------
    # Save outputs
    # -------------------------------
    geno_path = outdir / "all_individuals.npy"
    meta_path = outdir / "meta.pkl"
    snp_idx_path = outdir / "snp_index.npy"

    print(f"[build_genotypes_for_vae] Saving genotype matrix to {geno_path}")
    np.save(geno_path, G_subset.astype(np.float32))

    # Save which SNPs we kept (useful later if you care about positions)
    print(f"[build_genotypes_for_vae] Saving SNP indices to {snp_idx_path}")
    np.save(snp_idx_path, snp_idx.astype(np.int64))

    # Keep just the useful columns for VAE / plotting
    meta = pheno[["individual_id", "population"]].copy()
    print(f"[build_genotypes_for_vae] Saving meta to {meta_path}")
    meta.to_pickle(meta_path)

    print("[build_genotypes_for_vae] Done.")


if __name__ == "__main__":
    main()
