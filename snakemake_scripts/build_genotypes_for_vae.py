#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tskit


def _individual_genotype_matrix(ts: tskit.TreeSequence) -> np.ndarray:
    """
    Return (num_individuals, num_sites) diploid genotype matrix with entries 0/1/2.
    If no individuals are defined, fall back to haplotypes-as-individuals.
    """
    G_hap = ts.genotype_matrix()  # (sites, samples)

    if ts.num_individuals == 0:
        return G_hap.T.astype(np.float32)

    num_inds = ts.num_individuals
    num_sites = ts.num_sites
    G_ind = np.zeros((num_inds, num_sites), dtype=np.float32)

    for i, ind in enumerate(ts.individuals()):
        nodes = ind.nodes
        if len(nodes) > 0:
            G_ind[i] = G_hap[:, nodes].sum(axis=1)

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

    print("[build_genotypes_for_vae] Building individual genotype matrix")
    G = _individual_genotype_matrix(ts)  # (num_inds, num_sites)
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
