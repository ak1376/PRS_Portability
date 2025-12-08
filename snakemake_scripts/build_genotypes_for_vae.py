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
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[build_genotypes_for_vae] Loading tree sequence from {args.tree}")
    ts = tskit.load(args.tree)

    print("[build_genotypes_for_vae] Building individual genotype matrix")
    G = _individual_genotype_matrix(ts)  # (num_inds, num_sites)

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

    geno_path = outdir / "all_individuals.npy"
    meta_path = outdir / "meta.pkl"

    print(f"[build_genotypes_for_vae] Saving genotype matrix to {geno_path}")
    np.save(geno_path, G.astype(np.float32))

    # Keep just the useful columns for VAE / plotting
    meta = pheno[["individual_id", "population"]].copy()
    print(f"[build_genotypes_for_vae] Saving meta to {meta_path}")
    meta.to_pickle(meta_path)

    print("[build_genotypes_for_vae] Done.")


if __name__ == "__main__":
    main()
