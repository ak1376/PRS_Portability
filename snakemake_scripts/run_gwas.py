#!/usr/bin/env python3
"""
snakemake_scripts/run_gwas.py
Thin wrapper around src/gwas.py

This version supports:
  --genotype            .npy (num_inds, num_snps) dosage {0,1,2}
  --phenotype           meta.pkl produced by build_inputs (must include phenotype + population)
  --trait               effect_sizes.pkl (must include a site_id column of tskit site IDs)
  --variant-site-ids    .npy mapping from SNP index -> tskit site ID (produced by build_inputs)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gwas import run_gwas  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run naive GWAS on processed genotype matrix")
    p.add_argument("--genotype", type=Path, required=True, help="Path to genotype .npy (N x L)")
    p.add_argument("--phenotype", type=Path, required=True, help="Path to meta.pkl with phenotype/population")
    p.add_argument("--trait", type=Path, required=True, help="Path to effect_sizes.pkl with causal site IDs")
    p.add_argument(
        "--variant-site-ids",
        type=Path,
        required=True,
        help="Path to variant_site_ids.npy (length L) mapping SNP index -> tskit site id",
    )
    p.add_argument("--output-prefix", type=str, required=True, help="Prefix for output files")
    p.add_argument("--discovery-pop", type=str, default=None, help="Discovery population label (e.g. CEU)")
    args = p.parse_args()

    outdir = Path(args.output_prefix).parent
    outdir.mkdir(parents=True, exist_ok=True)

    run_gwas(
        genotype_path=args.genotype,
        phenotype_path=args.phenotype,
        trait_path=args.trait,
        variant_site_ids_path=args.variant_site_ids,
        output_prefix=args.output_prefix,
        discovery_pop=args.discovery_pop,
    )


if __name__ == "__main__":
    main()