#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.build_genotypes_for_vae import BuildGenotypesArgs, build_genotypes_for_vae


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Snakemake wrapper: build genotype arrays + train/val split for VAE."
    )

    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--phenotype", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)

    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config with maf_threshold under data.maf_threshold",
    )
    ap.add_argument("--maf-threshold", type=float, default=None)

    # subsetting
    ap.add_argument("--subset-snps", type=int, default=5000)
    ap.add_argument("--subset-bp", type=float, default=None)
    ap.add_argument("--subset-mode", type=str, default="first", choices=["first", "middle", "random"])
    ap.add_argument("--subset-seed", type=int, default=0)

    # split
    ap.add_argument(
        "--split-mode",
        type=str,
        default="within_pop",
        choices=["random", "within_pop", "discovery_only", "cross_pop"],
    )
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--discovery-pop", type=str, default="CEU")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    a = BuildGenotypesArgs(
        tree=args.tree,
        phenotype=args.phenotype,
        outdir=args.outdir,
        config=args.config,
        maf_threshold=args.maf_threshold,
        subset_snps=int(args.subset_snps),
        subset_bp=args.subset_bp,
        subset_mode=str(args.subset_mode),
        subset_seed=int(args.subset_seed),
        split_mode=str(args.split_mode),
        val_frac=float(args.val_frac),
        split_seed=int(args.split_seed),
        discovery_pop=str(args.discovery_pop),
    )

    summary = build_genotypes_for_vae(a)
    print("[build_genotypes_for_vae wrapper] summary:", summary)


if __name__ == "__main__":
    main()