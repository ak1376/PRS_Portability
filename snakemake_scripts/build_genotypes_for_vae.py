#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.build_genotypes_for_vae import BuildGenotypesArgs, build_genotypes_for_vae


def _str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {x!r}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Snakemake wrapper: build genotype arrays + discovery train/val + target split (optional normalization)."
    )

    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--phenotype", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)

    # Optional YAML config (maf_threshold under data.maf_threshold)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config with maf_threshold under data.maf_threshold",
    )
    ap.add_argument("--maf-threshold", type=float, default=None)

    # NEW: experiment config JSON (reads top-level key: discovery)
    ap.add_argument(
        "--experiment-config-json",
        type=Path,
        default=None,
        help="Optional experiment config JSON; reads top-level key 'discovery' unless --discovery-pop is provided.",
    )

    # subsetting
    ap.add_argument("--subset-snps", type=int, default=5000)
    ap.add_argument("--subset-bp", type=float, default=None)
    ap.add_argument("--subset-mode", type=str, default="first", choices=["first", "middle", "random"])
    ap.add_argument("--subset-seed", type=int, default=0)

    # split (discovery -> train/val; target = non-discovery)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument(
        "--discovery-pop",
        type=str,
        default=None,
        help="Override discovery population label. If omitted, read from --experiment-config-json; else default CEU.",
    )

    # NEW: normalization
    ap.add_argument("--normalize", type=_str2bool, default=True)
    ap.add_argument(
        "--norm-mode",
        type=str,
        default="zscore_snp",
        choices=["none", "zscore_snp", "af_residual", "divide_by_2"],
    )
    ap.add_argument("--norm-eps", type=float, default=1e-6)
    ap.add_argument("--norm-clip-std-min", type=float, default=1e-3)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    a = BuildGenotypesArgs(
        tree=args.tree,
        phenotype=args.phenotype,
        outdir=args.outdir,
        config=args.config,
        maf_threshold=args.maf_threshold,
        experiment_config_json=args.experiment_config_json,
        subset_snps=int(args.subset_snps),
        subset_bp=args.subset_bp,
        subset_mode=str(args.subset_mode),
        subset_seed=int(args.subset_seed),
        val_frac=float(args.val_frac),
        split_seed=int(args.split_seed),
        discovery_pop=args.discovery_pop,
        normalize=bool(args.normalize),
        norm_mode=str(args.norm_mode),
        norm_eps=float(args.norm_eps),
        norm_clip_std_min=float(args.norm_clip_std_min),
    )

    summary = build_genotypes_for_vae(a)
    print("[build_genotypes_for_vae wrapper] summary:", summary)


if __name__ == "__main__":
    main()