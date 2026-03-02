#!/usr/bin/env python3
"""
snakemake_scripts/plot_vae_diagnostics.py

Thin wrapper calling src.vae.plotting.run_diagnostics(...)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.vae.plotting import PlotArgs, run_diagnostics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--resolved-hparams", type=Path, required=True)
    ap.add_argument("--genotype", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-step-points", type=int, default=5000)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    run_diagnostics(
        PlotArgs(
            logdir=args.logdir,
            checkpoint=args.checkpoint,
            resolved_hparams=args.resolved_hparams,
            genotype=args.genotype,
            meta=args.meta,
            outdir=args.outdir,
            batch_size=args.batch_size,
            max_step_points=args.max_step_points,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()