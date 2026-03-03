#!/usr/bin/env python3
"""
snakemake_scripts/plot_vae_diagnostics.py

Thin wrapper calling src.vae.plotting.run_diagnostics(...)

Supports:
  New interface:
    --train-genotype, --val-genotype, --target-genotype, [--meta]
  Legacy interface:
    --genotype, --meta
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
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-step-points", type=int, default=5000)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # ---- New preferred inputs ----
    ap.add_argument("--train-genotype", type=Path, default=None, help="discovery_train.npy")
    ap.add_argument("--val-genotype", type=Path, default=None, help="discovery_val.npy")
    ap.add_argument("--target-genotype", type=Path, default=None, help="target.npy (e.g., YRI)")

    # ---- Legacy inputs (backwards compatible) ----
    ap.add_argument("--genotype", type=Path, default=None, help="legacy: full genotype matrix (N,L)")

    args = ap.parse_args()

    # Decide which mode we are in:
    have_new = (args.train_genotype is not None) or (args.val_genotype is not None) or (args.target_genotype is not None)

    if have_new:
        # Require all three for split-mode diagnostics
        missing = [k for k, v in {
            "--train-genotype": args.train_genotype,
            "--val-genotype": args.val_genotype,
            "--target-genotype": args.target_genotype,
        }.items() if v is None]
        if missing:
            raise SystemExit(f"Missing required args for split-mode diagnostics: {', '.join(missing)}")

        run_diagnostics(
            PlotArgs(
                logdir=args.logdir,
                checkpoint=args.checkpoint,
                resolved_hparams=args.resolved_hparams,
                outdir=args.outdir,
                batch_size=args.batch_size,
                max_step_points=args.max_step_points,
                device=args.device,
                train_genotype=args.train_genotype,
                val_genotype=args.val_genotype,
                target_genotype=args.target_genotype,
            )
        )
        return

    # Legacy mode
    if args.genotype is None:
        raise SystemExit("Legacy mode requires --genotype (or use split-mode args).")

    run_diagnostics(
        PlotArgs(
            logdir=args.logdir,
            checkpoint=args.checkpoint,
            resolved_hparams=args.resolved_hparams,
            outdir=args.outdir,
            batch_size=args.batch_size,
            max_step_points=args.max_step_points,
            device=args.device,
            genotype=args.genotype,
        )
    )


if __name__ == "__main__":
    main()