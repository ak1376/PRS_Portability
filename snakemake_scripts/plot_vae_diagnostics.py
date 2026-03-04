#!/usr/bin/env python3
"""
snakemake_scripts/plot_vae_diagnostics.py

Thin CLI wrapper that directly calls src.vae.plotting.run(...)
(no subprocess, no second Python process).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.vae.plotting import Args, run


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--resolved-hparams", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)

    ap.add_argument("--train-genotype", type=Path, required=True)
    ap.add_argument("--val-genotype", type=Path, required=True)
    ap.add_argument("--target-genotype", type=Path, default=None)

    ap.add_argument("--recon-dir", type=Path, default=None, help="directory with {split}_recon.npz files")

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # NOTE: your Snakefile passes this; if you want to keep it, accept it here
    ap.add_argument("--max-step-points", type=int, default=5000)

    a = ap.parse_args()

    # If your src/vae/plotting.py doesn't use max_step_points anymore,
    # we just ignore it (but accepting the flag avoids argparse errors).
    run(
        Args(
            logdir=a.logdir,
            checkpoint=a.checkpoint,
            resolved_hparams=a.resolved_hparams,
            outdir=a.outdir,
            train_genotype=a.train_genotype,
            val_genotype=a.val_genotype,
            target_genotype=a.target_genotype,
            batch_size=a.batch_size,
            device=a.device,
            recon_dir=a.recon_dir,
        )
    )


if __name__ == "__main__":
    main()