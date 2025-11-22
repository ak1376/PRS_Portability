#!/usr/bin/env python3
"""
Wrapper script to run GWAS using src/gwas.py.
Called by Snakemake.
"""

import argparse
import sys
from pathlib import Path

# ------------------------------------------------------------------
# project paths & local imports
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gwas import run_gwas

def main():
    parser = argparse.ArgumentParser(description="Run naive GWAS on genotype and phenotype data")
    parser.add_argument("--genotype", type=Path, required=True, help="Path to genotype file (.trees or .npy)")
    parser.add_argument("--phenotype", type=Path, required=True, help="Path to phenotype file (.pkl or .csv)")
    parser.add_argument("--trait", type=Path, required=True, help="Path to trait info file (.pkl or .csv)")
    parser.add_argument("--output-prefix", type=str, required=True, help="Prefix for output files (including directory)")
    parser.add_argument("--discovery-pop", type=str, default=None, help="Run GWAS only on a specified population (e.g. 'EUR')")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_gwas(
        genotype_path=args.genotype,
        phenotype_path=args.phenotype,
        trait_path=args.trait,
        output_prefix=args.output_prefix,
        discovery_pop=args.discovery_pop
    )

if __name__ == "__main__":
    main()
