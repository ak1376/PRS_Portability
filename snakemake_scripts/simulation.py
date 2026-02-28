#!/usr/bin/env python3
"""
Standalone simulator + cache (engine-aware: neutral with msprime, BGS with SLiM)

Generates one simulation (tree-sequence + SFS) for the chosen model and stores
artefacts under <simulation-dir>/<simulation-number>/.

Behavior:
- If config["engine"] == "msprime": neutral (no BGS), no coverage sampling.
- If config["engine"] == "slim":    BGS via stdpopsim/SLiM, coverage sampling enabled.

Requires: src/simulation.py providing:
  - simulation(sampled_params, model_type, experiment_config, sampled_coverage=None)
  - create_SFS(ts)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import demesdraw
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# project paths & local imports
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simulation import (  # noqa: E402
    simulation,   # engine-aware (msprime = neutral; slim = BGS)
    create_SFS,   # builds a moments.Spectrum from the ts,
    simulate_traits,
    calculate_fst,
    sample_params
)


# ------------------------------------------------------------------
# main workflow
# ------------------------------------------------------------------
def run_simulation(
    simulation_dir: Path,
    experiment_config: Path,
    model_type: str,
    simulation_number: Optional[str] = None,
    replicate: int = 0,
    output_dir: Optional[Path] = None,
):

    # Load config and inspect engine
    cfg: Dict[str, object] = json.loads(experiment_config.read_text())
    engine = cfg["engine"]  # "msprime" or "slim"

    sel_cfg = cfg.get("selection") or {}
    
    # decide destination folder name
    if output_dir is not None:
        out_dir = output_dir
        # If simulation_number is not provided but we need it for logic, we might need to infer or require it.
        # But here simulation_number is passed from CLI usually.
        if simulation_number is None:
             # fallback if needed, but usually provided
             simulation_number = "0000" 
    else:
        if simulation_number is None:
            existing = {int(p.name) for p in simulation_dir.glob("[0-9]*") if p.is_dir()}
            simulation_number = f"{max(existing, default=0) + 1:04d}"
        out_dir = simulation_dir / simulation_number
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique seed for this simulation
    base_seed = cfg.get("seed")
    if base_seed is not None:
        # Use simulation_number (grid index) + replicate to make a unique seed
        simulation_seed = int(base_seed) + int(simulation_number) * 1000 + int(replicate)
        print(f"• Using seed {simulation_seed} (base: {base_seed}, grid: {simulation_number}, rep: {replicate})")
        rng = np.random.default_rng(simulation_seed)
    else:
        simulation_seed = None
        rng = np.random.default_rng()
        print("• No seed specified, using random state")

    # Sample demographic params
    grid_cfg = cfg.get("grid_sampling", {})
    if grid_cfg.get("enabled", False):
        fixed = grid_cfg["fixed_params"]
        var_param = grid_cfg["varying_param"]

        min_val = float(grid_cfg["min_value"])
        max_val = float(grid_cfg["max_value"])
        scale = grid_cfg.get("scale", "linear")

        num_grid = int(grid_cfg.get("num_grid_points", 1))

        # simulation_number is now the grid index (0 .. num_grid-1)
        grid_idx = int(simulation_number)
        if grid_idx < 0 or grid_idx >= num_grid:
            raise ValueError(f"Simulation number {grid_idx} out of range [0, {num_grid}).")

        if scale == "linear":
            grid_vals = np.linspace(min_val, max_val, num_grid)
        elif scale == "log":
            grid_vals = np.geomspace(min_val, max_val, num_grid)
        else:
            raise ValueError(f"Unknown scale: {scale}")

        val = float(grid_vals[grid_idx])
        sampled_params = dict(fixed)
        sampled_params[var_param] = val

        print(f"• Grid mode: {var_param} = {val:.2f} "
              f"(grid index {grid_idx}/{num_grid-1}, replicate {replicate})")
    else:
        sampled_params = sample_params(cfg["priors"], rng=rng)


    if engine == "msprime":
        # Neutral path: NO BGS and NO coverage sampling
        sampled_coverage = None
        print("• engine=msprime → neutral (no BGS); skipping coverage sampling.")
    else:
        raise ValueError("engine must be 'slim' or 'msprime'.")

    # Run simulation via src/simulation.simulation(...)
    # Create modified config with the specific simulation seed
    sim_cfg = dict(cfg)
    if simulation_seed is not None:
        sim_cfg["seed"] = simulation_seed
    
    ts, g = simulation(sampled_params = sampled_params, model_type = model_type, experiment_config = sim_cfg)

    # Get both trait effect sizes and complete phenotype simulation (with population info)
    trait_df, phenotype_df = simulate_traits(ts, cfg)
    
    # Save effect sizes from sim_trait as DataFrame (preserves column names)
    trait_df.to_pickle(f"{out_dir}/effect_sizes.pkl")
    
    # Save phenotype data as DataFrame (includes population, genetic_value, environmental_noise, phenotype)
    phenotype_df.to_pickle(f"{out_dir}/phenotype.pkl")

    # Build SFS from result
    sfs = create_SFS(ts)
    
    # Calculate Fst
    fst_val = calculate_fst(ts)
    sampled_params["Fst"] = fst_val
    print(f"• Fst (YRI-CEU): {fst_val:.4f}")

    # Save artefacts
    (out_dir / "sampled_params.pkl").write_bytes(pickle.dumps(sampled_params))
    (out_dir / "SFS.pkl").write_bytes(pickle.dumps(sfs))
    ts_path = out_dir / "tree_sequence.trees"
    ts.dump(ts_path)

    # Demography plot (always your demes graph)
    fig_path = out_dir / "demes.png"
    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("N")
    ax.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # friendly path for log
    try:
        rel = out_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_dir
    print(f"✓ simulation written to {rel}")


# ------------------------------------------------------------------
# argparse entry-point
# ------------------------------------------------------------------
def main():
    cli = argparse.ArgumentParser(description="Generate one simulation (neutral or BGS, engine-aware)")
    cli.add_argument(
        "--simulation-dir",
        type=Path,
        required=True,
        help="Base directory that will hold <number>/ subfolders",
    )
    cli.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
        help="JSON config with priors, genome length (or real window), etc.",
    )
    cli.add_argument(
        "--model-type",
        required=True,
        choices=[
            "IM_symmetric"
        ],
        help="Which demographic model to simulate",
    )
    cli.add_argument(
        "--simulation-number",
        type=str,
        help="Folder name to create (e.g. '0005'). If omitted, the next free index is used.",
    )

    cli.add_argument(
        "--replicate",
        type=int,
        default=0,
        help="Replicate index for this parameter setting (used only for the seed).",
    )
    
    cli.add_argument(
        "--output-dir",
        type=Path,
        help="Exact output directory (overrides --simulation-dir nesting behavior).",
    )

    args = cli.parse_args()
    run_simulation(
        args.simulation_dir,
        args.experiment_config,
        args.model_type,
        args.simulation_number,
        args.replicate,
        args.output_dir,
    )

if __name__ == "__main__":
    main()
