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
    calculate_fst
)

# ------------------------------------------------------------------
# parameter sampling helpers
# ------------------------------------------------------------------
def sample_params(
    priors: Dict[str, List[float]], *, rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    rng = rng or np.random.default_rng()
    params = {k: float(rng.uniform(*bounds)) for k, bounds in priors.items()}
    # keep bottleneck start > end if both are present
    if {"t_bottleneck_start", "t_bottleneck_end"}.issubset(params) and params[
        "t_bottleneck_start"
    ] <= params["t_bottleneck_end"]:
        params["t_bottleneck_start"], params["t_bottleneck_end"] = (
            params["t_bottleneck_end"],
            params["t_bottleneck_start"],
        )
    return params


def sample_coverage_percent(
    selection_cfg: Dict[str, List[float]], *, rng: Optional[np.random.Generator] = None
) -> float:
    """
    Sample a *percent* (e.g., 37.4) from selection_cfg["coverage_percent"] = [low, high].
    Only used when engine == "slim".
    """
    rng = rng or np.random.default_rng()
    low, high = selection_cfg["coverage_percent"]
    return float(rng.uniform(low, high))


# ------------------------------------------------------------------
# main workflow
# ------------------------------------------------------------------
def run_simulation(
    simulation_dir: Path,
    experiment_config: Path,
    model_type: str,
    simulation_number: Optional[str] = None,
):
    # Load config and inspect engine
    cfg: Dict[str, object] = json.loads(experiment_config.read_text())
    engine = cfg["engine"]  # "msprime" or "slim"

    sel_cfg = cfg.get("selection") or {}
    
    # decide destination folder name
    if simulation_number is None:
        existing = {int(p.name) for p in simulation_dir.glob("[0-9]*") if p.is_dir()}
        simulation_number = f"{max(existing, default=0) + 1:04d}"
    out_dir = simulation_dir / simulation_number
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique seed for this simulation
    base_seed = cfg.get("seed")
    if base_seed is not None:
        # Create a unique seed for each simulation using base_seed + simulation_number
        simulation_seed = int(base_seed) + int(simulation_number)
        print(f"• Using seed {simulation_seed} (base: {base_seed} + sim: {simulation_number})")
        rng = np.random.default_rng(simulation_seed)
    else:
        simulation_seed = None
        rng = np.random.default_rng()
        print("• No seed specified, using random state")

    # Sample demographic params
    grid_cfg = cfg.get("grid_sampling", {})
    if grid_cfg.get("enabled", False):
        # Grid mode
        fixed = grid_cfg["fixed_params"]
        var_param = grid_cfg["varying_param"]
        
        min_val = float(grid_cfg["min_value"])
        max_val = float(grid_cfg["max_value"])
        scale = grid_cfg.get("scale", "linear")
        
        # We need to know the total number of draws to generate the grid
        total_draws = int(cfg["num_draws"])
        
        idx = int(simulation_number)
        if idx >= total_draws:
             raise ValueError(f"Simulation number {idx} exceeds num_draws {total_draws}")

        if scale == "linear":
            if total_draws == 1:
                 val = min_val
            else:
                 val = np.linspace(min_val, max_val, total_draws)[idx]
        elif scale == "log":
             if total_draws == 1:
                 val = min_val
             else:
                 val = np.geomspace(min_val, max_val, total_draws)[idx]
        else:
             raise ValueError(f"Unknown scale: {scale}")
             
        sampled_params = dict(fixed)
        sampled_params[var_param] = float(val)
        print(f"• Grid mode: {var_param} = {sampled_params[var_param]} (Index {idx}/{total_draws})")
    else:
        sampled_params = sample_params(cfg["priors"], rng=rng)

    # Decide coverage based on engine
    if engine == "slim":
        if not sel_cfg.get("enabled", False):
            raise ValueError("engine='slim' requires selection.enabled=true in your config.")
        if "coverage_percent" not in sel_cfg:
            raise ValueError("engine='slim' requires selection.coverage_percent=[low, high].")
        sampled_coverage = sample_coverage_percent(sel_cfg, rng=rng)  # percent, e.g. 37.4
        print(f"• engine=slim → sampled coverage: {sampled_coverage:.2f}%")
    elif engine == "msprime":
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
    
    ts, g = simulation(sampled_params, model_type, sim_cfg, sampled_coverage)

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

    # Metadata sidecar
    sel_summary = getattr(ts, "_bgs_selection_summary", {}) or {}

    # Only include BGS / selection knobs if we actually ran SLiM
    is_bgs = (engine == "slim")
    meta = dict(
        engine=str(engine),
        model_type=str(model_type),
        # Neutral vs BGS:
        selection=is_bgs,
        # Species/DFE only meaningful for SLiM runs
        species=(str(sel_cfg.get("species", "HomSap")) if is_bgs else None),
        dfe_id=(str(sel_cfg.get("dfe_id", "Gamma_K17")) if is_bgs else None),
        # Real-window keys (None if unused or neutral)
        chromosome=(sel_cfg.get("chromosome") if is_bgs else None),
        left=(sel_cfg.get("left") if is_bgs else None),
        right=(sel_cfg.get("right") if is_bgs else None),
        genetic_map=(sel_cfg.get("genetic_map") if is_bgs else None),
        # Synthetic-contig parameters (always in cfg)
        genome_length=float(cfg.get("genome_length")),
        mutation_rate=float(cfg.get("mutation_rate")),
        recombination_rate=float(cfg.get("recombination_rate")),
        # BGS tiling / coverage knobs from config (only if SLiM)
        coverage_fraction=(
            None if not is_bgs else
            (None if sel_cfg.get("coverage_fraction") is None else float(sel_cfg["coverage_fraction"]))
        ),
        coverage_percent=(
            None if not is_bgs else
            (None if sel_cfg.get("coverage_percent") is None else
             [float(sel_cfg["coverage_percent"][0]), float(sel_cfg["coverage_percent"][1])])
        ),
        exon_bp=(int(sel_cfg.get("exon_bp", 200)) if is_bgs else None),
        jitter_bp=(int(sel_cfg.get("jitter_bp", 0)) if is_bgs else None),
        tile_bp=(int(sel_cfg["tile_bp"]) if is_bgs and sel_cfg.get("tile_bp") is not None else None),
        # Realized selection span (after interval building; zeroes for neutral)
        selected_bp=(int(sel_summary.get("selected_bp", 0)) if is_bgs else 0),
        selected_frac=(float(sel_summary.get("selected_frac", 0.0)) if is_bgs else 0.0),
        # Persist the actual sampled coverage (percent) only for SLiM
        sampled_coverage_percent=(float(sampled_coverage) if is_bgs and sampled_coverage is not None else None),
        # Also a fraction version (helpful for window scripts)
        target_coverage_frac=(
            (float(sampled_coverage) / 100.0) if is_bgs and sampled_coverage is not None and float(sampled_coverage) > 1.0
            else (float(sampled_coverage) if is_bgs and sampled_coverage is not None else None)
        ),
        # SLiM options
        slim_scaling=(float(sel_cfg.get("slim_scaling", 10.0)) if is_bgs else None),
        slim_burn_in=(float(sel_cfg.get("slim_burn_in", 5.0)) if is_bgs else None),
        # misc
        num_samples={k: int(v) for k, v in (cfg.get("num_samples") or {}).items()},
        base_seed=(None if cfg.get("seed") is None else int(cfg.get("seed"))),
        simulation_seed=simulation_seed,
        sequence_length=float(ts.sequence_length),
        tree_sequence=str(ts_path),
        # sampled priors
        sampled_params={k: float(v) for k, v in sampled_params.items()},
    )

    (out_dir / "bgs.meta.json").write_text(json.dumps(meta, indent=2))

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
            "bottleneck",
            "split_isolation",
            "split_migration",
            "drosophila_three_epoch",
        ],
        help="Which demographic model to simulate",
    )
    cli.add_argument(
        "--simulation-number",
        type=str,
        help="Folder name to create (e.g. '0005'). If omitted, the next free index is used.",
    )
    args = cli.parse_args()
    run_simulation(
        args.simulation_dir,
        args.experiment_config,
        args.model_type,
        args.simulation_number,
    )


if __name__ == "__main__":
    main()
