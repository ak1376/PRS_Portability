from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any

import demes
import numpy as np
import stdpopsim as sps
import tskit
import tstrait
import moments
import math
import sys
from pathlib import Path
import msprime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demes_models import IM_symmetric_model

# ──────────────────────────────────
# Minimal helpers
# ──────────────────────────────────

def sample_params(
    priors: Dict[str, List[float]], *, rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    rng = rng or np.random.default_rng()
    params: Dict[str, float] = {}

    for k, bounds in priors.items():
        params[k] = float(rng.uniform(*bounds))

    return params


def build_demes_graph(
    model_type: str, sampled_params: Dict[str, float]
) -> demes.Graph:
    """
    Build a Demes graph for the given model_type + sampled_params.
    Mirrors the logic inside simulation(...), but returns only the graph.
    """
    if model_type == "IM_symmetric":
        return IM_symmetric_model(sampled_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    

def simulation_runner(
    g: demes.Graph, experiment_config: Dict[str, Any]
) -> Tuple[tskit.TreeSequence, demes.Graph]:
    
    species = experiment_config.get("species", "HomSap")
    sp = sps.get_species(species)

    # Direct instantiation — no subclass needed
    model = sps.DemographicModel(
        id="model",
        description="model",
        long_description="model",
        model=msprime.Demography.from_demes(g),
        generation_time=1,
    )

    genome_length = experiment_config["genome_length"]
    mutation_rate = experiment_config["mutation_rate"]
    recombination_rate = experiment_config['recombination_rate']

    contig = sp.get_contig(
          chromosome=None,
          length=genome_length,
          mutation_rate=mutation_rate,
          recombination_rate=recombination_rate,
      )
    
    print(f'• Using engine: {experiment_config.get("engine")}')
    print("contig.length:", contig.length)
    print("contig.mutation_rate:", contig.mutation_rate)
    print("contig.recombination_map.mean_rate:", contig.recombination_map.mean_rate)

    samples = {
        k: int(v) for k, v in (experiment_config.get("num_samples") or {}).items()
    }
    seed = experiment_config.get("seed", None)

    eng = sps.get_engine("msprime")

    ts = eng.simulate(
        model,
        contig,
        samples,
        seed=seed
    )

    return ts, g


def simulation(
    sampled_params: Dict[str, float],
    model_type: str,
    experiment_config: Dict[str, Any],
) -> Tuple[tskit.TreeSequence, demes.Graph]:
    # Build demes graph

    g = build_demes_graph(model_type=model_type, sampled_params=sampled_params)

    return simulation_runner(
        g, experiment_config
    )



def _individual_genotype_matrix(ts: tskit.TreeSequence) -> np.ndarray:
    """
    Return an (num_individuals, num_sites) genotype matrix with diploid genotypes.
    """
    G_hap = ts.genotype_matrix()  # (sites, samples/nodes)
    if ts.num_individuals == 0:
        return G_hap.T  # treat haplotypes as individuals

    num_inds = ts.num_individuals
    num_sites = ts.num_sites
    G_ind = np.zeros((num_inds, num_sites), dtype=np.float32)

    for i, ind in enumerate(ts.individuals()):
        nodes = ind.nodes
        if len(nodes) > 0:
            G_ind[i] = G_hap[:, nodes].sum(axis=1)  # sum over haplotypes

    return G_ind

def create_SFS(ts: tskit.TreeSequence) -> moments.Spectrum:
    """Build a moments.Spectrum using pops that have sampled individuals."""
    sample_sets: List[np.ndarray] = []
    pop_ids: List[str] = []
    for pop in ts.populations():
        samps = ts.samples(population=pop.id)
        if len(samps):
            sample_sets.append(samps)
            meta = pop.metadata if isinstance(pop.metadata, dict) else {}
            pop_ids.append(meta.get("name", f"pop{pop.id}"))
    if not sample_sets:
        raise ValueError("No sampled populations found.")
    arr = ts.allele_frequency_spectrum(
        sample_sets=sample_sets, mode="site", polarised=True, span_normalise=False
    )
    sfs = moments.Spectrum(arr)
    sfs.pop_ids = pop_ids
    return sfs

def simulate_traits(ts: tskit.TreeSequence, experiment_config: dict) -> Tuple:
    """
    Simulate a quantitative trait. Two modes:

    1) Shared architecture (default):
       - One set of causal SNPs for all individuals (current behavior).
    2) Population-specific architecture:
       - discovery_pop and target_pop share only a fraction of causal SNPs,
         controlled by config["causal_architecture"]["overlap_fraction"].

    Returns:
        trait_df: DataFrame of effect sizes for the DISCOVERY trait
        phenotype_df: DataFrame with phenotypes + population assignments
    """
    import pandas as pd

    distribution = experiment_config['trait_distribution']
    mean = experiment_config['trait_distribution_parameters']['mean']
    std = experiment_config['trait_distribution_parameters']['std']
    num_causal = int(experiment_config.get('num_causal_variants', 100))
    heritability = float(experiment_config.get('heritability', 0.7))
    random_seed = int(experiment_config.get('seed', 42))

    arch_cfg = experiment_config.get("causal_architecture", {}) or {}
    pop_specific = bool(arch_cfg.get("population_specific_causals", False))
    discovery_pop = arch_cfg.get("discovery_pop", "CEU")
    target_pop = arch_cfg.get("target_pop", "YRI")
    overlap_fraction = float(arch_cfg.get("overlap_fraction", 1.0))
    overlap_fraction = min(max(overlap_fraction, 0.0), 1.0)

    model = tstrait.trait_model(distribution=distribution, mean=mean, var=std**2)

    # ------------------------------------------------------------------ #
    # CASE 1: old behavior (shared causal SNPs)
    # ------------------------------------------------------------------ #

    if not pop_specific:
        import pandas as pd

        # 1) Draw ONE causal architecture
        trait_df = tstrait.sim_trait(
            ts=ts, num_causal=num_causal, model=model, random_seed=random_seed
        )
        site_ids = trait_df["site_id"].to_numpy(dtype=int)
        betas = trait_df["effect_size"].to_numpy(dtype=float)

        num_sites = ts.num_sites
        beta = np.zeros(num_sites, dtype=float)
        # (in case tstrait gives any weird ids)
        site_ids = site_ids[(site_ids >= 0) & (site_ids < num_sites)]
        beta[site_ids] = betas[: len(site_ids)]

        # 2) Build genotype matrix (diploid, individuals)
        G = _individual_genotype_matrix(ts)  # (num_inds, num_sites)
        num_inds = G.shape[0]

        # 3) Compute genetic values from the SAME beta
        g_values = G @ beta

        # 4) Add env noise to achieve desired heritability h2
        var_g = np.var(g_values, ddof=1)
        if var_g <= 0:
            sigma_e = 1.0
        else:
            sigma_e = math.sqrt(var_g * (1 - heritability) / max(heritability, 1e-8))

        rng_env = np.random.default_rng(random_seed + 54321)
        env_noise = rng_env.normal(loc=0.0, scale=sigma_e, size=num_inds)
        phenotypes = g_values + env_noise

        # 5) Add population labels
        population_map = {}
        indiv_pops = []
        for ind in ts.individuals():
            node_id = ind.nodes[0]
            pop_id = ts.node(node_id).population
            pop_meta = ts.population(pop_id).metadata
            pop_name = pop_meta.get("name", f"pop{pop_id}") if isinstance(pop_meta, dict) else f"pop{pop_id}"
            population_map[ind.id] = pop_name
            indiv_pops.append(pop_name)
        indiv_pops = np.array(indiv_pops)

        phenotype_df = pd.DataFrame({
            "individual_id": np.arange(num_inds, dtype=int),
            "population": indiv_pops,
            "genetic_value": g_values,
            "environmental_noise": env_noise,
            "phenotype": phenotypes,
        })[["individual_id", "population", "genetic_value", "environmental_noise", "phenotype"]]

        print("realized h2 =", np.var(g_values, ddof=1) / np.var(phenotypes, ddof=1))

        return trait_df, phenotype_df

    # ------------------------------------------------------------------ #
    # CASE 2: population-specific causal SNPs
    # ------------------------------------------------------------------ #

    # 1) Discovery trait: a single tstrait.sim_trait call defines the CEU architecture
    trait_df = tstrait.sim_trait(
        ts=ts, num_causal=num_causal, model=model, random_seed=random_seed
    )
    # trait_df must at least have columns ["site_id", "effect_size"]
    discovery_site_ids = trait_df["site_id"].to_numpy()
    discovery_betas = trait_df["effect_size"].to_numpy()
    num_sites = ts.num_sites

    # 2) Build mapping individual_id -> population
    population_map = {}
    indiv_pops = []  # parallel to individual index in ts.individuals()
    for ind in ts.individuals():
        node_id = ind.nodes[0]
        pop_id = ts.node(node_id).population
        pop_meta = ts.population(pop_id).metadata
        if isinstance(pop_meta, dict):
            pop_name = pop_meta.get("name", f"pop{pop_id}")
        else:
            pop_name = f"pop{pop_id}"
        population_map[ind.id] = pop_name
        indiv_pops.append(pop_name)
    indiv_pops = np.array(indiv_pops)

    # 3) Build individual genotype matrix
    G = _individual_genotype_matrix(ts)  # (num_inds, num_sites)
    num_inds = G.shape[0]

    # 4) Discovery population architecture: betas over ALL sites
    beta_disc = np.zeros(num_sites, dtype=float)
    beta_disc[discovery_site_ids] = discovery_betas

    # 5) Target population architecture: overlap + unique causal sites
    rng = np.random.default_rng(random_seed + 12345)

    k = num_causal
    n_shared = int(round(overlap_fraction * k))
    n_shared = min(n_shared, len(discovery_site_ids))
    shared_idx = rng.choice(len(discovery_site_ids), size=n_shared, replace=False)
    shared_sites = discovery_site_ids[shared_idx]

    # pool of candidate sites not already used in discovery trait
    all_sites = np.arange(num_sites, dtype=int)
    mask_not_disc = np.ones(num_sites, dtype=bool)
    mask_not_disc[discovery_site_ids] = False
    available_sites = all_sites[mask_not_disc]

    n_unique_target = max(k - n_shared, 0)
    if n_unique_target > len(available_sites):
        n_unique_target = len(available_sites)

    if n_unique_target > 0:
        target_unique_sites = rng.choice(available_sites, size=n_unique_target, replace=False)
    else:
        target_unique_sites = np.array([], dtype=int)

    target_sites = np.concatenate([shared_sites, target_unique_sites])

    # effect sizes for target population
    # shared sites reuse discovery betas; unique sites get fresh draws
    beta_target = np.zeros(num_sites, dtype=float)
    beta_target[shared_sites] = beta_disc[shared_sites]

    # draw new effect sizes for target-unique sites
    if len(target_unique_sites) > 0:
        # same marginal distribution as discovery betas
        beta_target[target_unique_sites] = rng.normal(loc=mean, scale=std, size=len(target_unique_sites))

    # 6) Compute genetic values for each individual
    g_values = np.zeros(num_inds, dtype=float)
    for i in range(num_inds):
        pop_name = indiv_pops[i]
        if pop_name == discovery_pop:
            g_values[i] = G[i] @ beta_disc
        elif pop_name == target_pop:
            g_values[i] = G[i] @ beta_target
        else:
            # default: discovery architecture for other pops
            g_values[i] = G[i] @ beta_disc

    # 7) Add environmental noise to match desired heritability
    var_g = np.var(g_values, ddof=1)
    if var_g <= 0:
        sigma_e = 1.0
    else:
        sigma_e = math.sqrt(var_g * (1 - heritability) / max(heritability, 1e-8))

    rng_env = np.random.default_rng(random_seed + 54321)
    env_noise = rng_env.normal(loc=0.0, scale=sigma_e, size=num_inds)
    phenotypes = g_values + env_noise

    # 8) Build phenotype_df to mirror original structure
    phenotype_df = pd.DataFrame({
        "individual_id": np.arange(num_inds, dtype=int),
        "population": indiv_pops,
        "genetic_value": g_values,
        "environmental_noise": env_noise,
        "phenotype": phenotypes,
    })

    # Keep column order consistent
    cols = ["individual_id", "population", "genetic_value", "environmental_noise", "phenotype"]
    phenotype_df = phenotype_df[cols]

    # trait_df is *discovery* trait (CEU) used as "true causal" in GWAS
    return trait_df, phenotype_df



def calculate_fst(ts: tskit.TreeSequence) -> float:
    import numpy as np

    pop_to_samps = {}
    pop_to_name = {}

    for pop in ts.populations():
        samps = ts.samples(population=pop.id)
        if len(samps) == 0:
            continue
        meta = pop.metadata if isinstance(pop.metadata, dict) else {}
        name = meta.get("name", f"pop{pop.id}")
        pop_to_samps[name] = samps
        pop_to_name[pop.id] = name

    # Choose pair in preferred order
    if "YRI" in pop_to_samps and "CEU" in pop_to_samps:
        s1, s2 = pop_to_samps["YRI"], pop_to_samps["CEU"]
    elif "AFR" in pop_to_samps and "EUR" in pop_to_samps:
        s1, s2 = pop_to_samps["AFR"], pop_to_samps["EUR"]
    else:
        # fallback: first two
        if len(pop_to_samps) < 2:
            return 0.0
        names = list(pop_to_samps.keys())
        s1, s2 = pop_to_samps[names[0]], pop_to_samps[names[1]]

    try:
        fst_val = ts.Fst(
            sample_sets=[s1, s2],
            indexes=[(0, 1)],
            windows=None,
            mode="site",
            span_normalise=True,
        )
        return float(fst_val[0])
    except Exception as e:
        print(f"Warning: Fst calculation failed: {e}")
        return 0.0

