from __future__ import annotations
from typing import Dict, Tuple, Optional, List

import demes
import msprime
import numpy as np
import stdpopsim as sps
import tskit
import tstrait
import moments


# ──────────────────────────────────
# Minimal helpers
# ──────────────────────────────────

class _ModelFromDemes(sps.DemographicModel):
    """Wrap a demes.Graph so stdpopsim engines can simulate it (bottleneck, drosophila)."""

    def __init__(
        self,
        g: demes.Graph,
        model_id: str = "custom_from_demes",
        desc: str = "custom demes",
    ):
        model = msprime.Demography.from_demes(g)
        super().__init__(
            id=model_id,
            description=desc,
            long_description=desc,
            model=model,
            generation_time=1,
        )


# Leaf-first stdpopsim models for SLiM (avoid p0=ANC extinction at split)
class _IM_Symmetric(sps.DemographicModel):
    """
    Isolation-with-migration, symmetric: YRI <-> CEU with rate m; split at time T from ANC.
    Populations are added as leaves first so p0/p1 are YRI/CEU (not ANC), avoiding zero-size errors.
    """

    def __init__(self, N0, N1, N2, T, m):
        dem = msprime.Demography()
        dem.add_population(name="YRI", initial_size=float(N1))
        dem.add_population(name="CEU", initial_size=float(N2))
        dem.add_population(name="ANC", initial_size=float(N0))
        m = float(m)
        dem.set_migration_rate(source="YRI", dest="CEU", rate=m)
        dem.set_migration_rate(source="CEU", dest="YRI", rate=m)
        dem.add_population_split(time=float(T), ancestral="ANC", derived=["YRI", "CEU"])
        super().__init__(
            id="IM_sym",
            description="Isolation-with-migration, symmetric",
            long_description="ANC splits at T into YRI and CEU; symmetric migration m.",
            model=dem,
            generation_time=1,
        )


class _IM_Asymmetric(sps.DemographicModel):
    """Isolation-with-migration, asymmetric: YRI→CEU rate m12; CEU→YRI rate m21."""

    def __init__(self, N0, N1, N2, T, m12, m21):
        dem = msprime.Demography()

        # ✅ Add leaves first so that p0/p1 are extant pops, not ANC
        dem.add_population(name="YRI", initial_size=float(N1))
        dem.add_population(name="CEU", initial_size=float(N2))
        dem.add_population(name="ANC", initial_size=float(N0))

        # asymmetric migration
        # Forward-time: m12 = YRI→CEU, m21 = CEU→YRI
        # Backward-time encoding for msprime:
        dem.set_migration_rate(source="CEU", dest="YRI", rate=float(m12))  # encode YRI→CEU
        dem.set_migration_rate(source="YRI", dest="CEU", rate=float(m21))  # encode CEU→YRI


        # split backward in time
        dem.add_population_split(time=float(T), ancestral="ANC", derived=["YRI", "CEU"])

        super().__init__(
            id="IM_asym",
            description="Isolation-with-migration, asymmetric",
            long_description=(
                "ANC splits at T into YRI and CEU; asymmetric migration m12 and m21."
            ),
            model=dem,
            generation_time=1,
        )

class _Bottleneck(sps.DemographicModel):
    """
    Single-population bottleneck implemented directly in msprime.Demography.
    Times in generations before present (t_start > t_end >= 0).
    """

    def __init__(self, N0, N_bottleneck, N_recover, t_bottleneck_start, t_bottleneck_end):
        t_start = float(t_bottleneck_start)
        t_end   = float(t_bottleneck_end)
        if not (t_start > t_end >= 0):
            raise ValueError("Require t_bottleneck_start > t_bottleneck_end >= 0.")

        dem = msprime.Demography()
        dem.add_population(name="ANC", initial_size=float(N0))

        # At t_start, drop to the bottleneck size
        dem.add_population_parameters_change(
            time=t_start, population="ANC", initial_size=float(N_bottleneck)
        )
        # At t_end, recover to N_recover (constant to present)
        dem.add_population_parameters_change(
            time=t_end, population="ANC", initial_size=float(N_recover)
        )

        super().__init__(
            id="bottleneck",
            description="Single-population bottleneck (N0 → N_bottleneck → N_recover).",
            long_description=(
                "One population with ancestral size N0 until t_bottleneck_start, "
                "then a bottleneck of size N_bottleneck until t_bottleneck_end, "
                "then constant size N_recover to the present."
            ),
            model=dem,
            generation_time=1,
        )

class _DrosophilaThreeEpoch(sps.DemographicModel):
    """
    Two-pop Drosophila-style three-epoch model.

    ANC (size N0) splits at T_AFR_EUR_split into:
      - AFR: constant size AFR (AFR_recover in your priors)
      - EUR: bottleneck of size EUR_bottleneck until T_EUR_expansion,
             then recovery to EUR_recover up to the present.

    Populations are added leaf-first so p0/p1 are AFR/EUR (not ANC),
    which plays nicely with SLiM’s population ordering.
    """

    def __init__(
        self,
        N0,
        AFR,
        EUR_bottleneck,
        EUR_recover,
        T_AFR_EUR_split,
        T_EUR_expansion,
    ):
        T_split = float(T_AFR_EUR_split)
        T_exp   = float(T_EUR_expansion)

        dem = msprime.Demography()

        # Leaf-first: extant pops first, then ANC
        dem.add_population(name="AFR", initial_size=float(AFR))
        dem.add_population(name="EUR", initial_size=float(EUR_bottleneck))
        dem.add_population(name="ANC", initial_size=float(N0))

        # EUR expansion (bottleneck -> recovery) at T_EUR_expansion
        dem.add_population_parameters_change(
            time=T_exp,
            population="EUR",
            initial_size=float(EUR_recover),
        )

        # Split backward in time at T_AFR_EUR_split: AFR/EUR merge into ANC
        dem.add_population_split(
            time=T_split,
            ancestral="ANC",
            derived=["AFR", "EUR"],
        )

        super().__init__(
            id="drosophila_three_epoch",
            description="Drosophila-style three-epoch AFR/EUR model",
            long_description=(
                "ANC (N0) until T_AFR_EUR_split, then split into AFR and EUR. "
                "AFR stays at AFR; EUR has a bottleneck (EUR_bottleneck) and "
                "expands at T_EUR_expansion to EUR_recover."
            ),
            model=dem,
            generation_time=1,
        )


# ──────────────────────────────────
# NEW: interval helpers for coverage-based tiling
# ──────────────────────────────────


def _sanitize_nonoverlap(intervals: np.ndarray, L: int) -> np.ndarray:
    if intervals.size == 0:
        return intervals
    iv = intervals[np.argsort(intervals[:, 0])]
    out = []
    prev_end = -1
    for s, e in iv:
        s = int(max(0, min(s, L)))
        e = int(max(0, min(e, L)))
        if e <= s:
            continue
        if s < prev_end:
            continue
        out.append((s, e))
        prev_end = e
    return np.array(out, dtype=int) if out else np.empty((0, 2), dtype=int)


def _build_tiling_intervals(
    L: int, exon_bp: int, tile_bp: int, jitter_bp: int = 0
) -> np.ndarray:
    starts = np.arange(0, max(0, L - exon_bp + 1), tile_bp, dtype=int)
    if jitter_bp > 0 and len(starts) > 0:
        rng = np.random.default_rng()
        jitter = rng.integers(-jitter_bp, jitter_bp + 1, size=len(starts))
        starts = np.clip(starts + jitter, 0, max(0, L - exon_bp))
    ends = np.minimum(starts + int(exon_bp), L).astype(int)
    iv = np.column_stack([starts, ends])
    return _sanitize_nonoverlap(iv, L)


def _intervals_from_coverage(
    L: int, exon_bp: int, coverage: float, jitter_bp: int = 0
) -> np.ndarray:
    """coverage in [0,1]. If 0 → empty; if 1 → whole contig; else tiling to approximate coverage."""
    if coverage <= 0:
        return np.empty((0, 2), dtype=int)
    if coverage >= 1.0:
        return np.array([[0, int(L)]], dtype=int)
    # spacing chosen so expected selected fraction ≈ coverage
    tile_bp = max(int(exon_bp), int(round(exon_bp / float(max(coverage, 1e-12)))))
    return _build_tiling_intervals(int(L), int(exon_bp), tile_bp, jitter_bp=jitter_bp)


def _contig_from_cfg(cfg: Dict, sel: Dict):
    """
    Synthetic-only contig builder.
    Builds a stdpopsim Contig of length = cfg["genome_length"],
    with user-specified mutation_rate and recombination_rate.
    """
    sp = sps.get_species(sel.get("species", "HomSap"))

    L = float(cfg["genome_length"])
    mu = float(cfg["mutation_rate"]) if "mutation_rate" in cfg else None
    r = float(cfg["recombination_rate"]) if "recombination_rate" in cfg else None

    try:
        # Newer stdpopsim supports recombination_rate kwarg
        return sp.get_contig(
            chromosome=None,
            length=L,
            mutation_rate=mu,
            recombination_rate=r,
        )
    except TypeError:
        # Older stdpopsim doesn’t accept recombination_rate
        if r is not None:
            print(
                "[warn] This stdpopsim version ignores custom recombination_rate; "
                "using species default instead."
            )
        return sp.get_contig(
            chromosome=None,
            length=L,
            mutation_rate=mu,
        )


# CHANGED: now supports optional coverage tiling across the contig
def _apply_dfe_intervals(
    contig, sel: Dict, sampled_coverage: Optional[float] = None
) -> Dict[str, float]:
    """
    Attach DFE over intervals determined by:
      1) sampled_coverage (takes precedence; may be percent >1 or fraction <=1),
      2) sel['coverage_fraction'] or sel['coverage_percent'],
      3) sel['tile_bp'], or
      4) whole contig by default.

    Returns summary {selected_bp, selected_frac}.
    """
    sp = sps.get_species(sel.get("species", "HomSap"))
    dfe = sp.get_dfe(sel.get("dfe_id", "Gamma_K17"))

    # robust length getter
    L = int(
        getattr(contig, "length", getattr(contig, "recombination_map").sequence_length)
    )

    exon_bp = int(sel.get("exon_bp", 200))
    jitter_bp = int(sel.get("jitter_bp", 0))

    # 1) sampled_coverage overrides config if provided
    cov_frac = None
    if sampled_coverage is not None:
        # Accept either fraction in [0,1] or percent in (1,100]
        cov_frac = float(sampled_coverage)
        if cov_frac > 1.0:  # assume user passed a percent
            cov_frac = cov_frac / 100.0

    # 2) fall back to config coverage
    if cov_frac is None:
        if "coverage_fraction" in sel:
            cov_frac = float(sel["coverage_fraction"])
        elif "coverage_percent" in sel:
            cov_frac = float(sel["coverage_percent"]) / 100.0

    # Build intervals
    if cov_frac is not None:
        intervals = _intervals_from_coverage(L, exon_bp, cov_frac, jitter_bp=jitter_bp)
    elif "tile_bp" in sel and sel["tile_bp"] is not None:
        intervals = _build_tiling_intervals(
            L, exon_bp, int(sel["tile_bp"]), jitter_bp=jitter_bp
        )
    else:
        intervals = np.array([[0, L]], dtype=int)

    if intervals.size > 0:
        contig.add_dfe(intervals=intervals, DFE=dfe)

    selected_bp = int(
        np.sum((intervals[:, 1] - intervals[:, 0])) if intervals.size else 0
    )
    return dict(
        selected_bp=selected_bp,
        selected_frac=(selected_bp / float(L) if L > 0 else 0.0),
    )


# ──────────────────────────────────
# Your demography builders (demes)
# ──────────────────────────────────


def bottleneck_model(
    sampled: Dict[str, float], cfg: Optional[Dict] = None
) -> demes.Graph:
    b = demes.Builder()
    b.add_deme(
        "ANC",
        epochs=[
            dict(
                start_size=float(sampled["N0"]),
                end_time=float(sampled["t_bottleneck_start"]),
            ),
            dict(
                start_size=float(sampled["N_bottleneck"]),
                end_time=float(sampled["t_bottleneck_end"]),
            ),
            dict(start_size=float(sampled["N_recover"]), end_time=0),
        ],
    )
    return b.resolve()


def split_isolation_model(
    sampled: Dict[str, float], cfg: Optional[Dict] = None
) -> demes.Graph:
    """Split + symmetric low migration (YRI/CEU)."""
    N0 = float(sampled.get("N_anc", sampled.get("N0")))
    N1 = float(sampled.get("N_YRI", sampled.get("N1")))
    N2 = float(sampled.get("N_CEU", sampled.get("N2")))
    T = float(sampled.get("T_split", sampled.get("t_split")))
    # accept MANY possible keys; if both directions provided, average them
    m_keys = ["m", "m_sym", "m12", "m21", "m_YRI_CEU", "m_CEU_YRI"]
    vals = [float(sampled[k]) for k in m_keys if k in sampled]
    m = float(np.mean(vals)) if vals else 0.0

    b = demes.Builder()
    b.add_deme("ANC", epochs=[dict(start_size=N0, end_time=T)])
    b.add_deme("YRI", ancestors=["ANC"], epochs=[dict(start_size=N1)])
    b.add_deme("CEU", ancestors=["ANC"], epochs=[dict(start_size=N2)])
    if m > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m)
        b.add_migration(source="CEU", dest="YRI", rate=m)
    return b.resolve()


def split_migration_model(
    sampled: Dict[str, float]
) -> demes.Graph:
    """
    Split + asymmetric migration (two rates).
    Deme names: 'YRI' and 'CEU'.
    """
    N0 = float(sampled.get("N_anc", sampled.get("N0")))
    N1 = float(sampled.get("N_YRI", sampled.get("N1")))
    N2 = float(sampled.get("N_CEU", sampled.get("N2")))
    T = float(sampled.get("T_split", sampled.get("t_split")))
    m12 = float(sampled.get("m_YRI_CEU", sampled.get("m12", sampled.get("m", 0.0))))
    m21 = float(sampled.get("m_CEU_YRI", sampled.get("m21", sampled.get("m", 0.0))))

    b = demes.Builder()
    b.add_deme("ANC", epochs=[dict(start_size=N0, end_time=T)])
    b.add_deme("YRI", ancestors=["ANC"], epochs=[dict(start_size=N1)])
    b.add_deme("CEU", ancestors=["ANC"], epochs=[dict(start_size=N2)])
    if m12 > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m12)
    if m21 > 0:
        b.add_migration(source="CEU", dest="YRI", rate=m21)
    return b.resolve()


def drosophila_three_epoch(
    sampled: Dict[str, float], cfg: Optional[Dict] = None
) -> demes.Graph:
    """
    Two-pop Drosophila-style model:
      ANC (size N0) → split at T_AFR_EUR_split → AFR (AFR_recover)
      and EUR with a bottleneck at T_EUR_expansion then recovery to EUR_recover.
    Deme names: 'AFR' and 'EUR'.
    """
    N0 = float(sampled["N0"])
    AFR_recover = float(sampled["AFR"])
    EUR_bottleneck = float(sampled["EUR_bottleneck"])
    EUR_recover = float(sampled["EUR_recover"])
    T_split = float(sampled["T_AFR_EUR_split"])
    T_EUR_exp = float(sampled["T_EUR_expansion"])

    b = demes.Builder()
    b.add_deme("ANC", epochs=[dict(start_size=N0, end_time=T_split)])
    b.add_deme("AFR", ancestors=["ANC"], epochs=[dict(start_size=AFR_recover)])
    b.add_deme(
        "EUR",
        ancestors=["ANC"],
        epochs=[
            dict(start_size=EUR_bottleneck, end_time=T_EUR_exp),
            dict(start_size=EUR_recover, end_time=0),
        ],
    )
    return b.resolve()

def define_sps_model(model_type: str, g: demes.Graph, sampled_params: Dict[str, float]) -> sps.DemographicModel:
    """Create appropriate stdpopsim model for SLiM based on model type."""
    if model_type == "split_isolation":
        # Symmetric migration model - extract parameters
        N0 = float(sampled_params.get("N_anc", sampled_params.get("N0")))
        N1 = float(sampled_params.get("N_YRI", sampled_params.get("N1"))) 
        N2 = float(sampled_params.get("N_CEU", sampled_params.get("N2")))
        T = float(sampled_params.get("T_split", sampled_params.get("t_split")))
        m_keys = ["m", "m_sym", "m12", "m21", "m_YRI_CEU", "m_CEU_YRI"]
        vals = [float(sampled_params[k]) for k in m_keys if k in sampled_params]
        m = float(np.mean(vals)) if vals else 0.0
        return _IM_Symmetric(N0, N1, N2, T, m)
    
    elif model_type == "split_migration":
        # Asymmetric migration model - extract parameters
        N0 = float(sampled_params.get("N_anc", sampled_params.get("N0")))
        N1 = float(sampled_params.get("N_YRI", sampled_params.get("N1")))
        N2 = float(sampled_params.get("N_CEU", sampled_params.get("N2")))
        T = float(sampled_params.get("T_split", sampled_params.get("t_split")))
        m12 = float(sampled_params.get("m_YRI_CEU", sampled_params.get("m12", sampled_params.get("m", 0.0))))
        m21 = float(sampled_params.get("m_CEU_YRI", sampled_params.get("m21", sampled_params.get("m", 0.0))))
        return _IM_Asymmetric(N0, N1, N2, T, m12, m21)
    
    elif model_type == "drosophila_three_epoch":
        # Two-pop Drosophila three-epoch model
        N0             = float(sampled_params["N0"])
        AFR            = float(sampled_params["AFR"])
        EUR_bottleneck = float(sampled_params["EUR_bottleneck"])
        EUR_recover    = float(sampled_params["EUR_recover"])
        T_split        = float(sampled_params["T_AFR_EUR_split"])
        T_EUR_exp      = float(sampled_params["T_EUR_expansion"])

        return _DrosophilaThreeEpoch(
            N0,
            AFR,
            EUR_bottleneck,
            EUR_recover,
            T_split,
            T_EUR_exp,
        )

    else:
        # For bottleneck or any other demes-based custom model
        return _ModelFromDemes(g, model_id=f"custom_{model_type}", desc="custom demes")

# ──────────────────────────────────
# Main entry: BGS only (SLiM via stdpopsim)
# ──────────────────────────────────

def msprime_simulation(g: demes.Graph,
    experiment_config: Dict
) -> Tuple[tskit.TreeSequence, demes.Graph]:

    samples = {pop_name: num_samples for pop_name, num_samples in experiment_config['num_samples'].items()}

    demog = msprime.Demography.from_demes(g)

    # Simulate ancestry for two populations (joint simulation)
    ts = msprime.sim_ancestry(
        samples=samples,  # Two populations
        demography=demog,
        sequence_length=experiment_config['genome_length'],
        recombination_rate=experiment_config['recombination_rate'],
        random_seed=experiment_config['seed'],
    )
    
    # Simulate mutations over the ancestry tree sequence
    ts = msprime.sim_mutations(ts, rate=experiment_config['mutation_rate'], random_seed=experiment_config['seed'])

    return ts, g

def stdpopsim_slim_simulation(g: demes.Graph,
    experiment_config: Dict, 
    sampled_coverage: float,
    model_type: str,
    sampled_params: Dict[str, float]
) -> Tuple[tskit.TreeSequence, demes.Graph]:

    # 1) Pick model (wrap Demes for stdpopsim)
    model = define_sps_model(model_type, g, sampled_params)

    # 2) Build contig and apply DFE intervals
    sel = experiment_config.get("selection") or {}
    contig = _contig_from_cfg(experiment_config, sel)
    sel_summary = _apply_dfe_intervals(contig, sel, sampled_coverage=sampled_coverage)

    # 3) Samples
    samples = {k: int(v) for k, v in (experiment_config.get("num_samples") or {}).items()}
    base_seed = experiment_config.get("seed", None)

    # 4) Run SLiM via stdpopsim
    eng = sps.get_engine("slim")
    ts = eng.simulate(
        model,
        contig,
        samples,
        slim_scaling_factor=float(sel.get("slim_scaling", 10.0)),
        slim_burn_in=float(sel.get("slim_burn_in", 5.0)),
        seed=base_seed,
    )

    ts._bgs_selection_summary = sel_summary
    return ts, g

def simulation(
    sampled_params: Dict[str, float],
    model_type: str,
    experiment_config: Dict,
    sampled_coverage: Optional[float] = None,
) -> Tuple[tskit.TreeSequence, demes.Graph]:

    # Build demes graph (kept for plotting/metadata)
    if model_type == "bottleneck":
        g = bottleneck_model(sampled_params, experiment_config)
    elif model_type == "split_isolation":
        g = split_isolation_model(sampled_params, experiment_config)  # symmetric m
    elif model_type == "split_migration":
        g = split_migration_model(sampled_params)  # asymmetric
    elif model_type == "drosophila_three_epoch":
        g = drosophila_three_epoch(sampled_params, experiment_config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    engine = str(experiment_config.get("engine", "")).lower()
    sel = experiment_config.get("selection") or {}

    if engine == "msprime":
        # Neutral path: no BGS, no coverage needed/used
        return msprime_simulation(g, experiment_config)

    if engine == "slim":
        # BGS path: require selection.enabled and a coverage
        if not sel.get("enabled", False):
            raise ValueError("engine='slim' requires selection.enabled=true in your config.")
        if sampled_coverage is None:
            raise ValueError("engine='slim' requires a non-None sampled_coverage (percent or fraction).")
        return stdpopsim_slim_simulation(g, experiment_config, sampled_coverage, model_type, sampled_params)

    raise ValueError("engine must be 'slim' or 'msprime'.")

# ──────────────────────────────────
# SFS utility
# ──────────────────────────────────

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
    Simulate a quantitative trait under a normal model
    using tstrait's trait simulator.
    
    Returns:
        Tuple containing:
        - trait_df: DataFrame with effect sizes from sim_trait (columns: position, site_id, effect_size, etc.)
        - phenotype_df: DataFrame with phenotypes AND population assignments
    """
    import pandas as pd

    distribution = experiment_config['trait_distribution']
    mean = experiment_config['trait_distribution_parameters']['mean']
    std = experiment_config['trait_distribution_parameters']['std']
    num_causal = experiment_config.get('num_causal_variants', 100)
    heritability = experiment_config.get('heritability', 0.7)
    random_seed = experiment_config.get('seed', 42)

    model = tstrait.trait_model(distribution=distribution, mean=mean, var=std**2)
    
    # Simulate effect sizes only
    trait_df = tstrait.sim_trait(
        ts=ts, num_causal=num_causal, model=model, random_seed=random_seed
    )
    
    # Simulate complete phenotypes (effect sizes + genetic values + environmental noise)
    phenotype_result = tstrait.sim_phenotype(
        ts=ts, num_causal=num_causal, model=model, h2=heritability, random_seed=random_seed
    )
    
    # Add population information to phenotype DataFrame
    phenotype_df = phenotype_result.phenotype.copy()
    
    # Extract population assignment for each individual from tree sequence
    population_map = {}
    for ind in ts.individuals():
        # Get the population ID for this individual's first node
        node_id = ind.nodes[0]
        pop_id = ts.node(node_id).population
        # Get population name from metadata
        pop_metadata = ts.population(pop_id).metadata
        if isinstance(pop_metadata, dict):
            pop_name = pop_metadata.get('name', f'pop{pop_id}')
        else:
            pop_name = f'pop{pop_id}'
        population_map[ind.id] = pop_name
    
    # Add population column to phenotype DataFrame
    phenotype_df['population'] = phenotype_df['individual_id'].map(population_map)
    
    # Reorder columns to put population after individual_id
    cols = list(phenotype_df.columns)
    # Move 'population' to be right after 'individual_id'
    cols.remove('population')
    individual_id_idx = cols.index('individual_id')
    cols.insert(individual_id_idx + 1, 'population')
    phenotype_df = phenotype_df[cols]

    # Return both: trait effect sizes and complete phenotype simulation with population info
    return trait_df, phenotype_df



def calculate_fst(ts: tskit.TreeSequence) -> float:
    """
    Calculate Fst between two populations (YRI and CEU) if they exist.
    Returns the mean Fst.
    """
    # Identify sample sets for YRI and CEU
    sample_sets = []
    pop_names = []
    
    # Map common names
    target_pops = ["YRI", "CEU", "AFR", "EUR"]
    
    for pop in ts.populations():
        samps = ts.samples(population=pop.id)
        if len(samps) > 0:
            meta = pop.metadata if isinstance(pop.metadata, dict) else {}
            name = meta.get("name", f"pop{pop.id}")
            
            # If we find specific target pops, prioritize them
            # Otherwise just take the first two we find
            if name in target_pops or len(sample_sets) < 2:
                 # Only add if we haven't already added this population (by ID)
                 # But here we iterate by population so it's unique
                 sample_sets.append(samps)
                 pop_names.append(name)
    
    if len(sample_sets) < 2:
        return 0.0 # Not enough pops for Fst
        
    # Calculate Fst genome-wide (windows=None)
    # indexes=[(0, 1)] compares the first two sample sets found
    try:
        fst_val = ts.Fst(sample_sets, indexes=[(0, 1)], windows=None, mode='site', span_normalise=True)
        return float(fst_val[0])
    except Exception as e:
        print(f"Warning: Fst calculation failed: {e}")
        return 0.0
