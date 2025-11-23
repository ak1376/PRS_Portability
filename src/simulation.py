from __future__ import annotations
from typing import Dict, Tuple, Optional, List

import demes
import msprime
import numpy as np
import stdpopsim as sps
import tskit
import tstrait
import moments
import math


# ──────────────────────────────────
# Minimal helpers
# ──────────────────────────────────

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

def out_of_africa_model(
    sampled: Dict[str, float], cfg: Optional[Dict] = None
) -> demes.Graph:
    """
    Gutenkunst et al. (2009) Out-of-Africa model.
    Parameters can be overridden by 'sampled' dict.
    Defaults are from the original paper (converted to generations assuming g=25).
    """
    # Default parameters (generations)
    # Ancestral size
    N_A = sampled.get("N_A", 7300)
    
    # Bottleneck size (OOA)
    N_B = sampled.get("N_B", 2100)
    
    # Modern African size
    N_AF = sampled.get("N_AF", 12300)
    
    # European initial size
    N_EU0 = sampled.get("N_EU0", 1000)
    
    # Asian initial size
    N_AS0 = sampled.get("N_AS0", 510)
    
    # Growth rates
    r_EU = sampled.get("r_EU", 0.004)
    r_AS = sampled.get("r_AS", 0.0055)
    
    # Times (generations ago)
    # T_AFR_OOA: Split of African and OOA (Bottleneck)
    T_AFR_OOA = sampled.get("T_AFR_OOA", 5600) # ~140k years
    
    # T_EU_AS: Split of European and Asian
    T_EU_AS = sampled.get("T_EU_AS", 848) # ~21.2k years
    
    # Migration rates
    m_AF_B = sampled.get("m_AF_B", 25e-5)
    m_AF_EU = sampled.get("m_AF_EU", 3e-5)
    m_AF_AS = sampled.get("m_AF_AS", 1.9e-5)
    m_EU_AS = sampled.get("m_EU_AS", 9.6e-5)
    
    b = demes.Builder()
    
    # Ancestral population
    b.add_deme("ANC", epochs=[dict(start_size=N_A, end_time=T_AFR_OOA)])
    
    # African population (YRI)
    # Expands from N_A to N_AF instantly at T_AFR_OOA (in the paper it's instantaneous)
    b.add_deme("YRI", ancestors=["ANC"], epochs=[dict(start_size=N_AF, end_time=0)])
    
    # OOA Bottleneck population
    b.add_deme("OOA", ancestors=["ANC"], epochs=[dict(start_size=N_B, end_time=T_EU_AS)])
    
    # European population (CEU)
    # Exponential growth from N_EU0
    # End size = N_EU0 * exp(r_EU * T_EU_AS)
    N_EU_end = N_EU0 * np.exp(r_EU * T_EU_AS)
    b.add_deme("CEU", ancestors=["OOA"], epochs=[dict(start_size=N_EU0, end_size=N_EU_end, end_time=0)])
    
    # Asian population (CHB)
    # Exponential growth from N_AS0
    N_AS_end = N_AS0 * np.exp(r_AS * T_EU_AS)
    b.add_deme("CHB", ancestors=["OOA"], epochs=[dict(start_size=N_AS0, end_size=N_AS_end, end_time=0)])
    
    # Migration
    # Migration between AFR and OOA (Bottleneck)
    b.add_migration(source="YRI", dest="OOA", rate=m_AF_B)
    b.add_migration(source="OOA", dest="YRI", rate=m_AF_B)
    
    # Migration between AFR and EUR/ASN
    b.add_migration(source="YRI", dest="CEU", rate=m_AF_EU)
    b.add_migration(source="CEU", dest="YRI", rate=m_AF_EU)
    
    b.add_migration(source="YRI", dest="CHB", rate=m_AF_AS)
    b.add_migration(source="CHB", dest="YRI", rate=m_AF_AS)
    
    # Migration between EUR and ASN
    b.add_migration(source="CEU", dest="CHB", rate=m_EU_AS)
    b.add_migration(source="CHB", dest="CEU", rate=m_EU_AS)
    
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
    
    elif model_type == "out_of_africa":
        # For OOA, we can use the custom demes wrapper since we built it as a demes graph
        return _ModelFromDemes(g, model_id="custom_ooa", desc="Custom OOA")

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
    elif model_type == "out_of_africa":
        g = out_of_africa_model(sampled_params, experiment_config)
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
        # Simulate effect sizes only
        trait_df = tstrait.sim_trait(
            ts=ts, num_causal=num_causal, model=model, random_seed=random_seed
        )

        # Simulate phenotypes (shared architecture)
        phenotype_result = tstrait.sim_phenotype(
            ts=ts, num_causal=num_causal, model=model,
            h2=heritability, random_seed=random_seed
        )
        phenotype_df = phenotype_result.phenotype.copy()

        # Add population information
        population_map = {}
        for ind in ts.individuals():
            node_id = ind.nodes[0]
            pop_id = ts.node(node_id).population
            pop_meta = ts.population(pop_id).metadata
            if isinstance(pop_meta, dict):
                pop_name = pop_meta.get("name", f"pop{pop_id}")
            else:
                pop_name = f"pop{pop_id}"
            population_map[ind.id] = pop_name

        phenotype_df["population"] = phenotype_df["individual_id"].map(population_map)
        cols = list(phenotype_df.columns)
        cols.remove("population")
        idx = cols.index("individual_id")
        cols.insert(idx + 1, "population")
        phenotype_df = phenotype_df[cols]

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

