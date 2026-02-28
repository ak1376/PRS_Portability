import numpy as np
import stdpopsim as sps
import matplotlib.pyplot as plt
import pandas as pd
import demes
import demesdraw
from typing import Dict, Optional, Tuple, Set, List
import tskit
import msprime
from pathlib import Path
import tstrait

"""
Simulate the IM Symmetric model, simulate traits, and examine:
1) Allele-frequency differences by population (all SNPs + causal SNPs)
2) LD tagging differences around causal SNPs (YRI vs CEU), FILTERED to causal SNPs
   that are polymorphic in BOTH populations (so LD is defined in both)
3) Phenotype / noise distributions

Outputs saved to: simulated_data/
  - demes_plot.png
  - simulated.trees
  - simulated_traits.csv
  - simulated_phenotypes.csv
  - phenotype_hist.png
  - environmental_noise_hist.png
  - allele_freqs_by_pop.csv
  - allele_freq_delta_p_hist.png
  - allele_freq_scatter.png
  - ld_tagging/ (plots + per-causal LD tables)
"""


# =============================================================================
# Demography
# =============================================================================
def IM_symmetric_model(sampled: Dict[str, float]) -> demes.Graph:
    required_keys = ["N_anc", "N_YRI", "N_CEU", "m", "T_split"]
    for k in required_keys:
        assert k in sampled, f"Missing required key: {k}"

    N0 = float(sampled["N_anc"])
    N1 = float(sampled["N_YRI"])
    N2 = float(sampled["N_CEU"])
    T = float(sampled["T_split"])
    m = float(sampled["m"])

    assert T > 0, "T_split must be > 0"

    b = demes.Builder(time_units="generations", generation_time=1)

    b.add_deme(
        "YRI",
        epochs=[
            dict(start_size=N0, end_time=T),
            dict(start_size=N1, end_time=0),
        ],
    )

    b.add_deme(
        "CEU",
        ancestors=["YRI"],
        start_time=T,
        epochs=[dict(start_size=N2, end_time=0)],
    )

    if m > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m, start_time=T, end_time=0)
        b.add_migration(source="CEU", dest="YRI", rate=m, start_time=T, end_time=0)

    return b.resolve()


# =============================================================================
# Simulation
# =============================================================================
def simulation_runner(
    g: demes.Graph,
    L: float,
    mu: float,
    r: float,
    samples: Dict[str, int],
    seed: Optional[int] = 295,
) -> tskit.TreeSequence:
    sp = sps.get_species("HomSap")

    model = sps.DemographicModel(
        id="IM_symmetric",
        description="IM symmetric model",
        long_description="IM symmetric model",
        model=msprime.Demography.from_demes(g),
        generation_time=1,
    )

    contig = sp.get_contig(
        chromosome=None,
        length=L,
        mutation_rate=mu,
        recombination_rate=r,
    )

    print("contig.length:", contig.length)
    print("contig.mutation_rate:", contig.mutation_rate)
    print("contig.recombination_map.mean_rate:", contig.recombination_map.mean_rate)

    eng = sps.get_engine("msprime")
    ts = eng.simulate(model, contig, samples, seed=seed)
    return ts


# =============================================================================
# Population labels for phenotype DF (robust across tstrait versions)
# =============================================================================
def add_population_column(ts: tskit.TreeSequence, phenotype_df: pd.DataFrame) -> pd.DataFrame:
    possible_cols = ["individual", "individual_id", "ind", "ind_id", "sample", "sample_id"]
    id_col = next((c for c in possible_cols if c in phenotype_df.columns), None)

    if id_col is None:
        if pd.api.types.is_integer_dtype(phenotype_df.index):
            ind_ids = phenotype_df.index.to_numpy()
            id_series = pd.Series(ind_ids, index=phenotype_df.index, name="individual")
        else:
            raise KeyError(
                "Couldn't find an individual id column in phenotype_df. "
                f"Columns are: {list(phenotype_df.columns)}"
            )
    else:
        id_series = phenotype_df[id_col].astype(int)

    # population_id -> name (fallback if metadata isn't a dict)
    pop_id_to_name = {}
    for i in range(ts.num_populations):
        md = ts.population(i).metadata
        if isinstance(md, dict) and "name" in md:
            pop_id_to_name[i] = md["name"]
        else:
            pop_id_to_name[i] = f"pop{i}"

    # individual_id -> population_name
    ind_to_pop = {}
    for ind_id, ind in enumerate(ts.individuals()):
        if len(ind.nodes) == 0:
            ind_to_pop[ind_id] = None
            continue
        node_id = ind.nodes[0]
        pop_id = ts.node(node_id).population
        ind_to_pop[ind_id] = pop_id_to_name[pop_id]

    out = phenotype_df.copy()
    out["population"] = id_series.map(ind_to_pop).values
    return out


# =============================================================================
# Allele frequency diagnostics
# =============================================================================
def pop_sample_nodes(ts: tskit.TreeSequence) -> Dict[str, np.ndarray]:
    """
    Return sample node IDs grouped by population name.
    """
    pop_id_to_name = {}
    for i in range(ts.num_populations):
        md = ts.population(i).metadata
        if isinstance(md, dict) and "name" in md:
            pop_id_to_name[i] = md["name"]
        else:
            pop_id_to_name[i] = f"pop{i}"

    pops = {name: [] for name in pop_id_to_name.values()}
    for n in ts.samples():
        pid = ts.node(n).population
        pops[pop_id_to_name[pid]].append(n)

    return {k: np.array(v, dtype=int) for k, v in pops.items()}


def _pop_sample_indices_from_nodes(ts: tskit.TreeSequence, pop_nodes: np.ndarray) -> np.ndarray:
    """
    Convert sample NODE IDs -> indices into ts.samples() ordering (for var.genotypes slicing).
    """
    sample_nodes = ts.samples()
    node_to_sample_index = {node_id: i for i, node_id in enumerate(sample_nodes)}
    return np.array([node_to_sample_index[n] for n in pop_nodes], dtype=int)


def allele_freqs_two_pops(ts: tskit.TreeSequence, popA: str, popB: str) -> pd.DataFrame:
    """
    Compute derived allele frequencies per site for two populations and delta_p.
    Assumes derived allele is allele index 1 (typical for msprime biallelic sims).
    """
    pops = pop_sample_nodes(ts)
    if popA not in pops or popB not in pops:
        raise ValueError(f"Expected populations '{popA}' and '{popB}'. Found: {list(pops.keys())}")

    idxA = _pop_sample_indices_from_nodes(ts, pops[popA])
    idxB = _pop_sample_indices_from_nodes(ts, pops[popB])

    rows = []
    for var in ts.variants():
        g = var.genotypes
        pA = float(np.mean(g[idxA] == 1))
        pB = float(np.mean(g[idxB] == 1))
        rows.append(
            {
                "site_id": int(var.site.id),
                "position": float(var.site.position),
                f"p_{popA}": pA,
                f"p_{popB}": pB,
                "delta_p": abs(pA - pB),
            }
        )

    return pd.DataFrame(rows)


def plot_af_diagnostics(af_df: pd.DataFrame, popA: str, popB: str, outdir: Path) -> None:
    # Histogram of delta_p
    plt.figure()
    if "is_causal" in af_df.columns:
        plt.hist(af_df.loc[~af_df["is_causal"], "delta_p"], bins=50, alpha=0.6, label="all (non-causal)")
        plt.hist(af_df.loc[af_df["is_causal"], "delta_p"], bins=50, alpha=0.6, label="causal")
        plt.legend()
    else:
        plt.hist(af_df["delta_p"], bins=50)
    plt.xlabel(f"|p_{popA} - p_{popB}|")
    plt.ylabel("Number of SNPs")
    plt.title("Allele-frequency differences")
    plt.tight_layout()
    plt.savefig(outdir / "allele_freq_delta_p_hist.png", dpi=300)
    plt.close()

    # Scatter pA vs pB
    plt.figure()
    plt.scatter(af_df[f"p_{popA}"], af_df[f"p_{popB}"], s=5, alpha=0.35)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel(f"p_{popA}")
    plt.ylabel(f"p_{popB}")
    plt.title("Allele frequencies by population")
    plt.tight_layout()
    plt.savefig(outdir / "allele_freq_scatter.png", dpi=300)
    plt.close()


# =============================================================================
# LD tagging diagnostics
# =============================================================================
def _pop_id_to_name(ts: tskit.TreeSequence) -> Dict[int, str]:
    out = {}
    for i in range(ts.num_populations):
        md = ts.population(i).metadata
        if isinstance(md, dict) and "name" in md:
            out[i] = md["name"]
        else:
            out[i] = f"pop{i}"
    return out


def _get_pop_sample_indices(ts: tskit.TreeSequence, pop_name: str) -> np.ndarray:
    """
    Return indices into ts.samples() corresponding to sample nodes in the given population.
    (var.genotypes aligns with ts.samples() ordering)
    """
    pid2name = _pop_id_to_name(ts)
    sample_nodes = ts.samples()

    idx = []
    for j, node_id in enumerate(sample_nodes):
        pid = ts.node(node_id).population
        if pid2name[pid] == pop_name:
            idx.append(j)

    if len(idx) == 0:
        raise ValueError(f"No samples found for population {pop_name}. Available pops: {sorted(set(pid2name.values()))}")
    return np.array(idx, dtype=int)


def _variants_in_window(ts: tskit.TreeSequence, left: float, right: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (site_ids, positions, G) for variants with positions in [left, right).
    G has shape (m_variants, n_samples) with 0/1 allele-index genotypes for haploid samples.
    """
    site_ids = []
    positions = []
    rows = []
    for var in ts.variants():
        pos = float(var.site.position)
        if left <= pos < right:
            site_ids.append(int(var.site.id))
            positions.append(pos)
            rows.append(var.genotypes.copy())

    if len(rows) == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.zeros((0, ts.num_samples), dtype=np.int8)

    return np.array(site_ids, dtype=int), np.array(positions, dtype=float), np.vstack(rows)


def r2_with_causal(ts: tskit.TreeSequence, causal_site_id: int, window_bp: int, pop_name: str) -> pd.DataFrame:
    """
    Compute r^2 between a causal SNP and all SNPs within +-window_bp in the given population.
    Uses Pearson correlation squared on haploid 0/1 genotypes across sample nodes in that population.
    """
    causal_site = ts.site(int(causal_site_id))
    cpos = float(causal_site.position)
    left = max(0.0, cpos - window_bp)
    right = cpos + window_bp

    site_ids, positions, G = _variants_in_window(ts, left, right)
    if G.shape[0] == 0:
        return pd.DataFrame(columns=["site_id", "position", "distance", "r2", "pop", "causal_site_id", "causal_pos"])

    idx = _get_pop_sample_indices(ts, pop_name)
    Gp = G[:, idx].astype(float)

    matches = np.where(site_ids == int(causal_site_id))[0]
    if len(matches) == 0:
        return pd.DataFrame(columns=["site_id", "position", "distance", "r2", "pop", "causal_site_id", "causal_pos"])
    c_row = int(matches[0])

    c = Gp[c_row, :]
    if np.var(c) == 0:
        return pd.DataFrame(columns=["site_id", "position", "distance", "r2", "pop", "causal_site_id", "causal_pos"])

    c_center = c - c.mean()
    c_denom = np.sqrt(np.sum(c_center**2))

    r2 = np.empty(Gp.shape[0], dtype=float)
    for i in range(Gp.shape[0]):
        g = Gp[i, :]
        g_center = g - g.mean()
        g_denom = np.sqrt(np.sum(g_center**2))
        if g_denom == 0:
            r2[i] = np.nan
            continue
        r = float(np.sum(c_center * g_center) / (c_denom * g_denom))
        r2[i] = r * r

    out = pd.DataFrame(
        {
            "site_id": site_ids,
            "position": positions,
            "distance": positions - cpos,
            "r2": r2,
            "pop": pop_name,
            "causal_site_id": int(causal_site_id),
            "causal_pos": cpos,
        }
    )
    out = out[out["site_id"] != int(causal_site_id)].copy()
    out = out[np.isfinite(out["r2"])].copy()
    return out


def plot_ld_tagging(df_yri: pd.DataFrame, df_ceu: pd.DataFrame, causal_site_id: int, outdir: Path, r2_thresh: float) -> Path:
    outpath = outdir / f"ld_tagging_causal_site_{int(causal_site_id)}.png"

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(df_yri["distance"], df_yri["r2"], s=6, alpha=0.5)
    plt.axhline(r2_thresh, linestyle="--", linewidth=1)
    plt.title(f"YRI: r² vs distance\ncausal site {int(causal_site_id)}")
    plt.xlabel("distance (bp)")
    plt.ylabel("r²")

    plt.subplot(1, 2, 2)
    plt.scatter(df_ceu["distance"], df_ceu["r2"], s=6, alpha=0.5)
    plt.axhline(r2_thresh, linestyle="--", linewidth=1)
    plt.title(f"CEU: r² vs distance\ncausal site {int(causal_site_id)}")
    plt.xlabel("distance (bp)")
    plt.ylabel("r²")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    return outpath


def compare_tags_for_causal(
    ts: tskit.TreeSequence,
    causal_site_id: int,
    *,
    window_bp: int = 50_000,
    r2_thresh: float = 0.2,
    top_k: int = 10,
    outdir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[int], Set[int], Set[int]]:
    df_yri = r2_with_causal(ts, int(causal_site_id), window_bp, "YRI")
    df_ceu = r2_with_causal(ts, int(causal_site_id), window_bp, "CEU")

    tags_yri = set(df_yri.loc[df_yri["r2"] >= r2_thresh, "site_id"].astype(int).tolist())
    tags_ceu = set(df_ceu.loc[df_ceu["r2"] >= r2_thresh, "site_id"].astype(int).tolist())
    overlap = tags_yri & tags_ceu

    print(f"\n=== LD tagging around causal site {int(causal_site_id)} (±{window_bp} bp, tag r2≥{r2_thresh}) ===")
    print(f"YRI tags: {len(tags_yri)} | CEU tags: {len(tags_ceu)} | overlap: {len(overlap)}")

    print("\nTop tags in YRI:")
    if len(df_yri) == 0:
        print("  (no variants / monomorphic causal in YRI)")
    else:
        print(df_yri.sort_values("r2", ascending=False).head(top_k)[["site_id", "distance", "r2"]].to_string(index=False))

    print("\nTop tags in CEU:")
    if len(df_ceu) == 0:
        print("  (no variants / monomorphic causal in CEU)")
    else:
        print(df_ceu.sort_values("r2", ascending=False).head(top_k)[["site_id", "distance", "r2"]].to_string(index=False))

    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        p = plot_ld_tagging(df_yri, df_ceu, int(causal_site_id), outdir, r2_thresh)
        print(f"Saved LD tagging plot: {p}")

        if len(df_yri) > 0:
            df_yri.to_csv(outdir / f"ld_r2_table_YRI_causal_site_{int(causal_site_id)}.csv", index=False)
        if len(df_ceu) > 0:
            df_ceu.to_csv(outdir / f"ld_r2_table_CEU_causal_site_{int(causal_site_id)}.csv", index=False)

    return df_yri, df_ceu, tags_yri, tags_ceu, overlap


# =============================================================================
# NEW: choose causal sites polymorphic in both pops (so LD is defined in both)
# =============================================================================
def choose_polymorphic_causal_sites(
    ts: tskit.TreeSequence,
    trait_df: pd.DataFrame,
    popA: str,
    popB: str,
    *,
    n_sites: int = 5,
    min_maf: float = 0.01,
    max_tries: int = 200,
    seed: int = 0,
) -> List[int]:
    """
    Return up to n_sites causal site_ids that are polymorphic in BOTH populations with MAF >= min_maf.

    This avoids the "monomorphic causal in CEU" issue, so LD comparisons are meaningful.
    """
    if "site_id" not in trait_df.columns:
        return []

    rng = np.random.default_rng(seed)

    pops = pop_sample_nodes(ts)
    if popA not in pops or popB not in pops:
        raise ValueError(f"Expected populations '{popA}' and '{popB}'. Found: {list(pops.keys())}")

    idxA = _pop_sample_indices_from_nodes(ts, pops[popA])
    idxB = _pop_sample_indices_from_nodes(ts, pops[popB])

    causal_ids = trait_df["site_id"].astype(int).unique().tolist()
    if len(causal_ids) == 0:
        return []

    causal_set = set(causal_ids)

    # helper: compute p for one site quickly using variant access by iterating once
    # We'll do a single pass over variants and record allele freqs for causal sites only.
    pA = {}
    pB = {}
    for var in ts.variants():
        sid = int(var.site.id)
        if sid not in causal_set:
            continue
        g = var.genotypes
        pA[sid] = float(np.mean(g[idxA] == 1))
        pB[sid] = float(np.mean(g[idxB] == 1))

    # filter by MAF in both pops
    good = []
    for sid in causal_ids:
        if sid not in pA or sid not in pB:
            continue
        pa, pb = pA[sid], pB[sid]
        maf_a = min(pa, 1 - pa)
        maf_b = min(pb, 1 - pb)
        if maf_a >= min_maf and maf_b >= min_maf:
            good.append(sid)

    if len(good) == 0:
        print(
            f"\n[warn] No causal sites found with MAF>={min_maf} in BOTH {popA} and {popB}. "
            f"Try lowering min_maf (e.g. 0.001) or increasing num_samples / num_causal / genome_length."
        )
        return []

    # choose up to n_sites, random subset for variety
    rng.shuffle(good)
    return good[: min(n_sites, len(good))]


# =============================================================================
# Main
# =============================================================================
def main():
    output_dir = Path("simulated_data")
    output_dir.mkdir(exist_ok=True)

    # Simulation parameters
    num_samples = {"YRI": 100, "CEU": 100}
    seed = 295
    L = 1e6
    mu = 1e-8
    r = 1e-8

    sampled_params = {
        "N_anc": 10000,
        "N_YRI": 15000,
        "N_CEU": 12000,
        "m": 0,  # toggle this
        "T_split": 2000,
    }
    g = IM_symmetric_model(sampled_params)
    ts = simulation_runner(g, L, mu, r, num_samples, seed)

    # Demes plot
    ax = demesdraw.tubes(g)
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("N")
    ax.figure.savefig(output_dir / "demes_plot.png", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # Save tree sequence
    ts_path = output_dir / "simulated.trees"
    ts.dump(ts_path)
    print(f"Simulated tree sequence saved to: {ts_path}")

    # Trait simulation
    model = tstrait.trait_model(distribution="normal", mean=0, var=1)
    sim_result = tstrait.sim_phenotype(ts=ts, num_causal=100, model=model, h2=0.3, random_seed=1)

    trait_df = sim_result.trait
    print(trait_df.head())
    trait_path = output_dir / "simulated_traits.csv"
    trait_df.to_csv(trait_path, index=False)
    print(f"Simulated traits saved to: {trait_path}")

    # Phenotypes (+ population labels)
    phenotype_df = sim_result.phenotype
    phenotype_df = add_population_column(ts, phenotype_df)
    print(phenotype_df.head())
    phenotype_path = output_dir / "simulated_phenotypes.csv"
    phenotype_df.to_csv(phenotype_path, index=False)
    print(f"Simulated phenotypes saved to: {phenotype_path}")

    # Phenotype / noise plots
    plt.figure()
    plt.hist(phenotype_df["phenotype"], bins=40)
    plt.title("Phenotype")
    plt.tight_layout()
    plt.savefig(output_dir / "phenotype_hist.png", dpi=300)
    plt.close()

    plt.figure()
    plt.hist(phenotype_df["environmental_noise"], bins=40)
    plt.title("Environmental Noise")
    plt.tight_layout()
    plt.savefig(output_dir / "environmental_noise_hist.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Allele-frequency differences between populations (all SNPs + causal)
    # -------------------------------------------------------------------------
    popA, popB = "YRI", "CEU"
    af_df = allele_freqs_two_pops(ts, popA, popB)

    af_df["is_causal"] = False
    if "site_id" in trait_df.columns:
        af_df["is_causal"] = af_df["site_id"].isin(trait_df["site_id"].astype(int))

    af_path = output_dir / "allele_freqs_by_pop.csv"
    af_df.to_csv(af_path, index=False)
    print(f"Allele freqs saved to: {af_path}")

    plot_af_diagnostics(af_df, popA, popB, output_dir)
    print("Saved AF plots:",
          output_dir / "allele_freq_delta_p_hist.png",
          output_dir / "allele_freq_scatter.png")

    print("\n=== AF summaries ===")
    print("delta_p mean:", af_df["delta_p"].mean())
    print("delta_p median:", af_df["delta_p"].median())
    print("delta_p 95th pct:", np.quantile(af_df["delta_p"], 0.95))
    if af_df["is_causal"].any():
        print("causal delta_p mean:", af_df.loc[af_df["is_causal"], "delta_p"].mean())
        print("non-causal delta_p mean:", af_df.loc[~af_df["is_causal"], "delta_p"].mean())

    # -------------------------------------------------------------------------
    # LD tagging differences around causal SNPs (FILTERED: polymorphic in BOTH pops)
    # -------------------------------------------------------------------------
    ld_outdir = output_dir / "ld_tagging"
    window_bp = 50_000
    r2_thresh = 0.2
    top_k = 10
    n_causal_to_check = 5
    min_maf = 0.01  # require at least 1% MAF in BOTH pops so LD is meaningful

    chosen = choose_polymorphic_causal_sites(
        ts,
        trait_df,
        popA,
        popB,
        n_sites=n_causal_to_check,
        min_maf=min_maf,
        seed=0,
    )

    if len(chosen) == 0:
        print(
            "\n[warn] Could not find enough causal sites polymorphic in BOTH pops at the chosen min_maf. "
            "Try lowering min_maf or increasing sample size / genome length / num_causal."
        )
    else:
        print(f"\nChecking LD tagging around {len(chosen)} causal sites polymorphic in BOTH pops (min_maf={min_maf}): {chosen}")
        for sid in chosen:
            compare_tags_for_causal(
                ts,
                sid,
                window_bp=window_bp,
                r2_thresh=r2_thresh,
                top_k=top_k,
                outdir=ld_outdir,
            )


if __name__ == "__main__":
    main()