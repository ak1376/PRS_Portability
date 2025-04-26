"""
Simulate variation (stdpopsim) ➜ phenotypes (tstrait)
with biallelic-only SNPs and NaN safety checks.
"""
from __future__ import annotations
import argparse, stdpopsim, demesdraw, tstrait
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_style("whitegrid")


def main(a: argparse.Namespace) -> None:
    # ───────── 1) simulate genetic variation ─────────
    sp     = stdpopsim.get_species("HomSap")
    model  = sp.get_demographic_model("OutOfAfrica_2T12")

    fig, ax = plt.subplots()
    demesdraw.tubes(model.model.to_demes(), ax=ax)
    fig.savefig(a.demography_fig); plt.close(fig)

    samples = {"AFR": a.n_individuals, "EUR": a.n_individuals}
    contig  = sp.get_contig(length=a.chrom_length,
                            mutation_rate=model.mutation_rate)
    ts      = stdpopsim.get_engine("msprime").simulate(model, contig, samples)

    # keep *only* biallelic sites (works on all tskit versions)
    bial   = [i for i, s in enumerate(ts.sites()) if len(s.alleles) == 2]
    non_bi = sorted(set(range(ts.num_sites)) - set(bial))
    if non_bi:                      # safe-guard: only delete if needed
        ts = ts.delete_sites(non_bi)

    ts.dump(a.output_ts)
    print(f"Segregating sites   : {ts.num_sites:,}")
    print(f"Diploid individuals: {ts.num_individuals:,}")

    # ───────── 2) simulate quantitative trait ─────────
    trait_mod = tstrait.trait_model(distribution="normal", mean=0, var=5)
    sim = tstrait.sim_phenotype(
        ts=ts,                    # ← still positional is fine, but explicit works too
        num_causal=a.num_causal,
        model=trait_mod,
        h2=a.h2,
        random_seed=295
    )

    trait_df  = sim.trait
    pheno_df  = sim.phenotype

    # --- NaN check #1 ---------------------------------------------------------
    n0 = pheno_df.phenotype.isna().sum()
    print(f"NaNs from tstrait        : {n0}")
    if n0:
        fill = pheno_df.phenotype.mean()
        pheno_df.phenotype.fillna(fill, inplace=True)
        print(f"  filled with mean = {fill:.4f}")

    trait_df.to_csv(a.output_trait_df, index=False)

    # ───────── 3) attach population labels ─────────
    pop_df = pd.DataFrame({
        "individual_id": [ind.id for ind in ts.individuals()],
        "population"   : [ts.population(ind.population).metadata["id"]
                          for ind in ts.individuals()]
    })
    merged = pop_df.merge(pheno_df, on="individual_id", how="left")

    # --- NaN check #2 ---------------------------------------------------------
    n1 = merged.phenotype.isna().sum()
    print(f"NaNs after merge        : {n1}")

    # environmental shift
    merged.loc[merged.population == "EUR", "phenotype"] += 2.0

    merged.sort_values("individual_id").to_csv(
        a.output_csv, index=False, float_format="%.8g")

    # --- NaN check #3 ---------------------------------------------------------
    n2 = pd.read_csv(a.output_csv).phenotype.isna().sum()
    print(f"NaNs after CSV reload   : {n2}")
    assert n2 == 0, "Phenotype NaNs survived!"

    # ───────── 4) quick QC plots ─────────
    plt.figure(); plt.hist(trait_df.effect_size, 30, color="#1f77b4")
    plt.title("Distribution of causal effect sizes")
    plt.tight_layout(); plt.savefig(a.effect_size_fig); plt.close()

    plt.figure(figsize=(8,6))
    sns.histplot(merged, x="phenotype", hue="population",
                 bins=30, element="step", stat="count", common_norm=False)
    plt.title("Phenotype distribution by population")
    plt.tight_layout(); plt.savefig(a.phenotype_by_pop_fig); plt.close()

    print("\nOutputs:")
    for f in [a.output_ts, a.demography_fig, a.effect_size_fig,
              a.phenotype_by_pop_fig, a.output_csv, a.output_trait_df]:
        print(" •", f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_individuals", type=int, required=True)
    p.add_argument("--num_causal",   type=int, required=True)
    p.add_argument("--h2",           type=float, required=True)
    p.add_argument("--chrom_length", type=float, required=True)
    p.add_argument("--output_ts",           required=True)
    p.add_argument("--demography_fig",      required=True)
    p.add_argument("--effect_size_fig",     required=True)
    p.add_argument("--phenotype_by_pop_fig",required=True)
    p.add_argument("--output_csv",          required=True)
    p.add_argument("--output_trait_df",     required=True)
    main(p.parse_args())
