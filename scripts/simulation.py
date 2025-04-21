import argparse
import stdpopsim
import demesdraw
import tstrait
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

def main(args):
    ##############################################
    # 1) Simulate variation with stdpopsim
    ##############################################
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("OutOfAfrica_2T12")

    # Visualize demography
    fig, ax = plt.subplots()
    demesdraw.tubes(model.model.to_demes(), ax=ax)
    fig.savefig(args.demography_fig)
    plt.close(fig)

    # Sample diploids from AFR and EUR
    samples = {"AFR": args.n_individuals, "EUR": args.n_individuals}
    contig = species.get_contig(length=args.chrom_length, mutation_rate=model.mutation_rate)
    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(model, contig, samples)

    print("Number of segregating sites:", ts.num_sites)
    print("Number of diploid individuals:", ts.num_individuals)

    # Save the tree sequence for downstream use
    ts.dump(args.output_ts)

    ##############################################
    # 2) Simulate phenotypes with tstrait
    ##############################################
    trait_model = tstrait.trait_model(distribution="normal", mean=0, var=5)
    sim_result = tstrait.sim_phenotype(
        ts=ts,
        num_causal=args.num_causal,
        model=trait_model,
        h2=args.h2,
        random_seed=295
    )
    trait_df = sim_result.trait
    trait_df.to_csv(args.output_trait_df, index=False)
    phenotype_df = sim_result.phenotype

    ##############################################
    # 3) Map individuals to populations
    ##############################################
    pop_mapping = []
    for ind in ts.individuals():
        pop_id = ts.population(ind.population).metadata["id"]
        pop_mapping.append({"individual_id": ind.id, "population": pop_id})

    pop_df = pd.DataFrame(pop_mapping)
    merged = phenotype_df.merge(pop_df, on="individual_id")

    # Add phenotype shift to EUR
    eur_mask = (merged["population"] == "EUR")
    merged.loc[eur_mask, "phenotype"] += 2.0

    # Sort for alignment
    merged_sorted = merged.sort_values(by="individual_id")
    merged_sorted.to_csv(args.output_csv, index=False)

    ##############################################
    # 4) Save overall effect size histogram
    ##############################################
    plt.figure()
    plt.hist(trait_df["effect_size"], bins=30)
    plt.title("Distribution of Effect Sizes")
    plt.xlabel("Effect Size")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.effect_size_fig)
    plt.close()

    ##############################################
    # 5) Save population-colored phenotype histogram
    ##############################################
    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=merged,
        x="phenotype",
        hue="population",
        element="step",
        stat="count",
        bins=30,
        common_norm=False
    )
    plt.title("Phenotype Distribution by Population")
    plt.xlabel("Phenotype")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.phenotype_by_pop_fig)
    plt.close()

    print(f"Saved outputs to:\n"
          f"  - {args.output_ts}\n"
          f"  - {args.demography_fig}\n"
          f"  - {args.effect_size_fig}\n"
          f"  - {args.phenotype_by_pop_fig}\n"
          f"  - {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate phenotypes and demography with stdpopsim + tstrait")
    parser.add_argument("--n_individuals", type=int, required=True)
    parser.add_argument("--num_causal", type=int, required=True)
    parser.add_argument("--h2", type=float, required=True)
    parser.add_argument("--chrom_length", type=float, required=True)
    parser.add_argument("--output_ts", type=str, required=True)
    parser.add_argument("--demography_fig", type=str, required=True)
    parser.add_argument("--effect_size_fig", type=str, required=True)
    parser.add_argument("--phenotype_by_pop_fig", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_trait_df", type=str, required=True)

    args = parser.parse_args()
    main(args)
