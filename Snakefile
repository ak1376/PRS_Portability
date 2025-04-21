rule all:
    input:
        "results/data/simulation.ts",
        "results/data/merged_sorted.csv",
        "results/figs/demography.png",
        "results/figs/effect_sizes.png",
        "results/figs/phenotype_by_population.png",
        "results/data/trait_df.csv",
        "results/data/genotype_matrix.npy",
        "results/gwas/gwas_results.csv",
        "results/gwas/manhattan_plot.png",
        "results/gwas/af_diff_plot.png",
        "results/gwas/qq_plot.png",
        "results/gwas_pcs/gwas_results.csv",
        "results/gwas_pcs/manhattan_plot.png",
        "results/gwas_pcs/af_diff_plot.png",
        "results/gwas_pcs/qq_plot.png",
        "results/gwas_pcs/scree_plot.png"

rule simulate_trait_data:
    output:
        output_ts="results/data/simulation.ts",
        demography_fig="results/figs/demography.png",
        effect_size_fig="results/figs/effect_sizes.png",
        phenotype_by_pop_fig="results/figs/phenotype_by_population.png",
        output_csv="results/data/merged_sorted.csv",
        output_trait_df="results/data/trait_df.csv"
    params:
        n_individuals=500,
        num_causal=50,
        h2=0.8,
        chrom_length=5e6
    shell:
        """
        python scripts/simulation.py \
            --n_individuals {params.n_individuals} \
            --num_causal {params.num_causal} \
            --h2 {params.h2} \
            --chrom_length {params.chrom_length} \
            --output_ts {output.output_ts} \
            --demography_fig {output.demography_fig} \
            --effect_size_fig {output.effect_size_fig} \
            --phenotype_by_pop_fig {output.phenotype_by_pop_fig} \
            --output_csv {output.output_csv} \
            --output_trait_df {output.output_trait_df}
        """

rule build_genotype_matrix:
    input:
        ts_file="results/data/simulation.ts",
        sorted_csv="results/data/merged_sorted.csv"
    output:
        "results/data/genotype_matrix.npy"
    shell:
        """
        python scripts/build_genotype_matrix.py \
            --ts_file {input.ts_file} \
            --sorted_individuals_csv {input.sorted_csv} \
            --output_npy {output}
        """

rule naive_gwas:
    input:
        genotype_matrix="results/data/genotype_matrix.npy",
        sorted_phenotype="results/data/merged_sorted.csv",
        trait_info="results/data/trait_df.csv"
    output:
        output_csv="results/gwas/gwas_results.csv",
        manhattan_plot="results/gwas/manhattan_plot.png",
        af_diff_plot="results/gwas/af_diff_plot.png",
        qq_plot="results/gwas/qq_plot.png"
    shell:
        """
        python scripts/naive_gwas.py \
            --genotype_matrix {input.genotype_matrix} \
            --sorted_phenotype {input.sorted_phenotype} \
            --trait_info {input.trait_info} \
            --output_csv {output.output_csv} \
            --manhattan_plot {output.manhattan_plot} \
            --af_diff_plot {output.af_diff_plot} \
            --qq_plot {output.qq_plot}
        """

rule gwas_with_pcs:
    input:
        genotype_matrix="results/data/genotype_matrix.npy",
        sorted_phenotype="results/data/merged_sorted.csv",
        trait_info="results/data/trait_df.csv"
    output:
        output_csv="results/gwas_pcs/gwas_results.csv",
        manhattan_plot="results/gwas_pcs/manhattan_plot.png",
        af_diff_plot="results/gwas_pcs/af_diff_plot.png",
        qq_plot="results/gwas_pcs/qq_plot.png",
        scree_plot="results/gwas_pcs/scree_plot.png"
    params:
        num_pcs=20
    shell:
        """
        python scripts/gwas_with_pcs.py \
            --genotype_matrix {input.genotype_matrix} \
            --sorted_phenotype {input.sorted_phenotype} \
            --trait_info {input.trait_info} \
            --num_pcs {params.num_pcs} \
            --output_csv {output.output_csv} \
            --manhattan_plot {output.manhattan_plot} \
            --af_diff_plot {output.af_diff_plot} \
            --qq_plot {output.qq_plot} \
            --scree_plot {output.scree_plot}
        """
