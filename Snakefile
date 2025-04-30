# ================================================================
#  Snakemake workflow for PRS‑VAE experiments (one YAML per run)
#  --------------------------------------------------------------
#  • Pass a single YAML file that defines model hyper‑parameters
#    plus a unique `exp_tag`.  Example call:
#        snakemake -j 8 --configfile myrun.yaml
#
#  • All VAE artefacts are written to:
#        results/<exp_tag>/vae/
#    so multiple runs never overwrite each other.
#
#  • The YAML you used is copied into that same folder so the run
#    is fully self‑documenting.
# ================================================================

###############################################################################
#  0 — path to the config file (overridden by --configfile at CLI)            #
###############################################################################
# If you *always* call Snakemake with --configfile <file>, you can delete
# the next line.  It only provides a default.
configfile: "experiment.yaml"

###############################################################################
#  1 — global variables derived from the YAML                                  #
###############################################################################

TAG      = config["exp_tag"]              # e.g. "lam4"
EPOCHS   = int(config.get("epochs", 10))  # fallback if key missing
OUTDIR   = f"results/{TAG}/vae"            # experiment‑specific output root

DATA_DIR = "results/data"
FIGS_DIR = "results/figs"

###############################################################################
#  2 — rule all (lists everything we expect when workflow finishes)           #
###############################################################################

rule all:
    input:
        # -------- once‑per‑dataset artefacts (never change) -----------------
        f"{DATA_DIR}/simulation.ts",
        f"{DATA_DIR}/merged_sorted.csv",
        f"{FIGS_DIR}/demography.png",
        f"{FIGS_DIR}/effect_sizes.png",
        f"{FIGS_DIR}/phenotype_by_population.png",
        f"{DATA_DIR}/trait_df.csv",
        f"{DATA_DIR}/genotype_matrix.npy",
        # -------- GWAS outputs (unchanged) ----------------------------------
        "results/gwas/gwas_results.csv",
        "results/gwas/manhattan_plot.png",
        "results/gwas/af_diff_plot.png",
        "results/gwas/qq_plot.png",
        "results/gwas_pcs/gwas_results.csv",
        "results/gwas_pcs/manhattan_plot.png",
        "results/gwas_pcs/af_diff_plot.png",
        "results/gwas_pcs/qq_plot.png",
        "results/gwas_pcs/scree_plot.png",
        "results/gwas_lmm/gwas_results.csv",
        "results/gwas_lmm/manhattan_plot.png",
        "results/gwas_lmm/af_diff_plot.png",
        "results/gwas_lmm/qq_plot.png",
        # -------- VAE artefacts for THIS experiment -------------------------
        f"{OUTDIR}/vae_epoch{EPOCHS-1}.ckpt",
        f"{OUTDIR}/recon_loss_curves.png",
        f"{OUTDIR}/phenotype_loss_curves.png",
        f"{OUTDIR}/cohort_adv_loss_curves.png",
        f"{OUTDIR}/pop_acc_curve.png",
        f"{OUTDIR}/pheno_scatter_all.png",
        f"{OUTDIR}/latent_pca.png",
        f"{OUTDIR}/cohort_confusion.png",
        f"{OUTDIR}/config.yaml",          # ← copy of the exact YAML used

###############################################################################
#  3 — (unchanged) rules to generate data & GWAS                              #
###############################################################################
# --- ONLY the path constants were parameterised with DATA_DIR / FIGS_DIR ----

rule simulate_trait_data:
    output:
        output_ts              = f"{DATA_DIR}/simulation.ts",
        demography_fig         = f"{FIGS_DIR}/demography.png",
        effect_size_fig        = f"{FIGS_DIR}/effect_sizes.png",
        phenotype_by_pop_fig   = f"{FIGS_DIR}/phenotype_by_population.png",
        output_csv             = f"{DATA_DIR}/merged_sorted.csv",
        output_trait_df        = f"{DATA_DIR}/trait_df.csv"
    params:
        n_individuals = 500,
        num_causal    = 100,
        h2            = 0.7,
        chrom_length  = 5e6
    shell:
        """
        python scripts/simulation.py \
            --n_individuals {params.n_individuals} \
            --num_causal   {params.num_causal} \
            --h2           {params.h2} \
            --chrom_length {params.chrom_length} \
            --output_ts            {output.output_ts} \
            --demography_fig       {output.demography_fig} \
            --effect_size_fig      {output.effect_size_fig} \
            --phenotype_by_pop_fig {output.phenotype_by_pop_fig} \
            --output_csv           {output.output_csv} \
            --output_trait_df      {output.output_trait_df}
        """

rule build_genotype_matrix:
    input:
        ts_file    = f"{DATA_DIR}/simulation.ts",
        sorted_csv = f"{DATA_DIR}/merged_sorted.csv"
    output:
        f"{DATA_DIR}/genotype_matrix.npy"
    shell:
        """
        python scripts/build_genotype_matrix.py \
            --ts_file {input.ts_file} \
            --sorted_individuals_csv {input.sorted_csv} \
            --output_npy {output}
        """

# ---------------------------- GWAS rules ----------------------------
#  Exactly as in your original workflow but using DATA_DIR / FIGS_DIR
#  so that upstream file locations are centralised.

rule naive_gwas:
    input:
        genotype_matrix = f"{DATA_DIR}/genotype_matrix.npy",
        sorted_phenotype = f"{DATA_DIR}/merged_sorted.csv",
        trait_info = f"{DATA_DIR}/trait_df.csv"
    output:
        output_csv      = "results/gwas/gwas_results.csv",
        manhattan_plot  = "results/gwas/manhattan_plot.png",
        af_diff_plot    = "results/gwas/af_diff_plot.png",
        qq_plot         = "results/gwas/qq_plot.png"
    params:
        discovery_pop = "EUR"
    shell:
        """
        python scripts/naive_gwas.py \
            --genotype_matrix {input.genotype_matrix} \
            --sorted_phenotype {input.sorted_phenotype} \
            --trait_info {input.trait_info} \
            --output_csv {output.output_csv} \
            --manhattan_plot {output.manhattan_plot} \
            --af_diff_plot {output.af_diff_plot} \
            --qq_plot {output.qq_plot} \
            --discovery_pop {params.discovery_pop}
        """

rule gwas_with_pcs:
    input:
        genotype_matrix = f"{DATA_DIR}/genotype_matrix.npy",
        sorted_phenotype = f"{DATA_DIR}/merged_sorted.csv",
        trait_info = f"{DATA_DIR}/trait_df.csv"
    output:
        output_csv      = "results/gwas_pcs/gwas_results.csv",
        manhattan_plot  = "results/gwas_pcs/manhattan_plot.png",
        af_diff_plot    = "results/gwas_pcs/af_diff_plot.png",
        qq_plot         = "results/gwas_pcs/qq_plot.png",
        scree_plot      = "results/gwas_pcs/scree_plot.png"
    params:
        num_pcs = 5,
        discovery_pop = "EUR"
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
            --scree_plot {output.scree_plot} \
            --discovery_pop {params.discovery_pop}
        """

rule gwas_lmm:
    input:
        genotype_matrix = f"{DATA_DIR}/genotype_matrix.npy",
        sorted_phenotype = f"{DATA_DIR}/merged_sorted.csv",
        trait_info = f"{DATA_DIR}/trait_df.csv"
    output:
        output_csv      = "results/gwas_lmm/gwas_results.csv",
        manhattan_plot  = "results/gwas_lmm/manhattan_plot.png",
        af_diff_plot    = "results/gwas_lmm/af_diff_plot.png",
        qq_plot         = "results/gwas_lmm/qq_plot.png"
    params:
        discovery_pop = "EUR"
    shell:
        """
        python scripts/gwas_lmm.py \
            --genotype_matrix {input.genotype_matrix} \
            --sorted_phenotype {input.sorted_phenotype} \
            --trait_info {input.trait_info} \
            --output_csv {output.output_csv} \
            --manhattan_plot {output.manhattan_plot} \
            --af_diff_plot {output.af_diff_plot} \
            --qq_plot {output.qq_plot} \
            --discovery_pop {params.discovery_pop}
        """

###############################################################################
#  4 — Copy YAML into the experiment folder                                   #                                   #
###############################################################################

# Determine which YAML was supplied on the CLI.  If none, fall back to
# the default "experiment.yaml" declared at the top.
try:
    CFG_PATH = workflow.overwrite_configfiles[0]
except IndexError:
    CFG_PATH = "experiment.yaml"

rule store_config:
    input:
        yaml_in = CFG_PATH
    output:
        yaml_copy = f"{OUTDIR}/config.yaml"
    shell:
        """
        mkdir -p {OUTDIR}
        cp {input.yaml_in} {output.yaml_copy}
        """

###############################################################################
#  5 — VAE training (uses params from YAML)                                   #
###############################################################################

rule train_vae:
    input:
        genotype_matrix = f"{DATA_DIR}/genotype_matrix.npy",
        meta            = f"{DATA_DIR}/merged_sorted.csv",
        cfg             = rules.store_config.output.yaml_copy
    output:
        ckpt            = f"{OUTDIR}/vae_epoch{EPOCHS-1}.ckpt",
        recon_curves    = f"{OUTDIR}/recon_loss_curves.png",
        phen_curves     = f"{OUTDIR}/phenotype_loss_curves.png",
        adv_curves      = f"{OUTDIR}/cohort_adv_loss_curves.png",
        pop_acc_curve   = f"{OUTDIR}/pop_acc_curve.png",
        pheno_scatter   = f"{OUTDIR}/pheno_scatter_all.png",
        latent_pca      = f"{OUTDIR}/latent_pca.png",
        cohort_conf     = f"{OUTDIR}/cohort_confusion.png"
    params:
        latent      = config["latent"],
        hidden      = config["hidden"],
        lr          = config["lr"],
        lambda_adv  = config["lambda_adv"],
        recon_wt    = config.get("recon_wt", 0.1),
        mse_wt      = config.get("mse_wt", 10.0),
        beta_kl     = config.get("beta_kl", 1.0),
        warm_adv    = config.get("warm_adv", 5),
        batch       = config["batch"],
        epochs      = EPOCHS,
        outdir      = OUTDIR
    shell:
        """
        python scripts/genotype_vae.py \
            --genotype   {input.genotype_matrix} \
            --meta       {input.meta} \
            --latent     {params.latent} \
            --hidden     {params.hidden} \
            --lr         {params.lr} \
            --lambda_adv {params.lambda_adv} \
            --recon_wt   {params.recon_wt} \
            --mse_wt     {params.mse_wt} \
            --beta_kl    {params.beta_kl} \
            --warm_adv   {params.warm_adv} \
            --batch      {params.batch} \
            --epochs     {params.epochs} \
            --outdir     {params.outdir}
        """
