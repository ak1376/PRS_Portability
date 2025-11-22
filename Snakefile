import json, math, sys, os
from pathlib import Path
from snakemake.io import protected

SIM_SCRIPT   = "snakemake_scripts/simulation.py"
EXP_CFG      = "config_files/experiment_config_split_migration.json"

# Experiment metadata
CFG           = json.loads(Path(EXP_CFG).read_text())
MODEL         = CFG["demographic_model"]
NUM_DRAWS     = int(CFG["num_draws"])

SIM_BASEDIR = f"experiments/{MODEL}/simulations"                      # per‑sim artefacts
GWAS_BASEDIR = f"experiments/{MODEL}/gwas"                             # gwas results
SIM_IDS     = list(range(NUM_DRAWS))  # Generate simulation IDs based on num_draws


##############################################################################
# RULE all – final targets the workflow must create                          #
##############################################################################
rule all:
    input:
        # Simulation artifacts
        expand(f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",  sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",             sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees", sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/demes.png",           sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/effect_sizes.pkl",           sid=SIM_IDS),
        expand(f"{SIM_BASEDIR}/{{sid}}/phenotype.pkl",           sid=SIM_IDS),
        # GWAS results
        expand(f"{GWAS_BASEDIR}/{{sid}}/gwas_results.csv",          sid=SIM_IDS),
        
        


##############################################################################
# RULE simulate – one complete tree‑sequence + SFS
##############################################################################
rule simulate:
    output:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        tree   = f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees",
        fig    = f"{SIM_BASEDIR}/{{sid}}/demes.png",
        meta   = f"{SIM_BASEDIR}/{{sid}}/bgs.meta.json",
        trait  = f"{SIM_BASEDIR}/{{sid}}/effect_sizes.pkl",
        phenotype = f"{SIM_BASEDIR}/{{sid}}/phenotype.pkl",
        done   = protected(f"{SIM_BASEDIR}/{{sid}}/.done"),
    params:
        sim_dir = SIM_BASEDIR,
        cfg     = EXP_CFG,
        model   = MODEL
    threads: 1
    shell:
        r"""
        set -euo pipefail

        python "{SIM_SCRIPT}" \
          --simulation-dir "{params.sim_dir}" \
          --experiment-config "{params.cfg}" \
          --model-type "{params.model}" \
          --simulation-number {wildcards.sid}

        # ensure expected outputs exist, then create sentinel
        test -f "{output.meta}"
        touch "{output.done}"
        """


##############################################################################
# RULE gwas – run GWAS on simulated data
##############################################################################
rule gwas:
    input:
        tree   = f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees",
        pheno  = f"{SIM_BASEDIR}/{{sid}}/phenotype.pkl",
        trait  = f"{SIM_BASEDIR}/{{sid}}/effect_sizes.pkl"
    output:
        csv       = f"{GWAS_BASEDIR}/{{sid}}/gwas_results.csv",
        manhattan = f"{GWAS_BASEDIR}/{{sid}}/gwas_manhattan.png",
        qq        = f"{GWAS_BASEDIR}/{{sid}}/gwas_qq.png",
        af_diff   = f"{GWAS_BASEDIR}/{{sid}}/gwas_af_diff.png"
    params:
        out_prefix = f"{GWAS_BASEDIR}/{{sid}}/gwas"
    shell:
        """
        python snakemake_scripts/run_gwas.py \
            --genotype "{input.tree}" \
            --phenotype "{input.pheno}" \
            --trait "{input.trait}" \
            --output-prefix "{params.out_prefix}"
        """
