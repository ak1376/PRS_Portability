import json, sys
from pathlib import Path
from snakemake.io import protected

SIM_SCRIPT = "snakemake_scripts/simulation.py"
EXP_CFG    = "config_files/experiment_config_ooa.json"

# Experiment metadata
CFG   = json.loads(Path(EXP_CFG).read_text())
MODEL = CFG["demographic_model"]

grid_cfg = CFG.get("grid_sampling", {})
if grid_cfg.get("enabled", False):
    NUM_GRID = int(grid_cfg.get("num_grid_points", 1))
    REPS = int(grid_cfg.get("replicates_per_value", 1))
else:
    NUM_GRID = int(CFG["num_draws"])
    REPS = int(CFG.get("num_replicates", 1))

# Wildcard ranges
SID_RANGE = range(NUM_GRID)         # sid = 0, 1, ..., NUM_GRID-1 (parameter configs)
REP_RANGE = range(REPS)             # rep = 0, 1, ..., REPS-1 (replicates per config)

SIM_BASEDIR  = f"experiments/{MODEL}/simulations"
GWAS_BASEDIR = f"experiments/{MODEL}/gwas"


##############################################################################
# RULE all – final targets                                                   #
##############################################################################
rule all:
    input:
        # Simulation artefacts for *every* sid × rep
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/sampled_params.pkl",  sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/SFS.pkl",             sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/tree_sequence.trees", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/demes.png",           sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/effect_sizes.pkl",    sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/phenotype.pkl",       sid=SID_RANGE, rep=REP_RANGE),
        # GWAS results for every sid × rep
        expand(f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_results.csv",   sid=SID_RANGE, rep=REP_RANGE),


##############################################################################
# RULE simulate – one sim (demography index = sid, replicate = rep)         #
##############################################################################
rule simulate:
    output:
        sfs       = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/SFS.pkl",
        params    = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/sampled_params.pkl",
        tree      = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/tree_sequence.trees",
        fig       = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/demes.png",
        meta      = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/bgs.meta.json",
        trait     = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/effect_sizes.pkl",
        phenotype = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/phenotype.pkl",
        done      = protected(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/.done"),
    params:
        sim_dir = SIM_BASEDIR,   # <- BASE dir, no sid/rep here
        cfg     = EXP_CFG,
        model   = MODEL
    threads: 1
    shell:
        r"""
        set -euo pipefail

        python "{SIM_SCRIPT}" \
          --simulation-dir "{SIM_BASEDIR}/{wildcards.sid}/rep{wildcards.rep}" \
          --experiment-config "{params.cfg}" \
          --model-type "{params.model}" \
          --simulation-number {wildcards.sid} \
          --replicate {wildcards.rep} \
          --output-dir "{SIM_BASEDIR}/{wildcards.sid}/rep{wildcards.rep}"

        # ensure expected outputs exist, then create sentinel
        test -f "{output.meta}"
        touch "{output.done}"
        """


##############################################################################
# RULE gwas – run GWAS for each sid × rep                                   #
##############################################################################
rule gwas:
    input:
        tree   = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/tree_sequence.trees",
        pheno  = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/phenotype.pkl",
        trait  = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/effect_sizes.pkl"
    output:
        csv       = f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_results.csv",
        manhattan = f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_manhattan.png",
        qq        = f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_qq.png",
        af_diff   = f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_af_diff.png"
    params:
        out_prefix    = f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas",
        discovery_pop = "CEU"
    shell:
        """
        python snakemake_scripts/run_gwas.py \
            --genotype "{input.tree}" \
            --phenotype "{input.pheno}" \
            --trait "{input.trait}" \
            --output-prefix "{params.out_prefix}" \
            --discovery-pop "{params.discovery_pop}"
        """
