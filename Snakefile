# Snakefile
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import yaml

##############################################################################
# VAE YAML grid expansion (reuses your vae.yaml + optional grid section)
##############################################################################
BASE_VAE_YAML = Path("config_files/model_hyperparams/vae.yaml")
BASE_VAE = yaml.safe_load(BASE_VAE_YAML.read_text())

GEN_VAE_DIR = Path("config_files/generated_vae")
GEN_VAE_DIR.mkdir(parents=True, exist_ok=True)


def set_by_path(d: dict, path: str, value):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def tagify(v):
    # floats: 0.005 -> 0p005 ; 10.0 -> 10
    if isinstance(v, float):
        s = f"{v:g}"
        return s.replace(".", "p")
    return str(v)


def exp_yaml_path(exp: str) -> Path:
    return GEN_VAE_DIR / f"{exp}.yaml"


GRID = BASE_VAE.get("grid", {}) or {}
GRID_ENABLED = bool(GRID.get("enabled", False))
DIMS = GRID.get("dims", []) or []

if GRID_ENABLED and len(DIMS) == 0:
    raise ValueError("vae.yaml has grid.enabled=true but grid.dims is empty")


def make_exp_name(assignments):
    name_cfg = GRID.get("name", {}) or {}
    prefix = name_cfg.get("prefix", "vae")
    sep = name_cfg.get("sep", "__")

    parts = []
    for dim, val in assignments:
        tag = dim.get("tag", None)
        pth = dim["path"]
        key = tag if tag else pth.split(".")[-1]
        parts.append(f"{key}{tagify(val)}")
    return prefix + sep + sep.join(parts)


EXP_SPECS: list[tuple[str, list[tuple[dict, object]]]] = []

if GRID_ENABLED:
    values_lists = [dim["values"] for dim in DIMS]
    for combo in itertools.product(*values_lists):
        assignments = list(zip(DIMS, combo))
        exp = make_exp_name(assignments)

        cfg = copy.deepcopy(BASE_VAE)
        cfg.pop("grid", None)

        for dim, val in assignments:
            set_by_path(cfg, dim["path"], val)

        cfg["_generated_from"] = str(BASE_VAE_YAML)
        cfg["_exp_name"] = exp

        exp_yaml_path(exp).write_text(yaml.safe_dump(cfg, sort_keys=False))
        EXP_SPECS.append((exp, assignments))

    EXP_NAMES = [e for e, _ in EXP_SPECS]
else:
    exp = "default"
    cfg = copy.deepcopy(BASE_VAE)
    cfg.pop("grid", None)
    cfg["_generated_from"] = str(BASE_VAE_YAML)
    cfg["_exp_name"] = exp
    exp_yaml_path(exp).write_text(yaml.safe_dump(cfg, sort_keys=False))
    EXP_NAMES = [exp]

##############################################################################
# Scripts + experiment config
##############################################################################
SIM_SCRIPT = "snakemake_scripts/simulation.py"
BUILD_INPUTS = "snakemake_scripts/build_genotypes_for_vae.py"
GWAS_SCRIPT = "snakemake_scripts/run_gwas.py"
TRAIN_VAE_SCRIPT = "snakemake_scripts/train_vae_wrapper.py"
PLOT_WRAPPER = "snakemake_scripts/plot_vae_diagnostics.py"

EXP_CFG = "config_files/experiment_config_IM_symmetric.json"
CFG = json.loads(Path(EXP_CFG).read_text())
MODEL = CFG["demographic_model"]

##############################################################################
# Ranges
##############################################################################
NUM_DRAWS = int(CFG.get("num_draws", 1))
REPS = int(CFG.get("num_replicates", 1))

SID_RANGE = range(NUM_DRAWS)
REP_RANGE = range(REPS)

##############################################################################
# Directories
##############################################################################
SIM_BASEDIR = f"experiments/{MODEL}/simulations"
GENO_BASEDIR = f"experiments/{MODEL}/processed_data"
GWAS_BASEDIR = f"experiments/{MODEL}/gwas"
VAE_BASEDIR = f"experiments/{MODEL}/vae"

Path(SIM_BASEDIR).mkdir(parents=True, exist_ok=True)
Path(GENO_BASEDIR).mkdir(parents=True, exist_ok=True)
Path(GWAS_BASEDIR).mkdir(parents=True, exist_ok=True)
Path(VAE_BASEDIR).mkdir(parents=True, exist_ok=True)

Path(f"{SIM_BASEDIR}/config.json").write_text(json.dumps(CFG, indent=2))

##############################################################################
# Optional VAE-input build settings in EXP_CFG under key "vae_data"
##############################################################################
VAE_DATA = CFG.get("vae_data", {}) or {}
SUBSET_MODE = VAE_DATA.get("subset_mode", "random")
SUBSET_SEED = int(VAE_DATA.get("subset_seed", 0))
MAF_THRESHOLD = float(CFG.get("maf_threshold", 0.05))
SUBSET_BP = VAE_DATA.get("subset_bp", None)
SUBSET_SNPS = int(VAE_DATA.get("subset_snps", 10000))

##############################################################################
# GWAS settings
##############################################################################
GWAS_CFG = CFG.get("gwas", {}) or {}
DISCOVERY_POP = GWAS_CFG.get("discovery_pop", "CEU")

##############################################################################
# RULE all
##############################################################################
rule all:
    input:
        # --- simulation artifacts ---
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/tree_sequence.trees", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/phenotype.pkl", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/effect_sizes.pkl", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/sampled_params.pkl", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/SFS.pkl", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/demes.png", sid=SID_RANGE, rep=REP_RANGE),

        # --- processed genotype artifacts ---
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/all_individuals.npy", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/hap1.npy", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/hap2.npy", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/meta.pkl", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/hap_meta.pkl", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/variant_site_ids.npy", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/genotype_site_stats.txt", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/site_filter_report.txt", sid=SID_RANGE, rep=REP_RANGE),

        # --- splits ---
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_train_idx.npy", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_val_idx.npy", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/target_idx.npy", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_train.npy", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_val.npy", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/target.npy", sid=SID_RANGE, rep=REP_RANGE),

        # --- VAE training for every exp ---
        expand(f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/.train_done", exp=EXP_NAMES, sid=SID_RANGE, rep=REP_RANGE),

        # --- VAE diagnostics for every exp ---
        expand(f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/diagnostics/.done", exp=EXP_NAMES, sid=SID_RANGE, rep=REP_RANGE),

##############################################################################
# RULE simulate
##############################################################################
rule simulate:
    input:
        cfg=EXP_CFG
    output:
        sfs=f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/SFS.pkl",
        params=f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/sampled_params.pkl",
        tree=f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/tree_sequence.trees",
        fig=f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/demes.png",
        trait=f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/effect_sizes.pkl",
        phenotype=f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/phenotype.pkl",
        done=f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/.done",
    params:
        model=MODEL
    threads: 1
    shell:
        r"""
        set -euo pipefail

        python -u "{SIM_SCRIPT}" \
          --simulation-dir "{SIM_BASEDIR}/{wildcards.sid}/rep{wildcards.rep}" \
          --experiment-config "{input.cfg}" \
          --model-type "{params.model}" \
          --simulation-number {wildcards.sid} \
          --replicate {wildcards.rep} \
          --output-dir "{SIM_BASEDIR}/{wildcards.sid}/rep{wildcards.rep}"

        touch "{output.done}"
        """

##############################################################################
# RULE build_inputs
##############################################################################
rule build_inputs:
    input:
        cfg=EXP_CFG,
        tree=f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/tree_sequence.trees",
        pheno=f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/phenotype.pkl",
        done=f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/.done",
    output:
        geno=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/all_individuals.npy",
        hap1=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/hap1.npy",
        hap2=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/hap2.npy",
        meta=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/meta.pkl",
        hap_meta=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/hap_meta.pkl",
        snp_idx=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/snp_index.npy",
        ts_ids=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/ts_individual_ids.npy",
        positions=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/variant_positions_bp.npy",
        site_ids=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/variant_site_ids.npy",
        stats_txt=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/genotype_site_stats.txt",
        filt_txt=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/site_filter_report.txt",

        disc_train_idx=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_train_idx.npy",
        disc_val_idx=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_val_idx.npy",
        target_idx=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/target_idx.npy",
        disc_train_npy=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_train.npy",
        disc_val_npy=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_val.npy",
        target_npy=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/target.npy",
    params:
        outdir=lambda wc: f"{GENO_BASEDIR}/{wc.sid}/rep{wc.rep}",
        subset_mode=SUBSET_MODE,
        subset_seed=SUBSET_SEED,
        maf_threshold=MAF_THRESHOLD,
        subset_bp=SUBSET_BP,
        subset_snps=SUBSET_SNPS,
        val_frac=0.2,
        split_seed=SUBSET_SEED,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH="{workflow.basedir}:${{PYTHONPATH:-}}"

        EXTRA_SUBSET_ARGS=""
        if [ "{params.subset_bp}" != "None" ] && [ -n "{params.subset_bp}" ]; then
            EXTRA_SUBSET_ARGS="--subset-bp {params.subset_bp}"
        else
            EXTRA_SUBSET_ARGS="--subset-snps {params.subset_snps}"
        fi

        python -u "{BUILD_INPUTS}" \
          --tree "{input.tree}" \
          --phenotype "{input.pheno}" \
          --outdir "{params.outdir}" \
          --experiment-config-json "{input.cfg}" \
          --maf-threshold "{params.maf_threshold}" \
          --subset-mode "{params.subset_mode}" \
          --subset-seed "{params.subset_seed}" \
          --val-frac "{params.val_frac}" \
          --split-seed "{params.split_seed}" \
          $EXTRA_SUBSET_ARGS

        test -f "{output.geno}"
        test -f "{output.hap1}"
        test -f "{output.hap2}"
        test -f "{output.meta}"
        test -f "{output.hap_meta}"
        test -f "{output.snp_idx}"
        test -f "{output.ts_ids}"
        test -f "{output.positions}"
        test -f "{output.site_ids}"
        test -f "{output.stats_txt}"
        test -f "{output.filt_txt}"
        test -f "{output.disc_train_idx}"
        test -f "{output.disc_val_idx}"
        test -f "{output.target_idx}"
        test -f "{output.disc_train_npy}"
        test -f "{output.disc_val_npy}"
        test -f "{output.target_npy}"
        """

##############################################################################
# RULE gwas
##############################################################################
rule gwas:
    input:
        geno=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/all_individuals.npy",
        meta=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/meta.pkl",
        site_ids=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/variant_site_ids.npy",
        trait=f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/effect_sizes.pkl",
    output:
        csv=f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_results.csv",
        manhattan=f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_manhattan.png",
        qq=f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_qq.png",
        af_diff=f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_af_diff.png",
    params:
        out_prefix=lambda wc: f"{GWAS_BASEDIR}/{wc.sid}/rep{wc.rep}/gwas",
        discovery_pop=DISCOVERY_POP,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH="{workflow.basedir}:${{PYTHONPATH:-}}"

        python -u "{GWAS_SCRIPT}" \
          --genotype "{input.geno}" \
          --phenotype "{input.meta}" \
          --trait "{input.trait}" \
          --variant-site-ids "{input.site_ids}" \
          --output-prefix "{params.out_prefix}" \
          --discovery-pop "{params.discovery_pop}"

        test -f "{output.csv}"
        test -f "{output.manhattan}"
        test -f "{output.qq}"
        test -f "{output.af_diff}"
        """

##############################################################################
# RULE train_vae
##############################################################################
rule train_vae:
    input:
        train=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_train.npy",
        val=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_val.npy",
        target=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/target.npy",
        hparams=lambda wc: str(exp_yaml_path(wc.exp)),
    output:
        best_ckpt=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/checkpoints/best.ckpt",
        summary=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/train_summary.json",
        logs_ok=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/logs_ok.txt",
        logs_dir=directory(f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/logs"),
        resolved=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/hparams.resolved.yaml",
        grid_yaml=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/hparams.grid.yaml",
        recon_dir=directory(f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/recon"),
        done=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/.train_done",
    params:
        outdir=lambda wc: f"{VAE_BASEDIR}/{wc.exp}/{wc.sid}/rep{wc.rep}",
        accelerator="gpu",
        devices="1",
        precision="32-true",
        save_recon=True,
        recon_n=16,
        recon_splits="train,val,target",
        skip_target_val=True,
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH="{workflow.basedir}:${{PYTHONPATH:-}}"
        mkdir -p "{params.outdir}"

        RECON_ARGS=""
        if [ "{params.save_recon}" = "True" ]; then
            RECON_ARGS="--save-recon --recon-n {params.recon_n} --recon-splits {params.recon_splits}"
        fi

        TARGET_VAL_ARGS=""
        if [ "{params.skip_target_val}" = "True" ]; then
            TARGET_VAL_ARGS="--no-target-val"
        fi

        python -u "{TRAIN_VAE_SCRIPT}" \
          --train "{input.train}" \
          --val "{input.val}" \
          --target "{input.target}" \
          --hparams "{input.hparams}" \
          --outdir "{params.outdir}" \
          --accelerator "{params.accelerator}" \
          --devices "{params.devices}" \
          --precision "{params.precision}" \
          $TARGET_VAL_ARGS \
          $RECON_ARGS

        cp "{input.hparams}" "{output.grid_yaml}"

        test -f "{output.best_ckpt}"
        test -f "{output.summary}"
        test -f "{output.logs_ok}"
        test -f "{output.resolved}"
        test -f "{output.grid_yaml}"
        test -d "{output.logs_dir}"

        if [ "{params.save_recon}" = "True" ]; then
            test -d "{output.recon_dir}"
            test -f "{output.recon_dir}/val_recon.npz"
        fi

        touch "{output.done}"
        """
##############################################################################
# RULE vae_diagnostics
##############################################################################
rule vae_diagnostics:
    input:
        train=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_train.npy",
        val=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/discovery_val.npy",
        target=f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/target.npy",
        ckpt=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/checkpoints/best.ckpt",
        logdir=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/logs",
        resolved=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/hparams.resolved.yaml",
        recon_dir=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/recon",
        done=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/.train_done",
    output:
        loss_epoch=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/diagnostics/plots/epoch_loss_mean.png",
        masked_mse=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/diagnostics/plots/masked_mse_mean.png",
        clean_mse=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/diagnostics/plots/clean_mse_mean.png",
        kl_logy=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/diagnostics/plots/kl_mean_logy.png",
        scatter_val=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/diagnostics/plots/scatter_val_masked_vs_clean.png",
        summary=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/diagnostics/recon_summary.txt",
        bal_summary=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/diagnostics/balanced_accuracy_summary.txt",
        done=f"{VAE_BASEDIR}/{{exp}}/{{sid}}/rep{{rep}}/diagnostics/.done",
    params:
        outdir=lambda wc: f"{VAE_BASEDIR}/{wc.exp}/{wc.sid}/rep{wc.rep}/diagnostics",
        batch_size=64,
        device="cpu",
        max_step_points=5000,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH="{workflow.basedir}:${{PYTHONPATH:-}}"
        mkdir -p "{params.outdir}"

        python -u "{PLOT_WRAPPER}" \
          --logdir "{input.logdir}" \
          --checkpoint "{input.ckpt}" \
          --resolved-hparams "{input.resolved}" \
          --train-genotype "{input.train}" \
          --val-genotype "{input.val}" \
          --target-genotype "{input.target}" \
          --recon-dir "{input.recon_dir}" \
          --outdir "{params.outdir}" \
          --batch-size "{params.batch_size}" \
          --max-step-points "{params.max_step_points}" \
          --device "{params.device}"

        test -f "{output.loss_epoch}"
        test -f "{output.masked_mse}"
        test -f "{output.clean_mse}"
        test -f "{output.kl_logy}"
        test -f "{output.scatter_val}"
        test -f "{output.summary}"
        test -f "{output.bal_summary}"

        touch "{output.done}"
        """