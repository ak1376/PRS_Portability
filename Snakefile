# Snakefile
import json
from pathlib import Path

##############################################################################
# Scripts + Config
##############################################################################
SIM_SCRIPT   = "snakemake_scripts/simulation.py"
BUILD_INPUTS = "snakemake_scripts/build_genotypes_for_vae.py"
GWAS_SCRIPT  = "snakemake_scripts/run_gwas.py"

TRAIN_VAE_SCRIPT = "snakemake_scripts/train_vae.py"
PLOT_WRAPPER     = "snakemake_scripts/plot_vae_diagnostics.py" # provided below

EXP_CFG = "config_files/experiment_config_IM_symmetric.json"
CFG = json.loads(Path(EXP_CFG).read_text())
MODEL = CFG["demographic_model"]

##############################################################################
# Ranges
##############################################################################
NUM_DRAWS = int(CFG.get("num_draws", 1))
REPS      = int(CFG.get("num_replicates", 1))

SID_RANGE = range(NUM_DRAWS)   # sid = 0..num_draws-1
REP_RANGE = range(REPS)        # rep = 0..num_replicates-1

##############################################################################
# Directories
##############################################################################
SIM_BASEDIR  = f"experiments/{MODEL}/simulations"
GENO_BASEDIR = f"experiments/{MODEL}/processed_data"
GWAS_BASEDIR = f"experiments/{MODEL}/gwas"
VAE_BASEDIR  = f"experiments/{MODEL}/vae"

Path(SIM_BASEDIR).mkdir(parents=True, exist_ok=True)
Path(GENO_BASEDIR).mkdir(parents=True, exist_ok=True)
Path(GWAS_BASEDIR).mkdir(parents=True, exist_ok=True)
Path(VAE_BASEDIR).mkdir(parents=True, exist_ok=True)
Path(f"{SIM_BASEDIR}/config.json").write_text(json.dumps(CFG, indent=2))

##############################################################################
# Optional: VAE-input build settings in EXP_CFG under key "vae_data"
##############################################################################
VAE_DATA = CFG.get("vae_data", {}) or {}
SUBSET_MODE   = VAE_DATA.get("subset_mode", "random")
SUBSET_SEED   = int(VAE_DATA.get("subset_seed", 0))
MAF_THRESHOLD = float(VAE_DATA.get("maf_threshold", 0.01))
SUBSET_BP     = VAE_DATA.get("subset_bp", None)         # if not None, used instead of subset_snps
SUBSET_SNPS   = int(VAE_DATA.get("subset_snps", 10000))

##############################################################################
# GWAS settings (optional; defaults here)
##############################################################################
GWAS_CFG = CFG.get("gwas", {}) or {}
DISCOVERY_POP = GWAS_CFG.get("discovery_pop", "CEU")

##############################################################################
# VAE training config (optional; defaults here)
##############################################################################
MODEL_CFG_YAML = "config_files/model_hyperparams/vae.yaml"
SEED_DEFAULT   = int(CFG.get("seed", 0))
TRAINING_CFG   = CFG.get("training", {}) or {}
MAX_EPOCHS     = int(TRAINING_CFG.get("max_epochs", 50))
BATCH_SIZE     = int(TRAINING_CFG.get("batch_size", 256))
NUM_WORKERS    = int(TRAINING_CFG.get("num_workers", 4))
PRECISION      = str(TRAINING_CFG.get("precision", "16-mixed"))
LOG_EVERY_N    = int(TRAINING_CFG.get("log_every_n_steps", 10))

##############################################################################
# RULE all
##############################################################################
rule all:
    input:
        # --- simulation artifacts ---
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/tree_sequence.trees", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/phenotype.pkl",       sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/effect_sizes.pkl",    sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/sampled_params.pkl",  sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/SFS.pkl",             sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/demes.png",           sid=SID_RANGE, rep=REP_RANGE),

        # --- processed genotype artifacts ---
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/all_individuals.npy",       sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/meta.pkl",                 sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/variant_site_ids.npy",     sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/train_idx.npy",            sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/val_idx.npy",              sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/train.npy",                sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/val.npy",                  sid=SID_RANGE, rep=REP_RANGE),

        # --- GWAS outputs ---
        expand(f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_results.csv",   sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_manhattan.png", sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_qq.png",        sid=SID_RANGE, rep=REP_RANGE),
        expand(f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_af_diff.png",   sid=SID_RANGE, rep=REP_RANGE),

        # --- VAE training ---
        expand(f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/.train_done", sid=SID_RANGE, rep=REP_RANGE),

        # --- VAE diagnostics ---
        # expand(f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/diagnostics/.done", sid=SID_RANGE, rep=REP_RANGE),

##############################################################################
# RULE simulate – one sim per sid×rep
##############################################################################
rule simulate:
    input:
        cfg = EXP_CFG
    output:
        sfs       = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/SFS.pkl",
        params    = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/sampled_params.pkl",
        tree      = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/tree_sequence.trees",
        fig       = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/demes.png",
        trait     = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/effect_sizes.pkl",
        phenotype = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/phenotype.pkl",
        done      = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/.done",
    params:
        model = MODEL
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
# RULE build_inputs – trees -> diploid + haplotype matrices + meta + splits
##############################################################################
rule build_inputs:
    input:
        tree  = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/tree_sequence.trees",
        pheno = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/phenotype.pkl",
        done  = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/.done",
    output:
        geno      = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/all_individuals.npy",
        hap1      = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/hap1.npy",
        hap2      = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/hap2.npy",
        meta      = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/meta.pkl",
        hap_meta  = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/hap_meta.pkl",
        snp_idx   = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/snp_index.npy",
        ts_ids    = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/ts_individual_ids.npy",
        positions = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/variant_positions_bp.npy",
        site_ids  = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/variant_site_ids.npy",
        stats_txt = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/genotype_site_stats.txt",
        filt_txt  = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/site_filter_report.txt",

        # NEW: split artifacts
        train_idx = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/train_idx.npy",
        val_idx   = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/val_idx.npy",
        train_npy = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/train.npy",
        val_npy   = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/val.npy",
    params:
        outdir        = lambda wc: f"{GENO_BASEDIR}/{wc.sid}/rep{wc.rep}",

        # subset settings (already in your Snakefile globals)
        subset_mode   = SUBSET_MODE,
        subset_seed   = SUBSET_SEED,
        maf_threshold = MAF_THRESHOLD,
        subset_bp     = SUBSET_BP,
        subset_snps   = SUBSET_SNPS,

        # split settings
        split_mode    = "cross_pop",     # or "random" | "discovery_only" | "cross_pop"
        val_frac      = 0.2,
        split_seed    = SUBSET_SEED,      # deterministic w.r.t. your subset seed
        discovery_pop = DISCOVERY_POP,    # used if split_mode in {"discovery_only","cross_pop"}
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
          --maf-threshold "{params.maf_threshold}" \
          --subset-mode "{params.subset_mode}" \
          --subset-seed "{params.subset_seed}" \
          --split-mode "{params.split_mode}" \
          --val-frac "{params.val_frac}" \
          --split-seed "{params.split_seed}" \
          --discovery-pop "{params.discovery_pop}" \
          $EXTRA_SUBSET_ARGS

        # sanity checks
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
        test -f "{output.train_idx}"
        test -f "{output.val_idx}"
        test -f "{output.train_npy}"
        test -f "{output.val_npy}"
        """
        
##############################################################################
# RULE gwas – run GWAS per sid×rep using processed genotype matrix
##############################################################################
rule gwas:
    input:
        geno     = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/all_individuals.npy",
        meta     = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/meta.pkl",
        site_ids = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/variant_site_ids.npy",
        trait    = f"{SIM_BASEDIR}/{{sid}}/rep{{rep}}/effect_sizes.pkl",
    output:
        csv       = f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_results.csv",
        manhattan = f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_manhattan.png",
        qq        = f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_qq.png",
        af_diff   = f"{GWAS_BASEDIR}/{{sid}}/rep{{rep}}/gwas_af_diff.png",
    params:
        out_prefix    = lambda wc: f"{GWAS_BASEDIR}/{wc.sid}/rep{wc.rep}/gwas",
        discovery_pop = DISCOVERY_POP,
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
# RULE train_vae – train + validate Lightning model (per sid×rep)
##############################################################################

rule train_vae:
    input:
        train   = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/train.npy",
        val     = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/val.npy",
        hparams = MODEL_CFG_YAML,
    output:
        best_ckpt = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/checkpoints/best.ckpt",
        summary   = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/train_summary.json",
        logs_ok   = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/logs_ok.txt",
        logs_dir  = directory(f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/logs"),
        resolved  = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/hparams.resolved.yaml",
        done      = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/.train_done",
    params:
        outdir = lambda wc: f"{VAE_BASEDIR}/{wc.sid}/rep{wc.rep}",
        accelerator = "gpu",
        devices     = "auto",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH="{workflow.basedir}:${{PYTHONPATH:-}}"
        export CUDA_VISIBLE_DEVICES=0

        python -u "snakemake_scripts/train_vae.py" \
          --train "{input.train}" \
          --val "{input.val}" \
          --hparams "{input.hparams}" \
          --outdir "{params.outdir}" \
          --accelerator "{params.accelerator}" \
          --devices "{params.devices}" \

        test -f "{output.best_ckpt}"
        test -f "{output.summary}"
        test -f "{output.logs_ok}"
        test -f "{output.resolved}"
        test -d "{output.logs_dir}"
        touch "{output.done}"
        """

##############################################################################
# RULE vae_diagnostics – thin wrapper calling src/plotting.py (per sid×rep)
##############################################################################
rule vae_diagnostics:
    input:
        geno   = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/all_individuals.npy",
        meta   = f"{GENO_BASEDIR}/{{sid}}/rep{{rep}}/meta.pkl",
        ckpt   = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/checkpoints/best.ckpt",
        logdir = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/logs",
        resolved = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/hparams.resolved.yaml",
        done   = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/.train_done",
    output:
        loss_epoch = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/diagnostics/loss_epoch.png",
        heatmap    = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/diagnostics/recon_abs_error_heatmap.png",
        summary    = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/diagnostics/recon_summary.txt",
        done       = f"{VAE_BASEDIR}/{{sid}}/rep{{rep}}/diagnostics/.done",
    params:
        outdir = lambda wc: f"{VAE_BASEDIR}/{wc.sid}/rep{wc.rep}/diagnostics",
        batch_size = 256,
        max_step_points = 5000,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH="{workflow.basedir}:${{PYTHONPATH:-}}"
        mkdir -p "{params.outdir}"

        python -u "snakemake_scripts/plot_vae_diagnostics.py" \
          --logdir "{input.logdir}" \
          --checkpoint "{input.ckpt}" \
          --resolved-hparams "{input.resolved}" \
          --genotype "{input.geno}" \
          --meta "{input.meta}" \
          --outdir "{params.outdir}" \
          --batch-size "{params.batch_size}" \
          --max-step-points "{params.max_step_points}" \
          --device auto

        test -f "{output.loss_epoch}"
        test -f "{output.heatmap}"
        test -f "{output.summary}"
        touch "{output.done}"
        """