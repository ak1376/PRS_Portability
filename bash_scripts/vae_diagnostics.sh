#!/bin/bash
#SBATCH --job-name=vae_diag
#SBATCH --output=logs/vae_diag_%A_%a.out
#SBATCH --error=logs/vae_diag_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=kern
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

set -euo pipefail

ROOT="/projects/kernlab/akapoor/PRS_Portability"
SNAKEFILE="$ROOT/Snakefile"
CFG="${CFG_PATH:-$ROOT/config_files/experiment_config_IM_symmetric.json}"
VAE_YAML="${VAE_YAML_PATH:-$ROOT/config_files/model_hyperparams/vae.yaml}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"

mkdir -p "$ROOT/logs"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "ROOT=$ROOT"
echo "SNAKEFILE=$SNAKEFILE"
echo "CFG=$CFG"
echo "VAE_YAML=$VAE_YAML"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "EXP_NAME=${EXP_NAME:-unset}"

MODEL=$(python - <<'PY' "$CFG"
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg["demographic_model"])
PY
)

NUM_DRAWS=$(python - <<'PY' "$CFG"
import json, sys
cfg = json.load(open(sys.argv[1]))
print(int(cfg.get("num_draws", 1)))
PY
)

NUM_REPS=$(python - <<'PY' "$CFG"
import json, sys
cfg = json.load(open(sys.argv[1]))
print(int(cfg.get("num_replicates", 1)))
PY
)

mapfile -t EXP_NAMES < <(python - <<'PY' "$VAE_YAML"
from pathlib import Path
import itertools
import yaml
import sys

vae_yaml = Path(sys.argv[1])
base = yaml.safe_load(vae_yaml.read_text()) or {}

grid = base.get("grid", {}) or {}
enabled = bool(grid.get("enabled", False))
dims = grid.get("dims", []) or []

def tagify(v):
    if isinstance(v, float):
        s = f"{v:g}"
        return s.replace(".", "p")
    return str(v)

def make_exp_name(assignments):
    name_cfg = grid.get("name", {}) or {}
    prefix = name_cfg.get("prefix", "vae")
    sep = name_cfg.get("sep", "__")
    parts = []
    for dim, val in assignments:
        tag = dim.get("tag", None)
        pth = dim["path"]
        key = tag if tag else pth.split(".")[-1]
        parts.append(f"{key}{tagify(val)}")
    return prefix + sep + sep.join(parts)

if enabled:
    values_lists = [dim["values"] for dim in dims]
    for combo in itertools.product(*values_lists):
        assignments = list(zip(dims, combo))
        print(make_exp_name(assignments))
else:
    print("default")
PY
)

NUM_EXPS=${#EXP_NAMES[@]}

if [[ "$NUM_EXPS" -eq 0 ]]; then
  echo "ERROR: no experiment names found"
  exit 1
fi

NUM_TASKS=$(( NUM_DRAWS * NUM_REPS ))
LAST_TASK=$(( NUM_TASKS - 1 ))

echo "MODEL=$MODEL"
echo "NUM_DRAWS=$NUM_DRAWS"
echo "NUM_REPS=$NUM_REPS"
echo "NUM_TASKS=$NUM_TASKS"
echo "NUM_EXPS=$NUM_EXPS"
echo "EXP_NAMES=${EXP_NAMES[*]}"

# -------------------------------------------------------------------
# Submit mode: no EXP_NAME and no array task yet -> submit one array per experiment
# -------------------------------------------------------------------
if [[ -z "${EXP_NAME:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  for exp in "${EXP_NAMES[@]}"; do
    echo "Submitting diagnostics for experiment '$exp' as array 0-${LAST_TASK}%${MAX_CONCURRENT}"
    sbatch \
      --export=ALL,EXP_NAME="$exp",CFG_PATH="$CFG",VAE_YAML_PATH="$VAE_YAML",MAX_CONCURRENT="$MAX_CONCURRENT" \
      --array=0-"$LAST_TASK"%${MAX_CONCURRENT} \
      "$0"
  done
  exit 0
fi

# -------------------------------------------------------------------
# Safety checks
# -------------------------------------------------------------------
if [[ -z "${EXP_NAME:-}" ]]; then
  echo "ERROR: EXP_NAME is not set inside array task"
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set inside array task"
  exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID}

if [[ "$IDX" -lt 0 || "$IDX" -ge "$NUM_TASKS" ]]; then
  echo "ERROR: array index $IDX out of bounds for NUM_TASKS=$NUM_TASKS"
  exit 1
fi

# Decode array index -> (SID, REP)
SID=$(( IDX / NUM_REPS ))
REP=$(( IDX % NUM_REPS ))

TARGET="experiments/${MODEL}/vae/${EXP_NAME}/${SID}/rep${REP}/diagnostics/.done"

echo "Resolved:"
echo "  EXP_NAME=$EXP_NAME"
echo "  IDX=$IDX"
echo "  SID=$SID"
echo "  REP=$REP"
echo "  TARGET=$TARGET"

if [[ -f "$TARGET" ]]; then
  echo "SKIP: target already exists: $TARGET"
  exit 0
fi

snakemake \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --rerun-triggers mtime \
  --nolock \
  --cores "${SLURM_CPUS_PER_TASK:-1}" \
  "$TARGET"

echo "Finished diagnostics EXP_NAME=$EXP_NAME IDX=$IDX SID=$SID REP=$REP"