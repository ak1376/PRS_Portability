#!/bin/bash
#SBATCH --job-name=vae_masks
#SBATCH --output=logs/vae_masks_%A_%a.out
#SBATCH --error=logs/vae_masks_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=kern,kerngpu,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

set -euo pipefail

ROOT="/projects/kernlab/akapoor/PRS_Portability"
SNAKEFILE="$ROOT/Snakefile"
CFG="${CFG_PATH:-$ROOT/config_files/experiment_config_IM_symmetric.json}"
VAE_YAML="${VAE_YAML_PATH:-$ROOT/config_files/model_hyperparams/vae.yaml}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"

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
echo "MAX_EPOCHS_EXP=${MAX_EPOCHS_EXP:-unset}"

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

# Build experiment names AND max_epochs for each experiment from the base VAE yaml.
mapfile -t EXP_AND_EPOCHS < <(python - <<'PY' "$VAE_YAML"
from pathlib import Path
import itertools
import yaml
import sys
import copy

vae_yaml = Path(sys.argv[1])
base = yaml.safe_load(vae_yaml.read_text()) or {}

grid = base.get("grid", {}) or {}
enabled = bool(grid.get("enabled", False))
dims = grid.get("dims", []) or []

def set_by_path(d, path, value):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

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
        exp = make_exp_name(assignments)

        cfg = copy.deepcopy(base)
        cfg.pop("grid", None)
        for dim, val in assignments:
            set_by_path(cfg, dim["path"], val)

        max_epochs = int((cfg.get("training", {}) or {}).get("max_epochs", 50))
        print(f"{exp}\t{max_epochs}")
else:
    cfg = copy.deepcopy(base)
    cfg.pop("grid", None)
    max_epochs = int((cfg.get("training", {}) or {}).get("max_epochs", 50))
    print(f"default\t{max_epochs}")
PY
)

NUM_EXPS=${#EXP_AND_EPOCHS[@]}

if [[ "$NUM_EXPS" -eq 0 ]]; then
  echo "ERROR: no experiment names found"
  exit 1
fi

echo "MODEL=$MODEL"
echo "NUM_DRAWS=$NUM_DRAWS"
echo "NUM_REPS=$NUM_REPS"
echo "NUM_EXPS=$NUM_EXPS"
printf 'EXP_AND_EPOCHS:\n%s\n' "${EXP_AND_EPOCHS[@]}"

# -------------------------------------------------------------------
# Submit mode: no EXP_NAME and no array task yet -> submit one array per experiment
# -------------------------------------------------------------------
if [[ -z "${EXP_NAME:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  for line in "${EXP_AND_EPOCHS[@]}"; do
    exp=$(printf '%s\n' "$line" | cut -f1)
    max_epochs=$(printf '%s\n' "$line" | cut -f2)

    num_tasks=$(( NUM_DRAWS * NUM_REPS * max_epochs ))
    last_task=$(( num_tasks - 1 ))

    echo "Submitting experiment '$exp' with max_epochs=$max_epochs as array 0-${last_task}%${MAX_CONCURRENT}"
    sbatch \
      --export=ALL,EXP_NAME="$exp",MAX_EPOCHS_EXP="$max_epochs",CFG_PATH="$CFG",VAE_YAML_PATH="$VAE_YAML",MAX_CONCURRENT="$MAX_CONCURRENT" \
      --array=0-"$last_task"%${MAX_CONCURRENT} \
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

if [[ -z "${MAX_EPOCHS_EXP:-}" ]]; then
  echo "ERROR: MAX_EPOCHS_EXP is not set inside array task"
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set inside array task"
  exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID}
MAX_EPOCHS=${MAX_EPOCHS_EXP}

NUM_TASKS=$(( NUM_DRAWS * NUM_REPS * MAX_EPOCHS ))

if [[ "$IDX" -lt 0 || "$IDX" -ge "$NUM_TASKS" ]]; then
  echo "ERROR: array index $IDX out of bounds for NUM_TASKS=$NUM_TASKS"
  exit 1
fi

# Decode IDX -> (SID, REP, EPOCH)
PER_SID=$(( NUM_REPS * MAX_EPOCHS ))
SID=$(( IDX / PER_SID ))
REM=$(( IDX % PER_SID ))
REP=$(( REM / MAX_EPOCHS ))
EPOCH=$(( REM % MAX_EPOCHS ))

TARGET_MASKED="experiments/${MODEL}/vae/${EXP_NAME}/${SID}/rep${REP}/masked_inputs/train_masked_epoch${EPOCH}.npy"
TARGET_MASK="experiments/${MODEL}/vae/${EXP_NAME}/${SID}/rep${REP}/masked_inputs/train_mask_epoch${EPOCH}.npy"

echo "Resolved:"
echo "  EXP_NAME=$EXP_NAME"
echo "  IDX=$IDX"
echo "  SID=$SID"
echo "  REP=$REP"
echo "  EPOCH=$EPOCH"
echo "  TARGET_MASKED=$TARGET_MASKED"
echo "  TARGET_MASK=$TARGET_MASK"

if [[ -f "$TARGET_MASKED" && -f "$TARGET_MASK" ]]; then
  echo "SKIP: both outputs already exist"
  exit 0
fi

snakemake \
  --snakefile "$SNAKEFILE" \
  --directory "$ROOT" \
  --rerun-incomplete \
  --rerun-triggers mtime \
  --nolock \
  --cores "${SLURM_CPUS_PER_TASK:-1}" \
  "$TARGET_MASKED" "$TARGET_MASK"

echo "Finished EXP_NAME=$EXP_NAME IDX=$IDX SID=$SID REP=$REP EPOCH=$EPOCH"