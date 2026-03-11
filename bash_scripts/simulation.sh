#!/bin/bash
#SBATCH --job-name=simulate_array
#SBATCH --output=logs/simulate_array_%A_%a.out
#SBATCH --error=logs/simulate_array_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=kern,kerngpu,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

set -euo pipefail

ROOT="/projects/kernlab/akapoor/PRS_Portability"
SNAKEFILE="$ROOT/Snakefile"
CFG="${CFG_PATH:-$ROOT/config_files/experiment_config_IM_symmetric.json}"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"

mkdir -p "$ROOT/logs"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "ROOT=$ROOT"
echo "SNAKEFILE=$SNAKEFILE"
echo "CFG=$CFG"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

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

NUM_TASKS=$(( NUM_DRAWS * NUM_REPS ))
LAST_TASK=$(( NUM_TASKS - 1 ))

echo "MODEL=$MODEL"
echo "NUM_DRAWS=$NUM_DRAWS"
echo "NUM_REPS=$NUM_REPS"
echo "NUM_TASKS=$NUM_TASKS"

# Submit mode
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submitting simulate array 0-${LAST_TASK}%${MAX_CONCURRENT}"
  sbatch \
    --export=ALL,CFG_PATH="$CFG",MAX_CONCURRENT="$MAX_CONCURRENT" \
    --array=0-"$LAST_TASK"%${MAX_CONCURRENT} \
    "$0"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID}

if [[ "$IDX" -lt 0 || "$IDX" -ge "$NUM_TASKS" ]]; then
  echo "ERROR: array index $IDX out of bounds for NUM_TASKS=$NUM_TASKS"
  exit 1
fi

SID=$(( IDX / NUM_REPS ))
REP=$(( IDX % NUM_REPS ))

TARGET="experiments/${MODEL}/simulations/${SID}/rep${REP}/.done"

echo "Resolved:"
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

echo "Finished simulate task IDX=$IDX SID=$SID REP=$REP"