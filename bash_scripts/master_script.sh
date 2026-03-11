#!/bin/bash
#SBATCH --job-name=vae_master
#SBATCH --output=logs/vae_master.out
#SBATCH --error=logs/vae_master.err
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --partition=kern,preempt
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu

set -euo pipefail

REPO="/projects/kernlab/akapoor/PRS_Portability"
mkdir -p "$REPO/logs"
cd "$REPO"

CFG_PATH="$REPO/config_files/experiment_config_IM_symmetric.json"
export CFG_PATH

submit() {
    sbatch --parsable --export=ALL "$@"
}

echo "Using config: $CFG_PATH"
echo "Submitting pipeline from: $PWD"

sim_id=$(submit bash_scripts/simulation.sh)
[[ -n "$sim_id" ]]
echo "simulation.sh job id: $sim_id"

build_id=$(submit --dependency=afterok:$sim_id bash_scripts/build_inputs.sh)
[[ -n "$build_id" ]]
echo "build_inputs.sh job id: $build_id"

mask_id=$(submit --dependency=afterok:$build_id bash_scripts/precompute_masks.sh)
[[ -n "$mask_id" ]]
echo "precompute_masks.sh job id: $mask_id"

train_id=$(submit --dependency=afterok:$mask_id bash_scripts/train_vae.sh)
[[ -n "$train_id" ]]
echo "train_vae.sh job id: $train_id"

diag_id=$(submit --dependency=afterok:$train_id bash_scripts/vae_diagnostics.sh)
[[ -n "$diag_id" ]]
echo "vae_diagnostics.sh job id: $diag_id"

echo "Final job ID (vae_diagnostics.sh): $diag_id"