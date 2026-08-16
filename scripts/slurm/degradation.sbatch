#!/bin/bash
# Generic template for the tile-budget degradation sweep on a SLURM cluster.
# Adjust the account/partition to your site, then:
#   sbatch scripts/slurm/degradation.sbatch configs/mycohort.yaml mycohort all
#SBATCH --job-name=degradation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
set -euo pipefail

CONFIG="${1:?pass a config, e.g. configs/mycohort.yaml}"
DOMAIN="${2:-}"
PROXY="${3:-all}"

mkdir -p logs
source .venv/bin/activate

# One fold per array task when submitted with --array=0-9; otherwise all folds.
FOLD_ARG=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    FOLD_ARG="--fold ${SLURM_ARRAY_TASK_ID}"
fi

srun tileselect-eval \
    --config "$CONFIG" \
    --domain "$DOMAIN" \
    --proxy  "$PROXY" \
    --split  test \
    $FOLD_ARG
