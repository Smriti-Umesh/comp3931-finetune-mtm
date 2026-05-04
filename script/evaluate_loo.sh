#!/bin/bash
#SBATCH --job-name=evaluate_loo
#SBATCH --output={{LOG_DIR}}/evaluate_loo_%j.out
#SBATCH --error={{LOG_DIR}}/evaluate_loo_%j.err
#SBATCH --partition={{GPU_PARTITION}}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={{CPUS_PER_TASK}}
#SBATCH --mem={{MEMORY}}
#SBATCH --time={{LOO_TIME_LIMIT}}

set -euxo pipefail

module purge
module load {{PYTORCH_MODULE}}

cd "{{REPO_ROOT}}"

mkdir -p "{{LOG_DIR}}" "{{LOO_OUTPUT_DIR}}"

RUN_DIR="{{RUN_DIR}}"
LABEL="{{LABEL}}"

python src/evaluate_loo.py \
  --run "$RUN_DIR" \
  --data-dir "{{DATA_DIR}}" \
  --output-dir "{{LOO_OUTPUT_DIR}}" \
  --label "$LABEL" \
  --batch-size "{{LOO_BATCH_SIZE}}" \
  --n-jobs "{{N_JOBS}}" \
  --split "{{LOO_SPLIT}}"
