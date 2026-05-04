#!/bin/bash
#SBATCH --job-name=ndt1_scratch_${MASKING_MODE}
#SBATCH --output=${LOG_DIR}/ndt1_scratch_${MASKING_MODE}_%j.out
#SBATCH --error=${LOG_DIR}/ndt1_scratch_${MASKING_MODE}_%j.err
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}

set -euxo pipefail

module purge
module load ${PYTORCH_MODULE}

cd "${REPO_ROOT}"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" data
printf "control_session\n" > data/target_eids.txt

RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
SAVE_DIR="${RESULTS_DIR}/ndt1_direct_${MASKING_MODE}_full_lr${LR}_${RUN_ID}"

python src/train_single_session_local_ndt1_scratch.py \
  --data-dir "${DATA_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --masking-mode "${MASKING_MODE}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --mask-ratio "${MASK_RATIO}" \
  --patience "${PATIENCE}" \
  --min-delta "${MIN_DELTA}" \
  --eval-split test \
  --extra-eval-splits train val \
  --eval-mask-ratio "${MASK_RATIO}" \
  --artifact-seed "${ARTIFACT_SEED}"
