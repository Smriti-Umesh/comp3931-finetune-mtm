#!/bin/bash
#SBATCH --job-name=ibl_mtm_combined
#SBATCH --output=${LOG_DIR}/ibl_mtm_combined_%j.out
#SBATCH --error=${LOG_DIR}/ibl_mtm_combined_%j.err
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}

set -euxo pipefail

module purge
module load ${PYTORCH_MODULE}

cd ${REPO_ROOT}

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" data
printf "control_session\n" > data/target_eids.txt

PRETRAIN_CKPT="${PRETRAIN_CKPT}"

if [ ! -f "$PRETRAIN_CKPT" ]; then
  echo "Missing pretrained checkpoint: $PRETRAIN_CKPT"
  exit 1
fi

RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
SAVE_DIR="${RESULTS_DIR}/ibl_mtm_combined_direct_full_lr${LR}_adapter${ADAPTER_LR}_${RUN_ID}"

python src/train_single_session_local_mtm_combined.py \
  --data-dir "${DATA_DIR}" \
  --pretrained-ckpt "$PRETRAIN_CKPT" \
  --save-dir "$SAVE_DIR" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --adapter-lr "${ADAPTER_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --warmup-pct "${WARMUP_PCT}" \
  --neuron-mask-ratio 0.3 \
  --causal-mask-ratio 0.6 \
  --causal-prob 0.5 \
  --patience 30 \
  --min-delta 1e-5 \
  --eval-splits test train val \
  --eval-neuron-mask-ratio 0.3 \
  --eval-causal-mask-ratio 0.6 \
  --artifact-seed 123
