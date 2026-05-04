#!/bin/bash
#SBATCH --job-name={{ANALYSIS_JOB_NAME}}
#SBATCH --output={{LOG_DIR}}/{{ANALYSIS_JOB_NAME}}_%j.out
#SBATCH --error={{LOG_DIR}}/{{ANALYSIS_JOB_NAME}}_%j.err
#SBATCH --partition={{CPU_PARTITION}}
#SBATCH --cpus-per-task={{CPUS_PER_TASK}}
#SBATCH --mem={{CPU_MEMORY}}
#SBATCH --time={{CPU_TIME_LIMIT}}

set -euxo pipefail

module purge
module load {{PYTORCH_MODULE}}

cd "{{REPO_ROOT}}"

mkdir -p "{{LOG_DIR}}" "{{ANALYSIS_OUTPUT_DIR}}"

python "src/{{ANALYSIS_SCRIPT}}" {{ANALYSIS_ARGS}}
