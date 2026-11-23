#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@120
#SBATCH --no-requeue
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" PYTHONUNBUFFERED=1
export HF_HOME="$PWD/cache/huggingface" TIKTOKEN_CACHE_DIR="$PWD/cache/tiktoken"
nvidia-smi
resume=()
if [[ "${XSA_RESUME:-0}" == 1 ]]; then resume=(--resume); fi
initial=()
if [[ -n "${XSA_INITIAL_STATE:-}" ]]; then initial=(--initial-state "$XSA_INITIAL_STATE"); fi
srun .venv/bin/python -m xsa.train --config "$XSA_CONFIG" --data "$XSA_DATA" --output "$XSA_OUTPUT" \
  --attention "$XSA_ATTENTION" --seed "$XSA_SEED" --device cuda "${resume[@]}" "${initial[@]}"
