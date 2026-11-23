#!/bin/bash
# Resources are supplied by python -m xsa.submission; direct sbatch is discouraged.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export HF_HOME="$PWD/cache/huggingface" TIKTOKEN_CACHE_DIR="$PWD/cache/tiktoken"
export PYTHONUNBUFFERED=1
test -n "${SLURM_JOB_ID:-}"
nvidia-smi
# timeout is additional to Slurm's hard allocation limit; no detached children.
srun timeout --signal=TERM --kill-after=5 "$((XSA_SMOKE_SECONDS-15))s" .venv/bin/python -m xsa.smoke --output "runs/smoke-$SLURM_JOB_ID"
