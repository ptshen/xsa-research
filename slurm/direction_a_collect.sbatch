#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
srun .venv/bin/python -m xsa.direction_a collect
