#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@120
#SBATCH --no-requeue
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
export XSA_CONFIG=configs/pilot.json XSA_ATTENTION=xsa XSA_SEED=17
export XSA_INITIAL_STATE=runs/initial/pilot-s17.pt XSA_DATA=data/pilot-v2 XSA_OUTPUT=runs/pilot-xsa-s17
export XSA_POST_EVALUATE=1 XSA_RESUME=0
.venv/bin/python -m xsa.pilot gate
bash slurm/train.sbatch
