#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@120
#SBATCH --no-requeue
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
srun .venv/bin/python -m xsa.direction_a gate
export XSA_CONFIG="configs/direction_a_${XSA_A_TAG}.json" XSA_ATTENTION=alignment_cap XSA_SEED=17
export XSA_INITIAL_STATE=runs/initial/pilot-s17.pt XSA_DATA=data/pilot-v2
export XSA_OUTPUT="runs/direction-a-tau${XSA_A_TAG}-s17" XSA_POST_EVALUATE=1
bash slurm/train.sbatch
