#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@120
#SBATCH --no-requeue
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
/usr/bin/python3 -m xsa.direction_bc gate
export XSA_CONFIG="configs/direction_${XSA_BC_TAG}.json" XSA_SEED=17
export XSA_INITIAL_STATE=runs/initial/pilot-s17.pt XSA_DATA=data/pilot-v2
export XSA_OUTPUT="runs/direction-${XSA_BC_TAG}-s17" XSA_POST_EVALUATE=1
if [[ "$XSA_BC_TAG" == c ]]; then export XSA_ATTENTION=output_geometry; else export XSA_ATTENTION=value_penalty; fi
bash slurm/train.sbatch
