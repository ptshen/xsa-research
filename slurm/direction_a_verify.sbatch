#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export HF_HOME="$PWD/cache/huggingface" TIKTOKEN_CACHE_DIR="$PWD/cache/tiktoken"
nvidia-smi
out="runs/direction-a-verification-$SLURM_JOB_ID"
mkdir -p "$out"
srun .venv/bin/python -m pytest -q --basetemp "$out/tests" --junitxml "$out/tests.xml" | tee "$out/tests.txt"
cp "$out/tests.xml" "$out/tests.txt" evidence/direction_a/
srun .venv/bin/python -m xsa.direction_a verify --test-xml "$out/tests.xml"
