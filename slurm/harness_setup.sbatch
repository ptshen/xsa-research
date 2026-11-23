#!/bin/bash
#SBATCH --job-name=xsa-harness-setup
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/harness-setup-%j.log
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
export PYTHONPATH="$PWD"
export HF_HOME="$PWD/cache/huggingface" TIKTOKEN_CACHE_DIR="$PWD/cache/tiktoken" OMP_NUM_THREADS=1
.venv/bin/python -m pip install --no-cache-dir 'lm_eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9'
.venv/bin/python -m pip freeze > evidence/cluster-harness-environment.lock.txt
.venv/bin/python scripts/verify_harness.py --checkpoint runs/smoke-14072282/sa/checkpoint.pt > evidence/harness_integration.json
