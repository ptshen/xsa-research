#!/bin/bash
#SBATCH --job-name=xsa-pilot-verify
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --output=logs/pilot-verify-%j.log
set -euo pipefail
cd ~/algollms/xsa
export OMP_NUM_THREADS=1
.venv/bin/python -m pytest -q tests/test_pilot_gate.py tests/test_pipeline.py::test_preparation_token_cap_preserves_complete_documents
GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15' git ls-remote origin refs/heads/main
