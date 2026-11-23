#!/bin/bash
#SBATCH --job-name=xsa-setup
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/setup-%j.log
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
module load anaconda/2023.09
base="$HOME/hpml/dust/.venv/baseline-cu128"
# Reuse the existing university CUDA environment without modifying it.
python -m venv .venv
site=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' "$base/lib/python3.11/site-packages" > "$site/existing-cluster.pth"
.venv/bin/python -m pip install --no-cache-dir 'tiktoken==0.9.0' 'matplotlib==3.10.5'
.venv/bin/python -m pip freeze > evidence/cluster-environment.lock.txt
.venv/bin/python -c 'import torch,numpy,tiktoken,matplotlib,pytest; print(torch.__version__,numpy.__version__,tiktoken.__version__,matplotlib.__version__,pytest.__version__)'
