#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
export HF_HOME="$PWD/cache/huggingface" TIKTOKEN_CACHE_DIR="$PWD/cache/tiktoken"
srun .venv/bin/python -m xsa.data fineweb --output "$XSA_DATA" --max-documents "$XSA_MAX_DOCUMENTS"
.venv/bin/python -m xsa.data validate --output "$XSA_DATA"
