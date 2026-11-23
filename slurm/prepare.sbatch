#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
export HF_HOME="$PWD/cache/huggingface" TIKTOKEN_CACHE_DIR="$PWD/cache/tiktoken"
export HF_HUB_DISABLE_XET=0 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
limits=()
if [[ -n "${XSA_MAX_TOKENS:-}" ]]; then limits+=(--max-tokens "$XSA_MAX_TOKENS"); fi
if [[ -n "${XSA_MAX_SHARDS:-}" ]]; then limits+=(--max-shards "$XSA_MAX_SHARDS"); fi
srun .venv/bin/python -m xsa.data fineweb --output "$XSA_DATA" --max-documents "$XSA_MAX_DOCUMENTS" "${limits[@]}"
.venv/bin/python -m xsa.data validate --output "$XSA_DATA"
if [[ "${XSA_PREPARE_INITIAL:-0}" == 1 ]]; then
  mkdir -p runs/initial
  .venv/bin/python -m xsa.initialize --config configs/pilot.json --seed 17 --output runs/initial/pilot-s17.pt
fi
