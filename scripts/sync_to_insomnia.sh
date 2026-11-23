#!/bin/bash
set -euo pipefail
# Never deletes remote outputs; excludes runtime data in both directions.
rsync -az --exclude='.git' --exclude='.venv*' --exclude='__pycache__' \
  --exclude='.pytest_cache' --exclude='data' --exclude='runs' --exclude='/logs/' \
  --exclude='cache' --exclude='evidence/job_ledger.json' \
  --exclude='evidence/submission.lock' --exclude='build' --exclude='*.egg-info' \
  --exclude='*.whl' --exclude='evidence/cluster/cluster/' \
  --exclude='evidence/cluster/results/' ./ insomnia:algollms/xsa/
