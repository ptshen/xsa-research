#!/bin/bash
set -euo pipefail
# Never deletes remote outputs; excludes runtime data in both directions.
rsync -az --exclude='.git' --exclude='.venv*' --exclude='__pycache__' \
  --exclude='.pytest_cache' --exclude='data' --exclude='runs' --exclude='/logs/' \
  --exclude='cache' --exclude='evidence/job_ledger.json' \
  --exclude='evidence/submission.lock' --exclude='evidence/direction_a/job_ledger.json' --exclude='evidence/direction_a/submission.lock' --exclude='build' --exclude='*.egg-info' \
  --exclude='evidence/direction_bc/job_ledger.json' --exclude='evidence/direction_bc/submission.lock' \
  --exclude='*.whl' --exclude='evidence/cluster/cluster/' \
  --exclude='evidence/cluster/results/' ./ insomnia:algollms/xsa/
