#!/bin/bash
set -euo pipefail
mkdir -p evidence/cluster evidence/results
rsync -az --exclude='/cluster/' --exclude='/results/' --exclude='*.lock' \
  insomnia:algollms/xsa/evidence/ evidence/cluster/
rsync -az insomnia:algollms/xsa/logs/ evidence/cluster/logs/
# Preserve checkpoints on Insomnia, collect compact results/plots/logs locally.
rsync -az --exclude='*.pt' --exclude='*.tmp' insomnia:algollms/xsa/runs/ evidence/results/
