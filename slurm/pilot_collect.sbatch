#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
set -euo pipefail
cd "${XSA_ROOT:-$HOME/algollms/xsa}"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
.venv/bin/python -m xsa.pilot finish
# Stage only this completed pilot's compact results; preserve all binary checkpoints.
git add PILOT_REPORT.md PILOT_MANIFEST.json PILOT_STATUS.md PROGRESS.md evidence/pilot
git diff --cached --check -- PILOT_REPORT.md PILOT_MANIFEST.json PROGRESS.md
git commit -m "Record completed one-seed FineWeb SA/XSA pilot results"
git push origin main
printf 'Pilot results published: '
git rev-parse HEAD
