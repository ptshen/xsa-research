# Approved pilot queued

Data preparation is complete. The two 67,108,864-token training arms are queued on Insomnia, with one A6000 active at a time. Approved allocation: **3600 total GPU seconds** and **5 GiB project storage**. Source, logs and checkpoints remain under `~/algollms/xsa`.

| Stage | Job ID | Status at submission |
|---|---|---|
| Initial HTTP preparation | 14072309 | Cancelled after 3m08s for insufficient throughput; log preserved |
| Parallel preparation | 14072321 | Completed in 2m43s; verified data and shared initialization |
| Compute-node gate/auth verification | 14072336 | Submitted CPU-only |
| SA | 14072327 | Pending resources, 30-minute GPU cap |
| XSA | 14072338 | Dependency `afterok:14072327`, 30-minute GPU cap |
| Verify, collect and publish | 14072339 | CPU-only, dependency `afterok:14072338`, 3-minute cap |

Slurm's most recent SA estimate was **17:11 EST on 2026-11-23**; queue estimates can change. These are queued jobs, not completed training results.

XSA first checks that SA's evaluation reports all 2048 updates / 67,108,864 tokens. Failed or incomplete SA prevents XSA training. The collector requires both full budgets, identical initialization and all ordered batch hashes, matching evaluation windows/GPU type, successful scheduler completion, GPU usage ≤3600 seconds, and project storage ≤5 GiB. Only then does it create `PILOT_REPORT.md`, `PILOT_MANIFEST.json`, curves, paired summaries and compact logs, and commit/push them to GitHub `main`. Checkpoints stay on Insomnia.

No retries, additional seeds, downstream benchmarks or larger studies launch automatically. A failed job leaves its logs/checkpoints for diagnosis. Do not resubmit without checking these exact IDs and `evidence/job_ledger.json`.

```bash
ssh insomnia
cd ~/algollms/xsa
squeue -j 14072327,14072338,14072339
sacct -j 14072309,14072321,14072336,14072327,14072338,14072339 -X -o JobID,JobName,State,Elapsed,ExitCode
```
