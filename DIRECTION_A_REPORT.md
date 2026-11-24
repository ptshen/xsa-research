# Direction A interim report — screen incomplete

Direction A is implemented and numerically verified. The three-threshold training screen is incomplete because Insomnia SSH is currently unreachable. No final Direction A language-model comparison is available locally.

## Verified work

- GPU verification job **14072510 COMPLETED**, exit 0, 38 seconds on NVIDIA RTX A6000. **49 tests passed**, zero failures or skips.
- Preserved SA/XSA checkpoints reproduce all original validation-window losses exactly (maximum difference 0.0).
- All three cap configurations load the same original initialization. Numerical source/config hashes match the passing CUDA verification.
- Collection rollback and stage-budget guards passed three local tests; the collector has not yet been exercised on three completed cap checkpoints.

## Measured profile

Same A6000, model/context, SDPA, bf16, batch 8, three warmups, twelve measured microsteps, and three accumulation-8 updates. Data I/O, evaluation and checkpoint writes are excluded.

| Method | Threshold | Forward/backward microstep | Full update | Peak allocated memory |
|---|---:|---:|---:|---:|
| sa | None | 31.047 ms | 243.932 ms | 3.603 GiB |
| xsa | None | 34.532 ms | 274.493 ms | 4.205 GiB |
| alignment_cap | 0.25 | 36.339 ms | 288.527 ms | 4.246 GiB |
| alignment_cap | 0.5 | 36.298 ms | 288.292 ms | 4.244 GiB |
| alignment_cap | 0.75 | 36.290 ms | 287.672 ms | 4.246 GiB |

The three cap runs project to about 590 seconds of compute each, before evaluation/checkpoint overhead. Conservative project peak storage was estimated at 7.06 GiB, within the 8 GiB cap.

## Training status and remaining work

| Threshold | Job | Last verified state |
|---|---|---|
| 0.25 | 14072518 | Running on ins093 at update 246; current outcome unknown |
| 0.50 | Not submitted | Unrun |
| 0.75 | Not submitted | Unrun |

Recorded reservations: **1980/7200 GPU seconds**, including **180/300 verification/profile seconds**. The 38-second verification runtime is confirmed; total actual training usage cannot yet be queried. No CPU-only collection job has been submitted.

Next: inspect `squeue`/`sacct` for **14072518** and read its final checkpoint/evaluation/diagnostics. Do not resubmit or infer failure from the SSH timeout. After verifying successful full-budget completion, run thresholds 0.50 and 0.75 sequentially, collect all five methods, audit the learning curves/identities/budgets, and publish the completed report.

Until those steps finish, final losses, perplexities, actual training memory/runtime, cap diagnostics and a complete comparison remain unverified. The results will be one-seed validation screening; neither geometric compliance nor a selected validation winner establishes a general quality improvement.

[Execution protocol and commands](DIRECTION_A_PROTOCOL.md) · [Progress and recovery history](PROGRESS.md) · [Stage ledger](evidence/direction_a/job_ledger.json)
