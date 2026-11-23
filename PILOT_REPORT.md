# Exploratory SA/XSA pilot results

Both arms completed 2048 updates / 67,108,864 training tokens on NVIDIA RTX A6000, with identical saved initialization and all 2048 ordered minibatch hashes. Actual allocated GPU time: **1287 seconds**, within the approved 3600 seconds. Project storage at collection: 4.430 GiB, within 5 GiB.

| Method | Held-out cross-entropy | Perplexity |
|---|---:|---:|
| SA | 4.89046994 | 133.01607 |
| XSA | 4.80333272 | 121.91605 |

XSA minus SA loss: **-0.08713721**. Paired evaluation-window bootstrap 95% interval: [-0.09983557593077422, -0.07541421875357629]. This is evaluation-sample uncertainty with an independent-window approximation; shared-document correlations may make it optimistic. There is one training seed, so no training-seed uncertainty estimate is available.

The pinned FineWeb prefix contains 99,962,616 training tokens from 146,329 documents and 36,418 validation tokens from 74 documents. Evaluation uses the same 64 nonoverlapping 512-token target windows (32,768 evaluated tokens). Tokenizer: GPT-2; document-preserving split seed 2026. This is a scaled-down prefix-data study, not paper reproduction. Downstream tasks and further seeds remain unrun.

[Learning curves](evidence/pilot/curves.png), [paired results](evidence/pilot/paired_results.json), [batch identity](evidence/pilot/pair_identity.json), [scheduler accounting](evidence/pilot/accounting.psv), and [manifest](PILOT_MANIFEST.json) contain the evidence. Diagnostics and all learning-curve rows are included in each arm's artifact directory. Checkpoints and prior checkpoints remain under `~/algollms/xsa/runs/pilot-{sa,xsa}-s17/`.

The first HTTP preparation attempt was cancelled for insufficient throughput; the parallel-download retry completed successfully. Its original log and data directory were retained. Only its abandoned partial-download cache was removed after the full pinned source was verified present.

No further training is automatically launched. Inspect these exploratory results before selecting any additional seeds, LR search, or extension experiments.

## Execution, verification and measured next-stage estimate

| Job | Work | Final state | Allocated time |
|---|---|---|---:|
| 14072309 | Initial HTTP preparation | CANCELLED deliberately | 188 s CPU |
| 14072321 | Bounded preparation and shared initialization | COMPLETED | 163 s CPU |
| 14072336 | New cap/gate tests and compute-node GitHub access | COMPLETED | 15 s CPU |
| 14072327 | SA, evaluation and diagnostics | COMPLETED | 624 s GPU |
| 14072338 | XSA, evaluation and diagnostics | COMPLETED | 663 s GPU |
| 14072339 | Pair validation, collection and GitHub publication | COMPLETED | 19 s CPU |

The successful jobs all exited 0:0. Training used account `edu`, partition `short`, one RTX A6000 on `ins092`, 8 CPUs and 32 GiB RAM per arm. The 1287-second GPU allocation is **0.3575 GPU-hours**. The original Part 1 smoke/profile budget remains **49/600 actual seconds**; this pilot used the subsequent explicit authorization. No full matrix or downstream task was launched.

The executed preparation/training/dependency commands are recorded in [README.md](README.md#approved-pilot-execution) and the exact scheduler argument arrays in [the job ledger](evidence/job_ledger.json). The data preparation section includes the 150,000-document, 100,000,000-token and one-shard caps. The completion check/publication entry point was:

```bash
ssh insomnia
cd ~/algollms/xsa
# Executed by the dependent CPU job 14072339:
.venv/bin/python -m xsa.pilot finish
# Read-only final verification:
sacct -j 14072309,14072321,14072327,14072336,14072338,14072339 -X -o JobID,State,Elapsed,ExitCode
squeue -u "$USER"
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

The pilot additions enforce document-preserving preparation caps, require SA to finish its complete token budget before XSA starts, and collect/publish results only after identity and resource checks. Red/green evidence is preserved for the data cap and completion gate. The two new tests passed on a compute node and locally; the earlier numerical/resume suite passed all 26 tests on CUDA. An independent local audit of the collected artifacts verified 2048 consecutive finite updates per arm, identical ordered batch hashes, matching checkpoint/data checksums, and exactly reproduced the paired summary. The learning-curve plot was visually checked. See [final audit](evidence/pilot/final_audit.json) and [pilot environment](evidence/pilot_environment.json).

The training-loop elapsed values were **581.262 s SA / 639.694 s XSA**, including in-loop validation and previous checkpoint writes but excluding final evaluation/diagnostics and process startup. XSA took **10.05% longer** by this measure; the complete allocations differed by **6.25%**, which also reflects startup and I/O variation. These are one sequential pair, not repeated speed measurements.

For the same architecture, token budget, cached data and hardware, two additional paired seeds (29 and 43 in `configs/confirmation.json`) project to **0.715 GPU-hours of actual allocated time** using this measured pair. Keeping the same hard caps would reserve **2 GPU-hours total** across four sequential jobs. Retaining their latest/previous checkpoints and initial states would bring project storage to roughly **7.3 GiB final / 7.7 GiB during an atomic save**, exceeding the present 5 GiB authorization. Queue time, data variation and system contention are excluded from this estimate. This is a planning estimate only; those seeds and any revised storage allowance are unapproved and unlaunched.

The bounded diagnostics use only four 128-token training windows and a separate float32 forward; they do not reproduce the paper's diagnostic sample size/context. The pilot uses a fixed assumed LR, a small prefix and only 32,768 held-out target tokens. Its window-bootstrap interval measures evaluation-sample uncertainty, not variation across training seeds. A seed confirmation and any larger study must retain these distinctions.
