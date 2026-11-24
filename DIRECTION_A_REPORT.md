# Direction A: completed exploratory screen

**Completed 2026-11-24.** All three caps trained successfully and passed the paired checks. Every tested cap improved on SA in this run, but **none improved on XSA**. The best cap was τ=0.50: validation cross-entropy 4.81451356 versus XSA 4.80333272, a gap of +0.01118083 nats/token. This is a one-seed validation screen, not independent confirmation.

Each method used 2048 updates / **67,108,864 processed training tokens**, identical saved seed-17 initialization, ordered minibatches, data, and evaluation windows. SA/XSA checkpoints were reused after exact re-evaluation; neither baseline was retrained.

## Final validation results

| Method | τ | Cross-entropy | Perplexity | Minus SA | Minus XSA |
|---|---:|---:|---:|---:|---:|
| SA | — | 4.89046994 | 133.01607 | +0.00000000 | +0.08713721 |
| XSA | — | 4.80333272 | 121.91605 | -0.08713721 | +0.00000000 |
| A | 0.25 | 4.82149275 | 124.15028 | -0.06897719 | +0.01816002 |
| A | 0.50 | 4.81451356 | 123.28683 | -0.07595638 | +0.01118083 |
| A | 0.75 | 4.87118640 | 130.47562 | -0.01928354 | +0.06785367 |

Lower loss is better. The paired window-bootstrap 95% intervals for A minus XSA are:

| τ | Interval (nats/token) |
|---:|---:|
| 0.25 | [+0.01062998, +0.02533872] |
| 0.50 | [+0.00370309, +0.01938553] |
| 0.75 | [+0.05724926, +0.07970203] |

These intervals use 2,000 resamples of the same 64 validation windows, with bootstrap seed 2026. They assume independent windows; shared-document correlations may make them optimistic. They measure evaluation-sample uncertainty, not training-seed variability. Threshold selection used this validation set, so its winning score is not an untouched test result.

[Full learning curves](evidence/direction_a/results/curves.png) and [the explicitly zoomed second half](evidence/direction_a/results/curves_zoom.png) show loss against tokens and elapsed training/evaluation time. Both plots were visually inspected. All outcomes and intervals are retained in [summary.json](evidence/direction_a/results/summary.json).

## Implementation and verification

The `alignment_cap` operator splits each ordinary causal SDPA aggregate into parallel and perpendicular components relative to its own value, clips the signed parallel coefficient at the prescribed cosine limit, and applies the original output projection after concatenating heads. It adds no parameters. All models have **29,938,944 total / 10,622,208 nonembedding parameters**, six layers, width 384, six 64-dimensional heads and context 512.

Tau zero invokes the existing XSA correction exactly. Projection reductions use fp32 under bf16, while fp64 references remain fp64. The final bf16 cast introduces small cosine-bound error. Clamped normalization retains the baseline epsilon convention below 1e-12; no strict cosine-bound claim applies there. Collinear and first-position outputs follow the formula and are zero in exact arithmetic for every τ<1; floating-point cancellation may leave small residuals. Zero-output cosine is undefined and is excluded from alignment means, with zero/near-zero fractions reported separately.

- **CUDA: 49 passed, zero failures or skips**, 10.77 s inside job 14072510. Coverage includes geometry, inactive caps, signed alignment, degeneracies, threshold validation, fp64 finite differences, reference/SDPA forward and backward agreement, causality, fp32/bf16 precision, saved initialization and CPU/CUDA resume with optimizer/data/RNG checks.
- Frozen-source regression tests verify unchanged SA/XSA outputs and gradients. Re-evaluating both real baseline checkpoints gave **maximum per-window loss difference 0.0**. The original initialization hash matches every cap.
- Three collection/budget unit tests passed locally after the final collector was added. Two overlap the CUDA suite; these counts must not be summed as distinct tests.
- The CPU collector loaded all five real final checkpoints, checked finite weights and every committed training row, verified identical final data state and all 2048 batch hashes, checked complete budgets/configuration identities, and computed latest/previous checkpoint checksums.
- The independent compact-artifact audit reproduced means, perplexities and the comparison summary; checked manifests, software versions, parameter counts, diagnostic dimensions/ranges and scheduler bounds; and verified that numerical sources/configs still match the CUDA gate.

Evidence: [CUDA output](evidence/direction_a/tests.txt), [JUnit cases](evidence/direction_a/tests.xml), [verification proof](evidence/direction_a/verification.json), [checkpoint identities](evidence/direction_a/results/identity.json), [independent audit](evidence/direction_a/final_audit.json), and [requirement audit](DIRECTION_A_COMPLETION_AUDIT.md). Red/green artifacts remain under `evidence/direction_a/`.

The implementation was published as `57aa6e2`; baseline source is preserved from `3e81815`. Subsequent commits changed collection/documentation/artifacts. The recorded numerical file hashes remained unchanged throughout screening. Actual cluster software is Python 3.11.5, torch 2.10.0+cu128, NumPy 2.4.6 and CUDA 12.8; full per-run software records are in the manifests.

## Runtime, memory and resource use

| Job | Work | Final state | Allocated time |
|---|---|---|---:|
| 14072510 | CUDA correctness, baseline rechecks and all five profiles | COMPLETED, 0:0 | 38 s GPU |
| 14072518 | A τ=0.25, evaluation and diagnostics | COMPLETED, 0:0 | 663 s GPU |
| 14074753 | A τ=0.50, evaluation and diagnostics | COMPLETED, 0:0 | 690 s GPU |
| 14074780 | A τ=0.75, evaluation and diagnostics | COMPLETED, 0:0 | 658 s GPU |
| 14074923 | Full checkpoint validation and collection | COMPLETED, 0:0 | 36 s CPU |

Actual stage GPU allocation was **2049 seconds (34m09s; 0.5692 GPU-hours)**, including 38 verification/profile seconds. Full requested GPU reservations totaled **5580/7200 seconds**, with **180/300 seconds** reserved for verification/profile. CPU-only collection used **36 seconds**, reserved 300/900 seconds. All GPU jobs used one A6000, `edu/short`, 8 CPUs and 32 GiB on ins093, sequentially; collection used 2 CPUs/8 GiB on ins069. Final accounting includes start/end times and allocated resources. No active jobs remained at the final queue check.

Project storage at collection was **6.701 GiB**, below 8 GiB. The conservative temporary-checkpoint peak estimate was **7.06 GiB**; peak disk use was not continuously sampled. Source data, caches, test artifacts, original logs, initialization, and latest/previous checkpoints remain preserved. The original Part 1 smoke/profile accounting remains 49 actual seconds and 480 reserved seconds of its separate 600-second limit.

[Final accounting](evidence/direction_a/final_accounting.psv), [stage ledger](evidence/direction_a/job_ledger.json), [queue snapshot](evidence/direction_a/final_queue.txt), and [manifest](DIRECTION_A_MANIFEST.json) contain the audit evidence.

Measured pilot-shape profile, all methods on the same A6000 with SDPA, bf16/fp32 correction, batch 8, three warmup microsteps, twelve measured microsteps and three accumulation-8 updates:

| Method | τ | Forward/backward microstep | Full update | Profile peak allocated |
|---|---:|---:|---:|---:|
| SA | — | 31.047 ms | 243.932 ms | 3.603 GiB |
| XSA | — | 34.532 ms | 274.493 ms | 4.205 GiB |
| A | 0.25 | 36.339 ms | 288.527 ms | 4.246 GiB |
| A | 0.50 | 36.298 ms | 288.292 ms | 4.244 GiB |
| A | 0.75 | 36.290 ms | 287.672 ms | 4.246 GiB |

Direction A full-update overhead was **4.80–5.11% relative to XSA**, or **17.93–18.28% relative to SA**, in this short profile. The common benchmark optimizer harness excludes data I/O, evaluation and checkpoint writes and is not a repeated end-to-end speed study. Raw measurements are in [profile.json](evidence/direction_a/profile.json).

The checkpoints also record cumulative training-loop elapsed time. It includes in-loop evaluation and previous checkpoint writes, but excludes startup and later external evaluation/diagnostics:

| Method | τ | Loop elapsed | Processed tokens/s | Full-training peak allocated |
|---|---:|---:|---:|---:|
| SA | — | 582.225 s | 115,263 | Not recorded |
| XSA | — | 640.602 s | 104,759 | Not recorded |
| A | 0.25 | 652.795 s | 102,802 | 3.753 GiB |
| A | 0.50 | 659.220 s | 101,800 | 3.753 GiB |
| A | 0.75 | 646.338 s | 103,829 | 3.753 GiB |

Baseline full-training peaks were not instrumented in the earlier pilot, so they are not inferred from the new profile. The cap training peaks and profile peaks come from different measured processes. Full-run timings also include I/O and node/date variation; the earlier SA/XSA training used ins092. See [timing.json](evidence/direction_a/results/timing.json).

## Diagnostics

All runs use the same four fixed 128-token training windows (sampling seed 7017), with float32 diagnostic forward passes. Each JSON retains all six layers × six heads: signed alignment before/after correction, aggregate/corrected/removed-update norms, exact-zero and near-zero output fractions. Caps also record activation fraction. The near-zero diagnostic cutoff is 1e-6.

Cap activation percentages averaged across the six heads within each layer:

| Layer | τ=0.25 | τ=0.50 | τ=0.75 |
|---:|---:|---:|---:|
| 1 | 80.01% | 60.19% | 37.86% |
| 2 | 99.71% | 98.73% | 99.54% |
| 3 | 99.90% | 93.36% | 74.15% |
| 4 | 99.97% | 96.78% | 98.80% |
| 5 | 99.48% | 99.12% | 84.38% |
| 6 | 98.83% | 95.51% | 94.66% |

The cap was frequently active in these trained models, especially in later layers. That describes the measured mechanism; it does not explain causally why a method helped or hurt. The sample is small, and changed model weights prevent interpreting cross-model norm differences as an isolated intervention.

Full layer/head records: [τ=0.25](evidence/direction_a/results/direction-a-tau025-s17/diagnostics.json), [τ=0.50](evidence/direction_a/results/direction-a-tau050-s17/diagnostics.json), [τ=0.75](evidence/direction_a/results/direction-a-tau075-s17/diagnostics.json), [SA](evidence/direction_a/results/pilot-sa-s17/diagnostics.json), [XSA](evidence/direction_a/results/pilot-xsa-s17/diagnostics.json).

## Actual commands and recovery

Executed from `~/algollms/xsa` on Insomnia. These keys are already used; inspect their existing jobs rather than resubmitting:

```bash
/usr/bin/python3 -m xsa.stage_submission verify --key direction-a-verify-v1 --seconds 180
/usr/bin/python3 -m xsa.stage_submission train --tag 025 --key direction-a-tau025-v1 --seconds 1800
/usr/bin/python3 -m xsa.stage_submission train --tag 050 --key direction-a-tau050-v1 --seconds 1800
/usr/bin/python3 -m xsa.stage_submission train --tag 075 --key direction-a-tau075-v1 --seconds 1800
/usr/bin/python3 -m xsa.stage_submission collect --key direction-a-collect-v1 --seconds 300
```

The stage scripts ran `xsa.train`, then `xsa.evaluate` and `xsa.diagnostics` within each training allocation. The CPU job ran `.venv/bin/python -m xsa.direction_a collect`. Actual `sbatch` argument arrays and scheduler snapshots are preserved in the ledger. The [protocol](DIRECTION_A_PROTOCOL.md) documents configuration, precision and resume conventions.

Independent local analysis was executed with:

```bash
PYTHONPATH=. python3 evidence/direction_a/audit_collected.py
```

Red tests initially failed for the absent cap, missing threshold/activation metadata, and missing rollback-aware collection; the corresponding implementations passed their green checks. No training job failed or was retried. SSH reachability failed after the first threshold started, blocking observation. On reconnection, job 14072518 was found COMPLETED; its run was not duplicated. The authoritative remote ledger was backed up and reconciled before further submissions. All logs/checkpoints were retained.

## Limitations and next decision

This is a 29.94M model, one seed, one fixed assumed LR recipe, a bounded FineWeb prefix, and only 32,768 validation target tokens. The three thresholds were selected/evaluated on that validation set. There is no untouched test result or training-seed uncertainty estimate. KV-cached decoding is not implemented. Downstream tasks, other model sizes/contexts, Directions B/C, extra seeds and mechanistic controls remain unrun. This does not reproduce the paper-scale study.

The screen provides no observed advantage for these caps over XSA under this recipe. If the mechanistic question remains the priority, the next small control would compare fixed-strength partial removal with equal tuning allowance before attributing anything to adaptive clipping. A three-strength pilot control screen would project to about **0.55 GPU-hours** using the measured 663-second XSA allocation as a proxy, or **1.5 GPU-hours reserved** with three 30-minute caps. Retaining its latest/previous checkpoints would bring projected project peak to roughly **9.1 GiB**, above the current 8 GiB cap. That control still needs implementation/testing/profiling; this estimate excludes those costs and queue time.

A separate two-fresh-seed comparison of SA, XSA and frozen A τ=0.50 would project to **1.10 GPU-hours actual allocation**, **3 GPU-hours reserved**, and roughly **11.3 GiB peak project storage** including two new initial states. Independent evaluation data would also need a bounded preparation protocol. These are planning estimates, not authorizations or launched work. No further experiment is scheduled.
