# Directions B/C: frozen seed-17 screening protocol

This new stage implements the two remaining operators and four full pilot runs. Existing SA, XSA and three A runs are preserved and reused after source regression, numerical re-evaluation and identity checks. Negative outcomes are retained. This is an exploratory validation screen, not paper-scale reproduction or independent confirmation.

## Operators and numerical policy

B (`value_penalty`) subtracts lambda times positive value cosine from scaled causal QK logits, including the diagonal, then softmaxes and aggregates the original values. Lambda is 0.1, 0.5 or 1.0. The implementation uses explicit fp32 attention with 128-query chunks; original absolute query positions determine the mask. Gradients flow through both normalization and similarity. Float64 remains float64 for references. Autocast is disabled within these reductions/matrix multiplies, and the output returns to the incoming dtype. No detachment or extra trainable parameters occurs. Clamped normalization epsilon is 1e-12; zero values have zero normalized value, and below-epsilon similarities are damped. First-position output remains its own value. Lambda zero matches ordinary attention within reference/backend tolerances. Query chunking limits individual temporaries; no unmeasured total-training memory reduction is claimed. The configured ordinary backend remains SDPA for baseline comparison/diagnostics, but B's training path is explicitly labeled as explicit attention.

C (`output_geometry`) computes each head's Gram matrix from the current output weight on every forward pass. PyTorch's output weight is `[d_model, heads*head_dim]`; the head block is a contiguous column slice. The fp32 Gram operation and projection preserve gradients through that weight; float64 references remain float64. The correction is applied before concatenation. Correction requires the squared written-value denominator to exceed `max(projection_eps**2, machine_epsilon * (d_model + 2*head_dim) * ||W_h||_F**2 * ||v||**2)`. Here `projection_eps**2 = 1e-24`; the relative floor accounts conservatively for finite-precision Gram/dot-product resolution. Unresolved near-null directions skip correction. The denominator is replaced by one in guarded branches before division, avoiding both zero division and huge unused intermediate quotients. This is a guard, not ridge regularization or a matrix inverse. Roundoff-negative denominators also take the guard. The per-head written-space orthogonality claim applies only to nondegenerate, numerically resolved values, and bf16 casting relaxes it. First-position nondegenerate outputs are zero in exact arithmetic. There is no claim about the summed heads or residual stream. SDPA remains the aggregate path.

SA/XSA/A arithmetic is unchanged and checked against `evidence/direction_bc/baseline_model.py` frozen from 8a1093a. No KV cache is added. Both methods retain identical parameter names, shapes, initialization and counts.

## Frozen experimental settings

Four configurations: `configs/direction_b010.json`, `direction_b050.json`, `direction_b100.json`, and `direction_c.json`. All inherit the verified pilot: 29,938,944 total / 10,622,208 nonembedding parameters; six layers, width 384, six 64-dimensional heads, context 512, bf16, dropout zero. AdamW recipe, LR schedule, evaluation and checkpoint intervals are unchanged. Shared initialization is `runs/initial/pilot-s17.pt`; data is `data/pilot-v2`; seeds are 17/1017/2017. Each run processes exactly 2048 updates * 32768 tokens = 67,108,864 training tokens. Training samples with replacement. The 64 original nonoverlapping validation windows contain 32,768 target tokens. No new data download or test split is introduced.

Only attention type, lambda/chunk settings and descriptive label vary. The architecture hash excludes shape-neutral method settings; the comparison hash excludes the declared intervention settings but preserves every other scientific setting. Resume requires exact full config equality, including chunk size, epsilon, and lambda. Raw resolved configs remain in each run manifest/checkpoint.

## Correctness, profiling, and freeze

Red/green tests precede implementation. The full CPU suite must pass with CUDA-only cases reported as skipped locally. An allocated A6000 verification job runs the full suite with no skips, re-evaluates all five preserved checkpoints (per-window loss tolerance 1e-6), checks shared initialization and batch histories, and profiles all nine configurations at batch 8/context 512. Profiling uses three warmup microsteps, twelve synchronized forward/backward steps, optimizer timing, and three actual accumulation-8 updates. It uses the same AdamW parameter groups as training. An extra untimed profiler step identifies actual kernel dispatch. Peak allocation includes forward/backward and optimizer state; diagnostics are disabled. The short profile excludes data I/O, validation, diagnostics and checkpoint writes, and is not a repeated end-to-end speed study.

`DIRECTIONS_BC_PLAN.json` is generated by that gate before screening. It reserves each run's measured compute extrapolation *1.5 +120 seconds for startup/I/O/evaluation/diagnostics, rounded up to a minute with a 20-minute minimum. All four runs plus the full 600-second verification allowance must fit 14400 seconds. Storage estimation includes four new latest/previous pairs, one temporary full checkpoint, and 128 MiB metadata/test allowance, within both project cap and current home quota. Source, config, tests, protocol and numerical-script hashes freeze at the successful gate; subsequent changes require re-verification within the stage budget.

## Authorization, accounting, and commands

Separate B/C caps: 14400 GPU seconds total; 600 verification/smoke/profile seconds within it; 1800 allocated CPU-only seconds; 12 GiB total project storage including transient headroom. Historical Part 1 and A budgets/ledgers are immutable. Existing quota and resources are checked. The default verified allocation is `edu/short`, one A6000, 8 CPUs/32 GiB; collection is 2 CPUs/8 GiB. Scheduler parameters remain configurable. No paid services, automatic requeues, or concurrent project jobs.

Run commands on Insomnia in `~/algollms/xsa`. Do not reuse submission keys:

```bash
/usr/bin/python3 -m xsa.bc_submission verify --key direction-bc-verify-v1 --seconds 300
# Read verification.json and DIRECTIONS_BC_PLAN.json; use the corresponding frozen seconds:
/usr/bin/python3 -m xsa.bc_submission train --tag b010 --key direction-bc-b010-v1 --seconds SECONDS
/usr/bin/python3 -m xsa.bc_submission train --tag b050 --key direction-bc-b050-v1 --seconds SECONDS
/usr/bin/python3 -m xsa.bc_submission train --tag b100 --key direction-bc-b100-v1 --seconds SECONDS
/usr/bin/python3 -m xsa.bc_submission train --tag c --key direction-bc-c-v1 --seconds SECONDS
/usr/bin/python3 -m xsa.bc_submission collect --key direction-bc-collect-v1 --seconds 300
```

Each submission checks prior IDs via sacct, current queue, uncertain ledger entries, reservations, gate hashes and storage. Requested walltime is reserved atomically before calling sbatch. Failed/uncertain submissions retain reservations until explicitly reconciled; no automatic reclaim. Train sequentially and inspect terminal state plus complete artifacts before the next run. An interrupted run resumes the same directory with `--resume`, a new key and a budgeted continuation; first inspect its job and checkpoint. Completed runs are rejected. Signal handling and atomic latest/previous checkpoints retain optimizer/RNG/data/schedule progress. An SSH observation failure never implies job failure.

## Evaluation and diagnostics

Primary metric: held-out token-weighted next-token cross-entropy in nats/token. Perplexity is exp(CE), giving the same ranking and no independent evidence. Report all nine configurations, paired loss differences versus both baselines, and full/zoomed curves versus tokens and training/evaluation elapsed time. Report allocated runtime separately from loop time, and profile memory separately from full-run memory. Historical SA/XSA full-training peak memory was not recorded; keep it explicitly missing rather than inventing a measurement.

Use the same four 128-token training diagnostic windows, sampling seed 7017, float32 forward. Each layer/head is retained. B reports actual penalized diagonal mass, positive value similarity weighted by attention before/after penalty, and mean applied penalty over all causal pairs including self. Ordinary pre-intervention aggregates use that trained model's current weights. C reports head-space and written-space alignment, written correction/output norms, denominator min/5th/50th/95th/max quantiles and guard fraction. The guard threshold mean is recorded with its activation fraction. Denominator summaries are computed per sequence then averaged across the four sequences; they are not pooled quantiles. Cosines exclude norms <=1e-6; zero/near-zero fractions are separate. None of these geometric diagnostics alone establishes prediction quality or contextual reasoning.

Paired window bootstrap: 2000 resamples, RNG 2026; independence approximation may be optimistic for correlated same-document windows. It does not estimate seed variability or correct selection bias. No untouched test, downstream task, mechanistic control, extra seed, model size or context sweep is authorized here. The final report will estimate a separately authorized confirmation study from measured costs.

## Numerical hardening before C training

A post-gate fp32 rank-one/null-space stress test exposed a positive Gram-roundoff denominator despite a computed zero written self-value, amplifying a normal-sized aggregate to norm 313.35. The absolute-only guard was replaced with the dimension/dtype/scale-aware threshold above, with failing/passing evidence preserved. The original gate/profile is archived under `evidence/direction_bc/verification_v1/`. No C screening run used the original guard. B lambda 0.1 started under commit 1db7eff; its operator is unchanged, and additional frozen-source B regression tests verify exact outputs/gradients. The hardened source requires a second allocated verification gate before the remaining runs. Both verification reservations count toward the same 600-second stage cap.
