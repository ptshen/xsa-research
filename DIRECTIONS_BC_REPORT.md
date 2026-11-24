# Directions B/C: completed exploratory screen

**Completed 2026-11-24.** All four new methods/settings reached the full matched token budget and passed independent collection. **No tested extension beats XSA in this one-seed validation screen.** C is the strongest extension: cross-entropy **4.80739097**, versus XSA **4.80333272**, a gap of **+0.00405825 nats/token**. Its paired-window 95% interval versus XSA is **[−0.00311979, +0.01139933]**; this screen does not resolve that small gap or establish equivalence. B's best setting, λ=0.5, has loss 4.88045164 and substantially higher measured compute cost than XSA.

Existing SA, XSA and three A runs were reused after exact GPU re-evaluation. No additional seed, control, downstream benchmark or full training matrix was launched. The frozen protocol is [DIRECTIONS_BC_PROTOCOL.md](DIRECTIONS_BC_PROTOCOL.md).

## Nine-configuration validation results

Cross-entropy in nats/token is primary; lower is better. Perplexity is its exponential transformation and provides the same ranking, not independent evidence or task accuracy.

| Method | Cross-entropy | Perplexity | Minus SA | Minus XSA |
|---|---|---|---|---|
| SA | 4.89046994 | 133.01607 | +0.00000000 | +0.08713721 |
| XSA | 4.80333272 | 121.91605 | -0.08713721 | +0.00000000 |
| A τ=0.25 | 4.82149275 | 124.15028 | -0.06897719 | +0.01816002 |
| A τ=0.50 | 4.81451356 | 123.28683 | -0.07595638 | +0.01118083 |
| A τ=0.75 | 4.87118640 | 130.47562 | -0.01928354 | +0.06785367 |
| B λ=0.1 | 4.88484551 | 132.27003 | -0.00562443 | +0.08151279 |
| B λ=0.5 | 4.88045164 | 131.69013 | -0.01001830 | +0.07711892 |
| B λ=1.0 | 4.90996214 | 135.63428 | +0.01949220 | +0.10662942 |
| C | 4.80739097 | 122.41182 | -0.08307897 | +0.00405825 |


Paired window-bootstrap 95% intervals for the loss differences:

| Method | Minus SA interval | Minus XSA interval |
|---|---|---|
| SA | [+0.000000, +0.000000] | [+0.075414, +0.099836] |
| XSA | [-0.099836, -0.075414] | [+0.000000, +0.000000] |
| A τ=0.25 | [-0.080860, -0.058496] | [+0.010630, +0.025339] |
| A τ=0.50 | [-0.087241, -0.064989] | [+0.003703, +0.019386] |
| A τ=0.75 | [-0.024861, -0.013482] | [+0.057249, +0.079702] |
| B λ=0.1 | [-0.011562, +0.000535] | [+0.068066, +0.096407] |
| B λ=0.5 | [-0.016053, -0.004111] | [+0.066366, +0.089430] |
| B λ=1.0 | [+0.011361, +0.027893] | [+0.090359, +0.124687] |
| C | [-0.094389, -0.072003] | [-0.003120, +0.011399] |


These use 2,000 resamples of the same 64 windows, RNG seed 2026. Independence is an approximation: same-document correlations can make intervals optimistic. They do not measure training-seed variability or correct configuration-selection bias. Selection uses this validation set, not an untouched test.

B λ=0.1 and 0.5 improve slightly on SA, while λ=1.0 is worse than SA. Every B setting is worse than XSA. C improves on SA and has the smallest observed gap to XSA among all seven extensions/settings, including A. The observed geometric interventions do not establish a causal explanation for these losses.

[Full learning curves](evidence/direction_bc/results/curves.png) and [the explicitly zoomed second half](evidence/direction_bc/results/curves_zoom.png) show validation loss against processed tokens and training/evaluation-loop time. Both were visually inspected. Raw per-window losses, every training row, and all comparisons remain in [results](evidence/direction_bc/results/summary.json).

## Implementation and numerical conventions

**B (`value_penalty`)** subtracts `λ max(0, cosine(v_i,v_j))` from causal `q_i·k_j/sqrt(d_head)` logits, includes the diagonal, applies softmax afterward, and aggregates the original values. Gradients flow through both similarity and aggregation. Lambda is finite and nonnegative; lambda zero matches SA outputs and gradients, including the explicit penalty path. First-position output remains its own value. The practical path uses 128-query chunks with absolute query indices for causal masking. Clamped normalization has epsilon 1e-12; zero vectors normalize to zero and sub-epsilon values have damped normalization. Scores, penalty, softmax and aggregation use fp32 for low-precision input with autocast disabled; float64 references stay float64 and output returns to the input dtype. Chunking bounds individual temporary score matrices; retained autograd tensors can still dominate backward memory.

**C (`output_geometry`)** slices `c_proj.weight [d_model, heads*head_dim]` into contiguous head-column blocks `W_h`, recomputes `G_h=W_h.T@W_h` each forward, and applies `z=y−[(v.T@G_h@y)/(v.T@G_h@v)]v` before head concatenation/output projection. Gradients flow through the metric and output weights. There is no inverse, ridge, detached metric or new trainable parameter. Ordinary aggregation retains flash SDPA. Geometry and reductions use fp32 under bf16, or float64 for references.

C divides only when the denominator exceeds `max(eps**2, machine_epsilon*(d_model+2*head_dim)*||W_h||_F**2*||v||**2)`, with epsilon 1e-12 and squared absolute floor 1e-24. A guarded branch substitutes denominator one **before division**, then applies zero correction. This skips unresolved near-null directions and roundoff-negative denominators without creating huge unused quotients. Per-head written-space orthogonality `(W_h z).T(W_h v)=0` applies only to resolved nondegenerate inputs within numerical tolerance. Guarding, bf16 casting and output-projection arithmetic weaken that guarantee. It does not apply to the summed heads or residual stream. Nondegenerate first-position output is zero in exact arithmetic; cancellation can leave a residual.

[model.py](xsa/model.py) contains the operators. Training/evaluation/diagnostics preserve explicit method metadata; `backend=sdpa` in the common configuration does not describe B's actual attention path. [bc_submission.py](xsa/bc_submission.py) reserves walltime atomically and rejects duplicates, uncertain submissions, nonterminal prior jobs, changed gate hashes and completed training directories. [direction_bc.py](xsa/direction_bc.py) freezes the measured plan and collects exactly nine configurations. Atomic latest/previous checkpoints retain model, optimizer, schedule, RNG and data state. Resume requires the exact resolved configuration, including lambda, chunk size and epsilon. Neither method adds parameters or changes existing SA/XSA/A behavior.

## Matched protocol and provenance

All models have **29,938,944 unique parameters**, including 10,622,208 nonembedding parameters: six layers, width 384, six 64-dimensional heads, context 512, vocabulary 50,304, tied embedding/output head and zero dropout. Model/data/evaluation seeds are **17/1017/2017**. All share `runs/initial/pilot-s17.pt`, `data/pilot-v2`, bf16 training with fp32 operator reductions, optimizer/LR schedule and ordered minibatches. Every run processes **2048 × 32,768 = 67,108,864 training tokens**; sampling with replacement means these are processed rather than unique tokens.

The existing pinned FineWeb prefix has 99,962,616 training tokens in 146,329 documents and 36,418 validation tokens in 74 documents. Revision is `9bb295ddab0e05d785b879661af7260fed5140fc`. Tokenizer/data manifests and hashes remain preserved; no new data was downloaded. The 64 original nonoverlapping validation windows contain 32,768 target tokens. Equal window lengths make mean window loss equal to token-weighted cross-entropy. Only the declared attention method/settings and descriptive label differ.

The original B/C implementation, first GPU gate and B λ=0.1 run use **1db7eff4ff61ade882e3dc0c3369e817fbc066f6**. The hardened gate and B λ=0.5, B λ=1.0 and C use **a752a9da4a790f6483816c5c1f0cbe14fe5b4b4e**. B's operator is unchanged, established by AST equality and exact output/gradient regression. Later commits collect artifacts or update documentation/accounting; frozen numerical hashes remain identical. [Source provenance](evidence/direction_bc/source_provenance_audit.json) checks both versions. Actual cluster software is Python 3.11.5, torch 2.10.0+cu128, CUDA 12.8 and NumPy 2.4.6; full versions are recorded per run.

## Verification, failure and recovery

- Red tests precede the operators, metadata and strict collection/budget implementation. References and tests cover fp64 finite differences, gradients through values/output weights, causal chunk boundaries, rectangular attention width/head layout, identity-metric C/XSA equivalence, degeneracies and first-position behavior.
- Frozen-source tests confirm unchanged SA/XSA/A outputs, gradients, parameter names/shapes/counts and initialization. CPU/CUDA bf16 tests check exact model/optimizer/data resume and changed-setting rejection. B lambda zero is compared with SA in fp32 and bf16. C's bf16 written-space angle tolerance is 0.01; stricter fp32/fp64 tolerances are explicit in the tests.
- First gate **14076373**: **87 passed, no failures or skips**, 13.55 s of tests, 59 s allocation. All five existing models re-evaluated with maximum per-window loss difference **0.0**.
- Additional fp32 rank-one/null-space stress found a real C flaw: a positive Gram-roundoff denominator despite computed zero written self-value amplified aggregate norm 0.7993 to corrected norm 313.35. [The failing case](evidence/direction_bc/geometry_guard_red.txt) is preserved. The scale-aware guard fixed it; 100 stressed cases now preserve the input exactly when unresolved. **No C training used the defective guard.** The original proof/profile/plan remain under `verification_v1/`.
- Hardened gate **14076777**: **92 passed, zero failures/errors/skips**, 42.10 s tests, 104 s total allocation. All five historical evaluations again match exactly. Local suite: **79 passed, 13 CUDA-only skipped**; those skips are not execution. A separate CPU accounting suite has **10 passing tests** and red/green evidence, with no numerical source changes.

[CUDA output](evidence/direction_bc/tests.txt), [JUnit cases](evidence/direction_bc/tests.xml), [proof](evidence/direction_bc/verification.json) and [guard stress](evidence/direction_bc/geometry_guard_stress.json) preserve the evidence. Numerical/config/test/protocol hashes freeze at the second gate. No training job failed, retried or duplicated. The authoritative remote records were preserved before synchronization, including stash `0c37e7478e0e1f71c99b0b1987db5f9480cd63c5`.

## Independent audits and preservation

The CPU collector loaded all nine real final models and checked finite weights, exact config/data/initialization/evaluation identities, software, all ordered batch hashes, final data state and complete token budgets. The independent checkpoint audit loaded **all 18 latest/previous checkpoints**, checked finite model/optimizer tensors, RNG and batch histories, recomputed CE/perplexity and paired bootstrap intervals, and checked diagnostic shapes/ranges and frozen source/data hashes. Historical checkpoint checksums, compact artifacts and Part 1/A ledgers are unchanged. All logs, data, initialization and checkpoints remain on university storage.

Evidence: [checkpoint identities](evidence/direction_bc/results/identity.json), [independent checkpoint audit](evidence/direction_bc/independent_checkpoint_audit.json), [resource/preservation audit](evidence/direction_bc/final_audit.json), [manifest](DIRECTIONS_BC_MANIFEST.json) and [requirement audit](DIRECTIONS_BC_COMPLETION_AUDIT.md). Publication of scientific commit `c1cfc3265e62911a6d77937f9c567c3b02d3d215` is verified separately in [delivery.json](evidence/direction_bc/delivery.json).


## Runtime, memory and resources

| Job | Work | Node | Allocated seconds | Reserved seconds |
|---|---|---|---|---|
| 14076373 | verify | ins093 | 59 | 300 |
| 14076378 | train b010 | ins093 | 1118 | 1740 |
| 14076777 | verify | ins083 | 104 | 300 |
| 14076783 | train b050 | ins083 | 1153 | 1740 |
| 14076831 | train b100 | ins083 | 1150 | 1740 |
| 14076871 | train c | ins081 | 726 | 1200 |
| 14076904 | collect | ins024 | 75 | 300 |


All jobs are COMPLETED, exit 0:0. Actual B/C-stage GPU allocation is **4310 seconds (71m50s; 1.1972 GPU-hours)**, including **163 verification/profile seconds**. Reservations are **7020/14400 GPU seconds**, **600/600 verification seconds**, and **300/1800 CPU-only seconds**. The CPU collector/audit used **75 allocated seconds**. GPU jobs ran sequentially on one A6000 each, edu/short, 8 CPUs/32 GiB; CPU collection used 2 CPUs/8 GiB. Scheduler start/end/TRES checks confirm no overlap. The final queue is empty.

Final measured project storage is **11,035,931,384 bytes (10.278 GiB)**. The frozen conservative temporary-checkpoint peak estimate was **11.40 GiB**, below the 12 GiB cap and actual home quota. Peak disk was estimated, not continuously sampled. No artifact was deleted to fit. The 50 GiB home quota was inspected with `/usr/lpp/mmfs/bin/mmlsquota -u "$USER" insomnia`; the final capture is retained. Historical Part 1 smoke accounting stays 49 actual/480 reserved seconds, and A stays 2049 actual/5580 reserved GPU seconds.

[Accounting](evidence/direction_bc/final_accounting.psv), [ledger](evidence/direction_bc/job_ledger.json), [storage](evidence/direction_bc/final_storage.json) and [audit](evidence/direction_bc/final_audit.json) contain the evidence.

### Matched pilot-shape profile

One RTX A6000 on ins083; batch 8, context 512, bf16/fp32 reductions; three warmup microsteps, twelve synchronized forward/backward microsteps, three real accumulation-8 updates, and an extra untimed dispatch probe. No compilation. Timing excludes data I/O, validation, diagnostics and checkpoint writes. Profile peak includes optimizer state and the probe.

| Method | Forward/backward microstep (ms) | Full update (ms) | Update overhead vs XSA | Profile peak allocated (GiB) | Observed attention path |
|---|---:|---:|---:|---:|---|
| SA | 31.735 | 245.166 | -11.38% | 3.872 | Flash SDPA |
| XSA | 36.330 | 276.660 | +0.00% | 4.205 | Flash SDPA |
| A τ=0.25 | 38.288 | 290.919 | +5.15% | 4.246 | Flash SDPA |
| A τ=0.5 | 38.529 | 291.168 | +5.24% | 4.244 | Flash SDPA |
| A τ=0.75 | 38.301 | 291.305 | +5.29% | 4.246 | Flash SDPA |
| B λ=0.1 | 64.480 | 518.353 | +87.36% | 4.987 | Explicit fp32 softmax/bmm |
| B λ=0.5 | 64.502 | 518.437 | +87.39% | 4.982 | Explicit fp32 softmax/bmm |
| B λ=1.0 | 64.516 | 517.717 | +87.13% | 4.980 | Explicit fp32 softmax/bmm |
| C | 38.526 | 286.987 | +3.73% | 4.247 | Flash SDPA + Gram bmm |


B's roughly 87% update overhead versus XSA includes replacing flash SDPA with explicit fp32 softmax/bmm. It is not an isolated penalty-kernel comparison, and chunking is not evidence of reduced total backward memory. C retains flash SDPA plus Gram computation, with 3.73% profile overhead versus XSA. Short within-allocation measurements do not establish repeated end-to-end speed differences.

### Complete training runs

Loop time includes in-loop evaluation and earlier checkpoint writes, but excludes startup and later external evaluation/diagnostics. Profile and full-training peaks come from different processes. Full-run timings include node/date and I/O variation; historical baselines ran on ins092.

| Method | Loop seconds | Processed tokens/s | Allocated seconds | Full-training peak GiB |
|---|---|---|---|---|
| SA | 582.225 | 115,263 | 624 | Not recorded |
| XSA | 640.602 | 104,759 | 663 | Not recorded |
| A τ=0.25 | 652.795 | 102,802 | 663 | 3.753 |
| A τ=0.50 | 659.220 | 101,800 | 690 | 3.753 |
| A τ=0.75 | 646.338 | 103,829 | 658 | 3.753 |
| B λ=0.1 | 1108.419 | 60,545 | 1118 | 4.490 |
| B λ=0.5 | 1127.467 | 59,522 | 1153 | 4.490 |
| B λ=1.0 | 1123.020 | 59,757 | 1150 | 4.490 |
| C | 668.439 | 100,396 | 726 | 3.757 |


Historical SA/XSA full-training peak memory was not instrumented; no value is inferred from the profile. Raw session timings are in [timing.json](evidence/direction_bc/results/timing.json).

## Bounded geometry diagnostics


| B λ | Mean causal penalty | Weighted positive similarity before → after | Diagonal mass before → after | Head alignment before → after | Correction norm |
|---|---|---|---|---|---|
| 0.1 | 0.014755 | 0.272485 → 0.265300 | 0.155057 → 0.148589 | 0.365523 → 0.352624 | 0.051835 |
| 0.5 | 0.073134 | 0.275182 → 0.238153 | 0.159186 → 0.126027 | 0.384329 → 0.315732 | 0.269113 |
| 1.0 | 0.146323 | 0.265455 → 0.195648 | 0.156088 → 0.094615 | 0.373081 → 0.239521 | 0.513412 |


C diagnostic means across the six layers and six heads:

| Quantity | Mean |
|---|---|
| alignment_before | 0.73104248 |
| alignment_after | -0.21810075 |
| written_alignment_before | 0.79400216 |
| written_alignment_after | -9.1905414e-05 |
| removed_update_norm | 4.9948132 |
| written_correction_norm | 3.6228311 |
| written_output_norm | 1.6464859 |
| geometry_denominator_min | 9.1600961 |
| geometry_denominator_q05 | 13.334784 |
| geometry_denominator_q50 | 25.973478 |
| geometry_denominator_q95 | 49.27902 |
| geometry_denominator_max | 67.154168 |
| geometry_threshold_mean | 0.033464322 |
| geometry_guard_fraction | 0 |
| zero_output_fraction | 0 |
| near_zero_output_fraction | 0.0069986979 |
| written_zero_output_fraction | 0 |
| written_near_zero_output_fraction | 0.0077582465 |


All B samples have zero exact-zero and near-zero output fractions. Zero and near-zero C fractions are separate from its cosine means; near-zero includes exact zero. The cutoff is 1e-6. A signed mean near zero alone would not prove every vector is orthogonal; independent per-vector numerical tests establish the operator property under their documented tolerances.

All diagnostics use the same four fixed 128-token **training** windows, sampling seed 7017, with float32 forward passes. Each JSON retains six layers × six heads. Tables above average per-head summaries equally; sequences/tokens have equal weight. Cosines exclude output or self-value norms ≤1e-6. B's mean penalty covers all causal pairs including self; weighted similarity sums over keys then averages queries. Before/after aggregates use that trained model's same weights. C denominator min/quantiles/max are computed per sequence and then averaged; they are not pooled quantiles/global extrema. Guard rates describe these diagnostic forwards, not every bf16 training step. Geometry alone does not establish contextual reasoning or better prediction.

## Actual commands and recovery

Executed with SSH alias `insomnia` in `~/algollms/xsa`. These keys already identify completed jobs; inspect them rather than resubmitting:

```bash
/usr/bin/python3 -m xsa.bc_submission verify --key direction-bc-verify-v1 --seconds 300
/usr/bin/python3 -m xsa.bc_submission train --tag b010 --key direction-bc-b010-v1 --seconds 1740
/usr/bin/python3 -m xsa.bc_submission verify --key direction-bc-verify-v2 --seconds 300
/usr/bin/python3 -m xsa.bc_submission train --tag b050 --key direction-bc-b050-v1 --seconds 1740
/usr/bin/python3 -m xsa.bc_submission train --tag b100 --key direction-bc-b100-v1 --seconds 1740
/usr/bin/python3 -m xsa.bc_submission train --tag c --key direction-bc-c-v1 --seconds 1200
/usr/bin/python3 -m xsa.bc_submission collect --key direction-bc-collect-v1 --seconds 300
```

The training wrapper runs training, final evaluation and bounded diagnostics within each GPU allocation. The CPU job runs `.venv/bin/python -m xsa.direction_bc collect`, followed by `.venv/bin/python evidence/direction_bc/audit_remote.py`. Final accounting uses `python3 scripts/audit_bc_accounting.py`. Actual sbatch arguments and pre-submission scheduler snapshots are in the ledger.

An ancillary readiness-print command expected an `update` key absent from the earlier B 0.1 readiness schema and raised KeyError. The guarded launcher independently checked all final evaluations and frozen proof before collection; direct uniform checks of evaluation records confirmed complete budgets. This observation-format error did not affect training or collection. Failed numerical tests and their fixes are preserved above; no scheduled job failed or needed a continuation.

For a future interrupted, incomplete run, first inspect its existing ID with `squeue`/`sacct` and its latest/previous checkpoint, then use a new reservation key with `--resume` within an authorized budget. The launcher rejects completed runs. An SSH observation timeout is not evidence of job failure. No continuation or further training is scheduled.

## Limits and measured next-stage proposal

This is one training seed, a 29.94M model, one assumed optimizer recipe, a bounded FineWeb prefix and only 32,768 validation targets. Configuration selection uses that validation set. There is no untouched-test result or training-seed uncertainty estimate. No downstream task, mechanistic control, larger model, fresh seed or context sweep was run. KV-cached decoding remains unsupported. This is not a paper-scale replication or a general advantage claim.

C is the most plausible extension to investigate further because it is close to XSA here and keeps the efficient aggregate path; the present screen supplies no established improvement. A bounded confirmation proposal is **three fresh paired seeds** (for example 29, 43 and 59) for **SA, XSA and frozen C**, with the same 2048-update protocol. That is **nine new runs / 603,979,776 processed tokens**. Keep seed 17 as selection evidence, and prepare a separately specified untouched evaluation split before using any final-test claim. Additional tuning or controls would be a separate study.


Measured complete allocations project to `3 × (624 + 663 + 726) = 6039 seconds`, or **1.68 A6000 GPU-hours**, excluding queue time. C's 726-second run includes unusually noticeable startup; the estimate deliberately uses total allocations. A conservative proposal is **nine 30-minute training caps (4.5 GPU-hours reserved)** plus a separate **5-minute verification reservation**, totaling **4h35m**, and a **5-minute CPU collection cap**. This is an extrapolation at the same model/context/token budget, not a scaling estimate for a larger model or guarantee across nodes.

Using the measured largest checkpoint (359,483,267 bytes), three new initialization files (119,770,862 bytes each), nine retained latest/previous pairs, one temporary checkpoint and 128 MiB metadata allowance, projected project peak is **17.10 GiB**. It exceeds the current 12 GiB cap; a proposed **18 GiB project cap**, with current home quota rechecked, would cover that estimate. New evaluation-data preparation/storage and any adapter development are not costed and need a bounded protocol. These figures are a concrete follow-up proposal, **not authorization or launched work**.
