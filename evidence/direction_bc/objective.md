Implement and verify research Directions B and C from
XSA_Insomnia_complete_handoff.md, then complete a bounded, reproducible
screening comparison against the existing SA, XSA, and Direction A results.

## Workspace and starting point

- Use SSH alias `insomnia`.
- Work remotely under `~/algollms/xsa`, within my `~/algollms` folder.
- Publish code, configurations, compact results, and reports to:
  https://github.com/ptshen/xsa-research.git
- Read the handoff, AGENTS.md instructions, PROGRESS.md,
  REPRODUCTION_SPEC.md, DIRECTION_A_PROTOCOL.md, DIRECTION_A_REPORT.md,
  and existing manifests and job ledgers before making changes.
- Inspect local, remote, and GitHub state. Preserve uncommitted work and
  reconcile newer remote records before synchronizing.
- Verify existing results and check recorded job IDs with the scheduler
  before submitting anything. Do not repeat completed experiments.

Use concise, reviewable code and red/green TDD. Explain the implementation
and its numerical conventions in the final report.

## Objective and scope

Implement both methods faithfully, establish numerical correctness, and
run four matched, single-seed screening configurations:

1. Direction B: lambda = 0.1
2. Direction B: lambda = 0.5
3. Direction B: lambda = 1.0
4. Direction C: output-geometry projection

Reuse the existing SA, XSA, and three Direction A runs after verifying
their compatibility. Produce a combined nine-configuration comparison.

A negative or inconclusive research result is a valid completed outcome.
Do not tune beyond the declared settings to obtain a favorable result.

## Direction B: penalize positive value similarity

For each head and causal query/key pair:

s'_ij = (q_i^T k_j) / sqrt(d_head)
        - lambda * max(0, cosine(v_i, v_j))

z_i = sum_{j <= i} softmax_j(s'_ij) * v_j

Requirements:

- Apply the penalty before softmax, including the diagonal.
- Aggregate the original values.
- Preserve the causal mask and the existing Q/K scaling.
- Allow gradients through the value-similarity calculation.
- Add no trainable parameters.
- Validate that lambda is finite and nonnegative.
- Lambda = 0 must recover standard attention in outputs and gradients.
- Preserve native first-position behavior.

Build a small explicit reference and a practical training implementation.
Use query chunking or another supported implementation if needed.
Verify its outputs and gradients against the reference, including causal
masking at chunk boundaries.

Do not claim reduced training memory from chunking without measuring
forward/backward peak memory; autograd can retain intermediate tensors.
Document attention backend differences and their timing implications.

## Direction C: project using the output geometry

For each head, extract its actual output-projection block W_O,h and compute:

G_h = W_O,h^T W_O,h

z_i = y_i - [(v_i^T G_h y_i) / (v_i^T G_h v_i)] * v_i

Requirements:

- Apply the correction before head concatenation and output projection.
- Verify the matrix orientation and head-block slicing against the code.
- Recompute the geometry from current weights on every forward pass.
- Preserve gradients through the geometry and output-projection weights.
- Add no trainable parameters or matrix inverse.
- Retain the efficient ordinary-attention path where possible.
- Define and test a stable zero/near-zero denominator policy.
- Do not silently introduce regularization or detach the metric.

The intended guarantee is:
(W_O,h z_i)^T (W_O,h v_i) = 0
for nondegenerate inputs, within numerical tolerance.

This guarantee is per head, not for the summed multihead update or the
entire residual stream. Document where numerical guards weaken it.

## Verification gate

Write meaningful failing tests before implementation and preserve
red/green evidence. Cover:

- B at lambda = 0 versus SA, including the explicit penalty computation.
- C with an identity metric versus XSA on nondegenerate inputs.
- C orthogonality after the per-head output projection.
- Reference versus training-path outputs and gradients.
- Gradients through B's values and C's output-projection weights.
- Float64 gradient checks away from nondifferentiable boundaries.
- Causality, head layout, and attention width differing from model width.
- Zero, near-zero, collinear, and rank-deficient cases.
- Native first-position behavior.
- Finite bf16 forward/backward execution on the allocated GPU.
- Unchanged SA/XSA/A behavior, parameter names, shapes, and counts.
- Shared initialization, deterministic data order, and token accounting.
- Checkpoint/resume equivalence for both new methods.
- Resume rejection when method-specific settings change.

Use fp32 numerical reductions for low-precision inputs while retaining
float64 for reference tests. State tolerances and epsilon conventions.

Do not add KV caching solely for this stage; if unsupported, report it.
Distinguish passing tests from skipped tests.

Before screening, pass the GPU correctness gate and re-evaluate preserved
baseline checkpoints on the established windows within documented
tolerances. Investigate mismatches before proceeding.

## Matched experimental protocol

Use the verified existing pilot protocol:

- Approximately 29.94M parameters, six layers, width 384.
- Six heads of dimension 64; context length 512.
- Existing `data/pilot-v2` and pinned tokenizer/data manifests.
- Existing shared initialization `runs/initial/pilot-s17.pt`.
- Model seed 17, data seed 1017, evaluation seed 2017.
- 2,048 optimizer updates.
- 32,768 training tokens per update.
- Exactly 67,108,864 training tokens per run.
- The existing optimizer, LR schedule, precision, global batch,
  evaluation schedule, and checkpoint schedule.
- The same 64 validation windows totaling 32,768 target tokens.

Verify these settings from the artifacts rather than trusting this
description alone. Record any discrepancy.

Only the intended attention method and its declared settings may change.
Do not silently reduce a method's token budget, change its optimization
recipe, or substitute data to make it fit.

Verify initialization, data/evaluation hashes, all ordered batch hashes,
resolved configurations, software versions, and actual tokens trained.

## Execution authorization and limits

This prompt authorizes implementation, tests, profiling, the four runs,
evaluation, collection, and GitHub publication within these limits:

- Existing university resources only; no paid services.
- One GPU and one active project GPU job at a time.
- Use the existing A6000 allocation if currently available and permitted;
  inspect current account/partition limits before selecting resources.
- New B/C stage: at most 4 GPU-hours total, including GPU tests,
  profiling, training, evaluation, diagnostics, and retries.
- Within that total: at most 10 minutes of GPU verification/smoke/profile.
- CPU-only Slurm jobs: at most 30 allocated minutes total.
- Total project storage: at most 12 GiB, including retained artifacts
  and temporary checkpoint-write headroom, subject to actual quota.

These are new stage limits. Preserve the historical Part 1 and Direction A
accounting; do not reset or borrow from their budgets.

Reserve requested walltime before submission and record actual usage
afterward. Include retries and continuations. Make reservations
conservative enough that outstanding jobs cannot exceed the limits.

Profile the actual training shapes first, with warmup, synchronization,
forward/backward/optimizer timing, and peak memory. Include estimates
for evaluation, diagnostics, and checkpoint I/O.

Freeze a written protocol and resource estimate before screening.
Proceed without another confirmation if verification passes and the
complete planned screen fits these limits.

If the screen cannot fit, finish independent implementation and reporting,
then provide the measured shortfall and a concrete revised proposal.
Do not launch knowingly insufficient runs or exceed the limits.

Run heavy computation on allocated compute nodes. Check job state and
artifacts before every submission or resume. An SSH timeout is not
evidence that a job failed. Do not duplicate uncertain submissions.

## Evaluation and interpretation

Primary outcome:
held-out token-weighted next-token cross-entropy at the fixed token budget.

Report perplexity as exp(cross-entropy), explicitly as a transformation
of the same metric, not an independent benchmark or task accuracy.

For all nine configurations, report:

- Cross-entropy and supplementary perplexity.
- Paired loss differences versus SA and XSA.
- Learning curves against training tokens and elapsed time.
- Training throughput, allocated runtime, and peak GPU memory.
- Actual attention backend and relevant implementation overhead.
- Complete method settings and provenance.

Use the established bounded diagnostic samples. Add:

- B: applied penalty, attention-weighted positive value similarity,
  diagonal attention mass, and output/self-value alignment.
- C: alignment before and after the head output projection, correction
  magnitude, denominator distribution, and guard activation rate.
- Separate zero and near-zero output counts from cosine statistics.

Document averaging and masking conventions. Geometric behavior alone
does not establish improved language modeling or contextual reasoning.

Report every configuration, including failures and negative results.
Validation-window uncertainty is not training-seed uncertainty.
Selection on this validation set is not an untouched-test result.

Do not launch downstream benchmarks, controls, additional seeds, larger
models, new context sweeps, or the full proposal matrix in this stage.
Document those omissions and estimate an appropriate next confirmation
study from measured costs.

## Persistence and deliverables

Maintain PROGRESS.md throughout with completed work, commands, evidence,
failures, job IDs, resource accounting, and the exact next action.

Preserve existing logs, data, initialization, and checkpoints. Retain
atomic latest/previous checkpoint pairs for new runs. Do not delete
historical artifacts to satisfy the storage cap.

Deliver:

- Tested B/C implementations and reproducible configurations.
- Slurm verification, training/resume, and collection commands.
- A frozen B/C protocol and machine-readable experiment manifest.
- Red/green test evidence and actual GPU verification results.
- Complete screening artifacts and a combined nine-method summary.
- A report explaining the code, numerical choices, results, limitations,
  actual commands/job IDs, compute/storage usage, and measured estimates
  for the next stage.
- An independent collection audit checking checkpoints, identities,
  full training budgets, and recomputed metrics.
- Code and compact evidence pushed to the owned GitHub repository,
  with the published commit verified.

Keep large datasets and checkpoints on permitted university storage;
publish paths and checksums rather than large binaries or credentials.

If SSH or GPU access is blocked, finish independent local work, preserve
uncertain job states, and report exactly what remains blocked and how to
resume. Do not claim GPU execution or a completed screen without evidence.