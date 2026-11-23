# XSA / Insomnia progress

## Goal and limits

Implement Part 1 of `XSA_Insomnia_complete_handoff.md`; do not implement proposal extensions or launch the full training matrix. Total smoke/profile training budget: **600 seconds**, including local smoke work. Correctness tests and environment/data setup are separate. Preserve logs and checkpoints. Remote working root: **`~/algollms/xsa`**, per user instruction.

## 2026-11-23 — inspection

- Local directory initially contained only the handoff; it was not a Git repository. No code or prior project job IDs existed.
- Read the Part 1 requirements and arXiv:2603.09078v1 (10 March 2026), including tables and captions. Pinned nanoGPT HEAD `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`; author's training commit is not specified in the paper.
- SSH alias `insomnia` works; login `pts2125`, account `edu`, group `somedu`; `short` permits 12 hours. No active jobs and no jobs today at initial inspection. Other homework jobs are unrelated and preserved.
- Scheduler and storage evidence: `evidence/cluster_initial.txt`, `evidence/cluster_storage.txt`. GPU inventory includes A6000, L40, L40S, H100; inventory alone does not prove allocatability.
- Home quota: 52,428,800 KiB (50 GiB), 28,657,104 KiB used. Full FineWeb corpus is not authorized and does not fit this quota. Project smoke artifacts will remain under `~/algollms/xsa`.
- Existing local Python 3.13.5 / torch 2.9.1 available. Existing ancestor AGENTS guidance concerns Prime/Verifiers environments and does not apply to this standalone PyTorch reproduction. User's concise-code, red/green TDD and explanation requirements apply.

## Execution ledger

| Work | Job ID | State | Smoke/profile seconds |
|---|---|---|---:|
| Inspection only | none | complete | 0 |

No training/profile jobs submitted yet. Check `squeue`, `sacct`, and this ledger before each submission; never infer job failure from a polling timeout.

## Next action

Write failing numerical and pipeline tests; implement the matched models and deterministic training/data/checkpoint contracts. Inspect existing remote Python environments, then prepare one bounded GPU verification job with a hard remaining-budget limit.

## Implementation and setup — 15:59 EST

- Added shared nanoGPT-derived SA/XSA model, independent head widths, adjacent-pair RoPE, embedding LayerNorm, tied token/head weights, fp32 projection of bf16 aggregates (float64 retained for reference tests).
- Model TDD: expected initial import failure saved in `evidence/tdd_model_red.txt`; green result `7 passed, 2 skipped` in `evidence/tdd_model_green.txt`. Skips are allocated-CUDA-only tests.
- Pipeline TDD red evidence: `evidence/tdd_pipeline_red.txt`. Implementing deterministic GPT-2 document packing, separate data/eval RNG, token/LR accounting, atomic latest/previous checkpoints and strict resume validation.
- FineWeb pinned to `HuggingFaceFW/fineweb` / `sample-100BT` at `9bb295ddab0e05d785b879661af7260fed5140fc`; no corpus downloaded.
- User additionally authorized pushing code and results to `https://github.com/ptshen/xsa-research.git`. Insomnia authentication worked; cloned the **empty** repository under `~/algollms/xsa`. No existing work overwritten.
- Environment setup job **14072260** submitted only after checking `squeue` and today's `sacct`. CPU-only job; no smoke/profile training. Reuses existing torch 2.10.0+cu128 packages via an isolated project venv; does not modify the existing environment. Modules are forbidden on login nodes, so setup runs on a compute node.
- Smoke/profile budget used: **0 seconds** (numerical and deterministic unit tests are correctness checks).

Next: finish/check pipeline tests, evaluation, diagnostics, profiling and Slurm budget guard; check setup job 14072260 before any new submission.

## Bounded GPU submission — 16:12 EST

- Setup **14072260 COMPLETED**, exit 0, elapsed 46 seconds. Environment: Python 3.11.5, torch 2.10.0+cu128, NumPy 2.4.6, tiktoken 0.9.0, matplotlib 3.10.5, pytest 9.2.0. Full lock is preserved remotely.
- Latest local correctness: **17 passed, 2 skipped** (CUDA-only skips), `evidence/local_tests.txt`; deterministic CPU resume is bitwise identical for both attention modes. LR boundary failure was assertion roundoff (0.0005999999999999998 versus 0.0006); changed assertion to numerical tolerance and preserved failure evidence.
- Submitted GPU job **14072282**, key `smoke-v1`, using `/usr/bin/python3 -m xsa.submission smoke --key smoke-v1 --seconds 360 --gpu l40s --cpus 8 --mem 32G` after checking queue and setup accounting. Account `edu`, partition `short`, one L40S, 8 CPUs, 32 GiB, **00:06:00 hard allocation cap**, no requeue.
- Smoke/profile **reservation 360/600 seconds**; conservative remaining reservation **240 seconds** regardless of early completion. No other smoke/profile run has occurred. Remote `evidence/job_ledger.json` is authoritative for submissions.
- Job includes numerical tests, tiny explicitly labeled fixture training, interrupted versus uninterrupted XSA, fixed-window evaluations, diagnostic collection, plot creation, and full pilot-model profiling. It does not launch the pilot training budget.

Next: poll **14072282** and inspect its actual artifacts; finish specification, retention/storage estimates, documentation, and GitHub delivery. Do not resubmit based on a polling timeout.

## GPU verification complete — 16:16 EST

- While **14072282 was confirmed PENDING**, read node allocation data. L40S estimate was 16:53; A6000 node ins093 had a free GPU and sufficient CPU/RAM. Updated the **same job ID** with `scontrol update JobId=14072282 TresPerNode=gres/gpu:A6000:1`; did not cancel or duplicate it. Job actually ran on **NVIDIA RTX A6000**. The initial ledger command records the original L40S request; this note and accounting record the update.
- **14072282 COMPLETED**, exit 0, Slurm elapsed **33 seconds**; orchestrator walltime **23.029 seconds**. Whole-job conservative reservation remains 360 seconds; 240 seconds remain reserved for any necessary follow-up. No full pilot/matrix launched.
- Cluster correctness **19 passed, 0 skipped**. GPU interrupted XSA versus uninterrupted XSA: maximum parameter difference **0.0**, tested atol 1e-6 / rtol 1e-5. SA/XSA initialization and all eight update batch hashes identical.
- Fixture eval at 1024 tokens: SA CE **10.6815280914**, XSA **10.6818139553**, XSA−SA **+0.0002858639**. These are execution-fixture results, not language-model findings; the worse XSA value is preserved.
- Actual pilot (29,938,944 total / 10,622,208 non-embedding params), bf16, batch8, context512, A6000: SA forward/backward **31.004 ms**, XSA **34.476 ms**. Three full accumulation-8 updates average ~**0.24319 s / 0.27389 s**. Compute-only 2048-update estimates: **0.13835 / 0.15581 GPU-hours**, excluding data/evaluation/checkpoint overhead. Peak allocated **3,877,071,872 / 4,521,738,752 bytes**.
- Native bf16 projection max absolute cosine **0.0025916** versus fp32 projection then cast **0.0008244**, on recorded random tensors; this quantifies the deliberate precision choice.
- Collected compact outputs under `evidence/results/smoke-14072282/`, logs and environment under `evidence/cluster/`. Remote checkpoints/logs remain intact.
- Added two hardening tests after the successful run: corrupt-current-checkpoint fallback and discarding abandoned future eval rows after resume. Both first failed as intended (`evidence/tdd_hardening_red.txt`); fixes implemented locally, not yet revalidated on GPU.

Next: finish robustness checks and downstream adapter integration, verify plots, document estimates/limits and exact next-stage commands, then push code plus compact evidence to the authorized GitHub repository.

## Follow-up robustness and downstream integration

- Added an explicit shared initialization artifact (`xsa.initialize`) so both pilot arms can load the same saved tensors, with architecture/seed validation. This complements the already verified deterministic initialization hashes.
- Added an actual SIGUSR1 subprocess test for the Slurm warning handler, corrupt-checkpoint recovery, source-kind refusal and divisibility checks. CPU validation is in progress.
- CPU-only pinned harness setup/integration job **14072287** submitted after verifying no active jobs and terminal states for 14072260/14072282. This job downloads software only, then checks likelihood/rolling-likelihood and task-ID discovery on a saved fixture checkpoint; it does not download task datasets or train.
- Preserved the first GPU run's source snapshot alongside its artifacts before syncing later code changes.
- Visually inspected `evidence/results/smoke-14072282/curves.png`: axes, labels, fixture warning, and both series render correctly. The wall-time curve includes startup effects and is not the synchronized performance benchmark.

- Audit found cross-seed summaries could accept different token budgets if each individual pair matched. A targeted test failed (`evidence/tdd_seed_summary_red.txt`); summaries now also require a common study configuration, dataset, held-out windows and budget across seeds.
- Initialization TDD evidence correction: the first attempted red check overlapped an implementation write and completed green. `evidence/tdd_initialization_red.txt` therefore contains a pass and is **not** claimed as red evidence. Other module red/green logs and targeted hardening failures are genuine; subsequent feature checks run sequentially.

## Final GPU validation submitted — 16:20 EST

- Harness setup **14072287 COMPLETED**, exit 0, 2m51s (CPU-only software setup/interface checks). Pinned harness **0.4.9.2** likelihood, rolling likelihood and all eight task IDs verified; no benchmark task datasets run. Evidence: `evidence/cluster/harness_integration.json`.
- Further corruption test exposed rotation of an unreadable latest over the valid previous file after fallback. Test failure is preserved in `evidence/tdd_valid_previous_red.txt`; recovery now quarantines the damaged file before writing so the valid previous survives.
- Latest CPU suite: **23 passed, 3 skipped** in 5.29s. Skips require CUDA and include new bf16 checkpoint/resume coverage.
- Submitted **14072292**, key `smoke-final-v2`, only after all earlier IDs were confirmed terminal. One A6000, `edu/short`, 8 CPUs, 32 GiB, **00:02:00** hard cap, no requeue. Revalidates the hardened code, explicit shared initialization, signal handling, bf16 GPU resume, analysis and pilot profile.
- Conservative reservations: **480/600 seconds**, **120 seconds unreserved**. Actual completed smoke allocation so far: **33 seconds**. Final job is pending/running; check its exact state before further work on the remote source.

Next: inspect final job artifacts, render plot, finalize audit/report, push authorized repository and verify remote commit. No source sync during this job.

## Verification and completion audit — 16:27 EST

- **14072292 COMPLETED**, exit 0, 16 seconds, NVIDIA RTX A6000 on ins093. **26 passed, 0 failed, 0 skipped** in 6.74s. CUDA resume maximum parameter difference remains **0.0**; the separate bf16 CUDA resume and real SIGUSR1 subprocess tests pass.
- Total actual GPU smoke/profile allocation time **49 seconds**; 480/600 seconds conservatively reserved. No longer experiments submitted. All four project jobs terminal; final queue/accounting saved.
- Repeated pilot profile: SA **31.031 ms** and XSA **34.581 ms** per forward/backward microstep; full updates **244.296 / 274.970 ms**. Compute-only pair estimate **0.2954 GPU-hours**. Both independent short measurements retained.
- Independently regenerated fixture locally: identical manifest `2c1bc0fa9cb6b0066992e267b94685758935f978ab48d680426e562b5e99c08f` and verified all shard/document-index checksums. Collection CLI and plot re-render pass. Both final plots visually checked.
- Packaging red check found automatic flat-layout discovery including non-code directories; restricted wheel to `xsa*`. Wheel build now passes. Shell scripts pass `bash -n`; Python modules compile. Packaging-only changes do not change GPU-tested training code.
- Repeated evidence round trips created redundant nested collection directories. Excluded these from future collection/sync and Git; original logs/checkpoints remain preserved. Canonical compact artifacts live in `evidence/results/` and `evidence/cluster/`.
- Added `REPORT.md`, complete `REPRODUCTION_SPEC.md`, command-oriented `README.md`, `RUN_MANIFEST.json`, environment locks and storage/compute arithmetic. Full-run timing values are explicitly coarse parameter-token proxies, not billion-parameter measurements.
- Completion checks: architecture/numerical tests, data/config reproducibility, actual allocation/scripts, atomic and signal resume, evaluation/diagnostics/downstream adapter, bounded profiling, checkpoint/plot loading, measured next-stage estimate and report are evidenced. Real FineWeb download, long pilot, paper-scale study and downstream datasets remain intentionally unrun under the handoff's budget/authorization boundary.

Next: push the code and compact evidence to the user's GitHub repository, verify branch/tree, then report completion. The next research-stage decision is authorization for one A6000 GPU-hour total plus bounded CPU preparation and 5 GiB for the exploratory pair; no such job is submitted.

## GitHub delivery

- Committed implementation and compact evidence as **72cb3c4802f39d54e2d971d38f2a1b68c3950ccb** and pushed `main` to **https://github.com/ptshen/xsa-research**. `git ls-remote origin refs/heads/main` matched that commit; remote worktree was clean.
- HTTPS push initially failed because the noninteractive server session had no HTTPS username credential. The earlier public clone did not prove authentication (correcting the initial wording). Switched the repository remote to `git@github.com:ptshen/xsa-research.git`; the user's existing SSH setup authenticated and pushed successfully. No new credentials were requested or created.
- Broad `git diff --check` flagged original handoff/reference/log whitespace. Those source/evidence files were preserved verbatim; the authored code/config/docs check passed. Checkpoints, token shards, environments and caches are excluded from Git. Canonical Slurm logs are included under `evidence/cluster/logs/`.
- All implementation and verification work is complete. Final documentation/log delivery is being committed on top of the source commit; the repository's `main` is the delivery branch.

**Next action:** review REPORT and authorize, if desired, one A6000 GPU-hour total, bounded CPU data preparation and 5 GiB for the seed-17 pilot pair. The handoff requires separate authorization for that longer stage. No active project jobs or long training launches remain.

## Final state — verified delivery

- Delivery commit **b86a31be5f37e544d81531e12e8314345f90669b** is present on GitHub `main`; remote `git ls-remote` matched it. Both local and Insomnia worktrees were synchronized and clean.
- Final artifact audit passed against the actual files: all recorded model/config source SHA256 values match; final run, pairing and resume checks pass; 26 cluster tests pass; independent data checksums and pinned harness integration pass; required reports/manifests exist; 49 actual smoke/profile seconds remain below 600; no active user jobs remain.
- The requested Part 1 pipeline and GitHub delivery are **complete**. Only the separately authorized next training stage remains a future action. Checkpoints and logs are preserved; no cleanup or long run was performed.

## Approved pilot stage — 2026-11-23

- User approved the proposed **one A6000 GPU-hour total**, bounded CPU preparation, and **5 GiB project storage**. This is a separate pilot authorization; the prior 49-second smoke/profile accounting remains unchanged.
- Rechecked clean local/remote repositories, no active jobs, terminal earlier IDs, and available home quota. Work stays under `~/algollms/xsa`.
- First pinned FineWeb-100BT parquet shard is 2,147,509,487 bytes. Preparation is capped at **150,000 documents, 100,000,000 packed tokens, and one source shard**; complete documents only. HTTP download disables Xet cache duplication. Token-cap red/green evidence is in `evidence/pilot_token_cap_{red,green}.txt`.
- The two training reservations will be 30 minutes each, sequential A6000 jobs, with fixed 67,108,864 tokens per arm, shared seed-17 initialization, post-run held-out evaluation and bounded diagnostics included in those allocations. No full downstream tasks are authorized here.

Next: submit bounded CPU data preparation, validate exact split sizes and storage, then submit SA only after preparation completes.
- CPU preparation job **14072309** submitted via the duplicate-checking launcher; 4 CPUs, 16 GiB, 30-minute cap, no GPU. Exact bound environment: `XSA_MAX_TOKENS=100000000 XSA_MAX_SHARDS=1 XSA_PREPARE_INITIAL=1`. GPU reservation consumed by this stage so far: **0 / 3600 seconds**.
- Initial preparation **14072309** deliberately cancelled after **3m08s**: measured HTTP download ~1 MB/s would exceed its 30-minute preparation bound. Logs and partial data directory are retained. This was an observed throughput/budget decision, not a polling-timeout restart.
- Switched to the installed parallel Xet downloader with `HF_XET_CHUNK_CACHE_SIZE_BYTES=0` and `HF_XET_HIGH_PERFORMANCE=1`; same immutable first shard and document/token caps. New data destination is `data/pilot-v2` to preserve v1. The guard correctly refused submission while 14072309 was still COMPLETING; waited for it to leave the queue before submitting the retry, capped at 25 minutes.
- Retry preparation job **14072321** is running; GPU budget still unused. It must produce at least 64 disjoint 513-token validation windows before either training arm is launched.
- Preparation **14072321 COMPLETED** in 2m43s, exit 0. Retained **146,329 training documents / 99,962,616 tokens**, **74 validation documents / 36,418 tokens**. There are **70 nonoverlapping validation windows**, sufficient for the fixed 64-window protocol. Shared initialization is 119,770,862 bytes.
- Removed only the abandoned **157,286,400-byte HTTP `.incomplete` cache file** after confirming the complete pinned 2,147,509,487-byte source shard existed; recorded the cleanup in `evidence/pilot_partial_cleanup.json`. This frees checkpoint-write headroom. Original logs, data directories and all checkpoints remain preserved.
- Submitted SA job **14072327**, one A6000 / edu / short / 8 CPUs / 32 GiB, 30-minute hard cap, no requeue. Data `data/pilot-v2`, initial state `runs/initial/pilot-s17.pt`, output `runs/pilot-sa-s17`; `XSA_POST_EVALUATE=1` includes evaluation/diagnostics in the same GPU allocation.
- Pilot GPU reservation: **1800 / 3600 seconds**. XSA has not been submitted. Next: monitor SA, verify final token count/checkpoint/evaluation and storage, then submit XSA sequentially.
- SA **14072327** is PENDING for resources. Slurm estimates shifted from 19:49 to **17:11 EST**; estimates are not guarantees. To avoid a manual handoff while queued, preparing XSA with `afterok:14072327`, a strict full-token-budget gate, and a dependent CPU collector that verifies both checkpoints/batch histories and automatically publishes compact results to GitHub. No simultaneous GPU experiments and no additional training beyond the two 30-minute allocations.
- Preparation storage after cache cleanup: **3,315,784,352 bytes** (~3.09 GiB). Worst-case addition for both latest/previous checkpoints plus one atomic temporary file is ~1.80 GB, keeping projected peak below 5 GiB. Collector enforces the final storage and 3600-second GPU bounds.
- Automatic sequence submitted after rechecking the existing IDs: **SA 14072327 → XSA 14072338 → CPU collector 14072339**. XSA and collector use Slurm `afterok` dependencies, `--kill-on-invalid-dep=yes`, no requeue, and strict full-budget/pair-identity gates. Exactly **3600 seconds** total maximum GPU allocation across the two arms; CPU collection capped at 3 minutes. No speculative retry or extra seed.
- CPU verification job **14072336** checks the new gate/token-cap tests and GitHub SSH read access from an actual compute node. Source/download caps remain explicit and tested (6 local gate/analysis checks passed).
- `PILOT_STATUS.md` records the queued workflow and exact inspection commands. Completion will automatically write `PILOT_REPORT.md`, `PILOT_MANIFEST.json`, compact artifacts and append PROGRESS, then push them to `main`. It fails closed if budgets, completed updates, paired identities or scheduler outcomes do not match.

Next action: the submitted Slurm dependency chain performs the authorized pair and result collection. Training is currently queued; no result is claimed yet. Inspect jobs 14072327, 14072338 and 14072339 rather than submitting duplicates.

## Pilot running — SA complete

SA job **14072327 COMPLETED**, exit 0, allocated **624 seconds** on ins092. Reached 2048 updates / 67,108,864 tokens; final held-out cross-entropy **4.89046994**, perplexity **133.01607**. Evaluation and diagnostics are saved. XSA **14072338** passed the full-budget gate and is running on the same GPU type. CPU collection **14072339** remains dependent on successful XSA completion. No additional submissions. Next action: verify XSA completion and automatic paired checks/publication.

## Pilot finished automatically

Both training jobs completed the full budget; identical initialization/batch order verified. SA CE 4.89046994, XSA CE 4.80333272, XSA−SA -0.08713721. GPU allocation 1287/3600 seconds; disk 4.430/5 GiB. See PILOT_REPORT.md. Next action: review the one-seed findings; no further training authorized or scheduled.

## Pilot final audit and delivery

- **SA 14072327 COMPLETED 624 s**, **XSA 14072338 COMPLETED 663 s**, **collector 14072339 COMPLETED 19 s**, all exit 0:0. Both arms reached 2048 updates / 67,108,864 tokens. Total actual pilot GPU allocation **1287 s (21m27s, 0.3575 GPU-hours)**; project storage at collection **4.430 GiB**. The original smoke/profile total remains 49 s.
- Collector verified identical initialization, all 2048 ordered minibatch hashes, final data RNG state, configs, fixed held-out windows and GPU type. Published compact results in commit **2aa3d9d8e3c5dcf621c75fce2ea963489590f55c**; remote GitHub main matched it. No active user jobs remained at final inspection.
- Independently audited downloaded compact artifacts: consecutive updates and finite losses/gradients, every paired batch hash, data/checkpoint checksum identity and exact paired-summary recomputation all pass. Plot visually checked. SA CE **4.89046994**, XSA CE **4.80333272**, difference **-0.08713721**; one-seed exploratory result only.
- The new preparation-cap and pilot-gate tests passed on a compute node (**2 passed, 11.63 s**) and locally (**2 passed, 1.08 s**). A local recheck first used a nonexistent shortened test selector and collected no tests; rerunning the actual test name passed. This was a command typo, not a code failure.
- Updated README to the successful `data/pilot-v2` workflow, documented the actual dependent Slurm commands, and labeled REPORT as the historical pre-pilot snapshot. Recorded pilot package versions including hf-xet 1.8.0. Final collector log and all scheduler outcomes are retained.
- Measured planning estimate for two further paired seeds: **0.715 GPU-hours actual-time projection**, or **2 GPU-hours reserved** with the same four 30-minute caps. Retaining artifacts would require roughly **7.7 GiB peak project storage**, above the existing 5 GiB cap. No further seed, downstream evaluation or full matrix has been submitted.

Next action: review PILOT_REPORT.md and the one-seed result before deciding whether to authorize seed confirmation and a revised storage cap. All current authorized work is complete; preserve the existing data, logs and latest/previous checkpoints under `~/algollms/xsa`.
