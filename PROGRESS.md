# XSA / Insomnia progress

**Current stage:** Directions B/C complete, including all four full-budget runs, independent audits and verified GitHub delivery. The objective is preserved in `evidence/direction_bc/objective.md`. Latest state and next action are at the end of this log. Earlier stage limits below are historical.

## Original Part 1 goal and limits

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

## Direction A — authorized implementation and screen

- Read the goal attachment in full; implementing Direction A only, thresholds 0.25/0.50/0.75, seed 17 and existing pilot protocol. Authorized stage bounds: 7200 GPU seconds total, 300 GPU verification/profile seconds within it, 8 GiB project storage, 900 allocated CPU seconds. Prior smoke/profile accounting remains unchanged.
- Rechecked local and remote clean source at 3e81815, no active jobs, all earlier IDs terminal. `edu/short` permits A6000. Home usage 33,307,936 KiB of 52,428,800 KiB; project 4,757,404,924 bytes. No AGENTS.md file found; the supplied AGENTS instructions apply. Existing baseline checkpoints and reports remain intact.
- Wrote numerical/integration tests first; import failed before implementation (`evidence/direction_a/model_red.txt`). Implemented the formula with unchanged SA/XSA arithmetic, fp32 reductions under bf16, explicit [0,1) threshold validation and baseline epsilon convention. Initial local suite: 30 passed / 6 CUDA skips (`model_green.txt`). Tau=0 forward/gradient equivalence, bounds, degeneracies, causality, shared init, checkpoint compatibility and CPU resume pass.
- Snapshot of baseline model source preserved from commit 3e81815 for exact regression comparisons.

Next action: finish threshold metadata/diagnostics/strict collection and budgeted Slurm scripts, then run the allocated CUDA correctness/profile gate before any full screen training.

- Metadata/diagnostics tests first failed for missing threshold fields and activation rate; implemented both. Full local suite now **43 passed / 6 CUDA skips**, including bitwise SA/XSA output/gradient regression against the frozen baseline source. Strict screen matching and stage reservation/duplicate guards also have red/green tests. No training submitted. Next: 180-second allocated CUDA correctness, baseline checkpoint re-evaluation and five-method pilot-shape profile.

- GPU verification **14072510 COMPLETED**, exit 0, 38 seconds on RTX A6000 ins093. **49 passed / 0 failed / 0 skipped**, 10.77 s. Actual preserved SA/XSA checkpoints re-evaluate within 1e-6 per-window tolerance; shared initialization loads into all caps and produces the original hash. Five-method SDPA/bf16 profiles saved. Cap compute-only projections: 590.904 / 590.422 / 589.152 seconds. Project 5,031,103,892 bytes after preserved test artifacts; conservative peak estimate 7,581,040,745 bytes (7.06 GiB), below 8 GiB.
- Collection rollback test failed before implementation, then all 3 screen/collection/budget tests passed locally. Numerical code/config hashes are frozen by the CUDA gate. Stage reserved GPU seconds so far: 180/7200; verification reservation 180/300. Actual GPU time: 38 seconds.

Next: commit the verified implementation, submit only tau=0.25 with a 30-minute cap, and inspect its terminal state/full-budget artifacts before the next threshold.

- Implemented source and verification evidence pushed to GitHub main as **57aa6e2**. Submitted **tau 0.25 job 14072518**, edu/short A6000, 8 CPUs/32 GiB, 1800-second cap. It is running on ins093; manifest initialization/data/evaluation hashes match the preserved pilot. Stage reservation is now **1980/7200 GPU seconds**, including **180/300 verification seconds**. Tau 0.50/0.75 remain unsubmitted.

- Connectivity interruption after job 14072518 started: ordinary SSH, bounded keepalive SSH and a fresh non-multiplexed connection all timed out on port 22. Last successful job observation: RUNNING on ins093 at update 246 / 8,060,928 tokens, finite loss/gradients. No evidence of job failure and no duplicate/retry submitted. The documentation SCP failed before the subsequent status query; local documentation is retained. Next: restore/read-only connectivity, inspect existing 14072518 and its artifacts, then continue the remaining thresholds only after successful full completion.

- Independent local audit confirms all numerical/config hashes still match the successful CUDA proof and both baseline re-evaluations have **maximum per-window difference 0.0**. General internet and GitHub SSH remain reachable; only Insomnia port 22 times out. Publishing protocol/status documentation from the local checkout while cluster access is unavailable. No additional Slurm submission or model change.

## Connectivity recheck — continuation 2

- Fresh bounded SSH retry again timed out. Existing job **14072518** remains unverified; no additional jobs submitted.
- Reconciled the local/GitHub stage ledger with the already-confirmed 14072518 submission and its 1800-second reservation, using the captured successful `sbatch` response. Total recorded reservation is **1980 GPU seconds**. Current outcome is explicitly unknown. The remote ledger must be read and preserved before synchronizing on reconnection.
- Added an explicitly incomplete DIRECTION_A_REPORT.md containing verified tests, measured five-method profiling, known resource estimates and exact outstanding work. This does not claim the screen is complete.

Next action: restore Insomnia reachability and inspect 14072518 before proceeding. Independent implementation, tests, measured profiling and interim reporting are finished; remaining training and final collection require cluster access.

## Connectivity recheck — continuation 3; blocked audit

- A fresh SSH connection bypassing multiplexing again timed out on port 22 (exit 255). The same access blocker has persisted across three consecutive goal turns. No current scheduler or artifact state can be read.
- Independent implementation, numerical tests, profiling, provenance checks, protocol documentation and interim reporting are complete and pushed. Remaining requirements depend on Insomnia: determine the outcome of 14072518, finish the remaining thresholds, collect all completed artifacts and verify/publish final results. The goal is incomplete and will be marked blocked on restored cluster access.
- No duplicate, replacement or additional job was submitted. Reservations remain 1980/7200 GPU seconds, including 180/300 verification seconds. Existing logs/checkpoints are preserved.

Resume action: restore the required network/VPN or cluster availability, then inspect 14072518 with squeue/sacct and read the remote stage ledger before any synchronization or submission. Reconcile the remote ledger with the locally published known submission; do not overwrite newer remote job records.

## Access restored — 2026-11-24

- Insomnia SSH is reachable. Read the authoritative remote ledger and preserved it as `evidence/direction_a/ledger-before-reconnect-20261124.json` and a named Git stash before fast-forwarding documentation. Reconciled all existing IDs without discarding records.
- **14072518 COMPLETED**, exit 0, **663 seconds**. Tau 0.25 reached 2048 updates / 67,108,864 tokens; validation CE **4.8214927465**, perplexity **124.1502774**. Evaluation/diagnostics and latest/previous checkpoints exist. All 2048 ordered batch hashes and config/data/initialization/evaluation identities match the baseline. No retry was needed.
- Rechecked no active project jobs and quota: 34,805,936 KiB used of 52,428,800 KiB. Project **5,751,180,272 bytes**; remaining checkpoint pairs plus atomic-write allowance keep projected peak near 7.06 GiB, below 8 GiB.
- Submitted tau **0.50 job 14074753**, A6000/edu/short, 8 CPUs/32 GiB, 1800-second cap. Stage reservation **3780/7200 GPU seconds**, including 180/300 verification seconds. Confirmed actual elapsed so far is 38+663=**701 GPU seconds**, plus the new job when allocated. Tau 0.75 remains unsubmitted.

Next: inspect 14074753 until terminal and verify complete artifacts, then submit tau 0.75. All existing results and checkpoints remain preserved.

- **Tau 0.50 / 14074753 COMPLETED**, exit 0, **690 seconds**, ins093 A6000. Reached 2048 updates / 67,108,864 tokens; validation CE **4.8145135567**, perplexity **123.2868257**. All 2048 batch hashes, protocol identities and finite training metrics pass; evaluation, diagnostics and latest/previous checkpoints exist. Project **6,472,059,560 bytes**, leaving sufficient headroom for the final pair of checkpoints and an atomic temporary file (projected peak ~7.06 GiB). Confirmed stage GPU time: **1391 seconds**. Next: submit only tau 0.75, then collect all five methods.

- Submitted final threshold **tau 0.75 / job 14074780**, same A6000/edu/short/8 CPUs/32 GiB/1800-second cap. Stage GPU reservations are **5580/7200 seconds**, including **180/300 verification seconds**. Prior two training jobs are terminal and their artifacts passed lightweight identity checks; full checkpoint loading/checksums for all five methods will run in the CPU collector. Next: monitor 14074780; no further training jobs are planned.

- **Tau 0.75 / 14074780 COMPLETED**, exit 0, **658 seconds**, ins093 A6000. Reached 2048 updates / 67,108,864 tokens; validation CE **4.8711863980**, perplexity **130.4756211**. All batch hashes, protocol identities and finite training metrics pass; evaluation/diagnostics exist.
- All three thresholds completed without training retries. Actual Direction A stage GPU allocation: **2049 seconds (34m09s)**, including 38 verification/profile seconds; conservative reservation 5580/7200 seconds. No further GPU work planned. Next: submit the CPU collector for full checkpoint/identity/budget validation and final report/plots.

## Direction A collected

All three thresholds completed and passed checkpoint, configuration, initialization, batch-order and evaluation identity checks. GPU allocation 2049/7200 s, verification/profile 38/300 s, project 6.701/8 GiB. Full outcomes in DIRECTION_A_REPORT.md. Next: independent artifact/plot audit, final accounting and GitHub publication; no further experiment scheduled.

## Direction A final audit — 2026-11-24

- **CPU collector 14074923 COMPLETED**, exit 0, **36 seconds**, ins069. Loaded all five real final checkpoints, checked finite parameters, complete budgets, matching configs/data/evaluation windows/initialization/final data state and all 2048 batch hashes, and recorded latest/previous checkpoint checksums.
- Full stage accounting is terminal and successful: verification **38 s**, tau 0.25 **663 s**, tau 0.50 **690 s**, tau 0.75 **658 s**. Actual GPU allocation **2049 s (34m09s)**; CPU-only collection **36 s**. Reservations **5580/7200 GPU seconds**, **180/300 verification seconds**, **300/900 CPU-only seconds**. Start/end/TRES checks prove sequential one-A6000 GPU allocations. Final queue inspection found no active user jobs.
- Storage at collection **7,195,200,663 bytes (6.701 GiB)**; temporary checkpoint peak had been conservatively estimated at 7.06 GiB, below 8 GiB. No existing logs, checkpoints, data or cache were deleted. The original Part 1 smoke accounting remains unchanged at 49 actual / 480 reserved seconds.
- Independent compact-artifact audit passes: five complete 2048-update histories, all batch hashes, raw configs except intended method/threshold/descriptive label, identical software and parameter counts, loss/perplexity recomputation, summary reproduction, data/checkpoint metadata consistency, and diagnostic dimensions/ranges. Current numerical/config hashes still match the passing 49-test CUDA proof. Full and zoomed learning curves visually checked.
- Final losses: SA **4.89046994**, XSA **4.80333272**, A 0.25 **4.82149275**, A 0.50 **4.81451356**, A 0.75 **4.87118640**. Every cap beats SA in this single validation screen, but none beats XSA. Best A is 0.50, **+0.01118083 nats/token versus XSA**. No training-seed or untouched-test conclusion is claimed.
- Added final DIRECTION_A_REPORT.md, DIRECTION_A_MANIFEST.json, requirement-by-requirement completion audit, raw artifacts, actual commands, failures/recovery history, diagnostic summaries and measured follow-up estimates. No control, extra seed, other research direction or downstream run was launched.

Next action: push the completed report/artifacts and verify GitHub branch equality. After delivery, review the negative comparison to XSA before deciding on any separately authorized mechanistic control or fresh-seed study.

## Direction A delivery complete

- Scientific results, source, report, plots and completion audit published as **1b6032933c5e64d891b4eaaf8558f7db3e6511d4** on GitHub main; `git ls-remote` matched the local commit and the remote worktree was clean.
- Final delivery checks confirm unchanged numerical/config hashes, all five latest/previous checkpoint pairs and original initialization preserved, no active user jobs, unchanged original job ledger, and project storage **7,195,989,506 bytes (6.702 GiB)** below 8 GiB. Evidence is in `evidence/direction_a/delivery.json`.
- The full authorized Direction A goal is achieved: implementation, correctness, profiling, all three full-budget threshold runs, five-method evaluation/diagnostics/collection, resource accounting and GitHub delivery are verified.

Next action: review DIRECTION_A_REPORT.md before choosing any separately authorized follow-up. No additional training, controls, seeds or downstream tasks are scheduled.

## Directions B/C — authorized stage, 2026-11-24

- Read the full goal attachment, handoff, reproduction specification, previous protocol/report/manifests and job ledgers. Local, Insomnia and GitHub HEAD all match `8a1093a`; both working trees are clean. Prior training/collection IDs are COMPLETED and `squeue --me` is empty. No submission made.
- New scope: B lambda 0.1/0.5/1.0 and C, same seed-17 2048-update pilot, combined nine-method comparison. Separate caps: 14400 GPU seconds, 600 verification/profile GPU seconds within it, 1800 CPU-only allocated seconds, 12 GiB total project including atomic-write headroom. Historical budgets remain unchanged.
- Current project size 7,196,015,435 bytes. `edu/short` and A6000 inventory remain available; partition maximum is 12 hours. `quota -s` gave no output and `lfs` is unavailable; determine the existing filesystem-specific quota command before launch.
- Saved the pre-change model for exact SA/XSA/A regression tests. User-supplied AGENTS instructions apply; ancestor Prime-lab guidance describes a different project type and is not applicable to this existing PyTorch experiment.

Next: write failing numerical and resume tests, implement both operators, then complete diagnostics, strict collection and the bounded GPU gate before any screening.

- B/C implementation and metadata/diagnostics/collection TDD are green: **77 local passed / 10 CUDA-only skipped**. Frozen-source SA/XSA/A outputs and gradients match exactly. New CPU tests cover formulas, fp64 gradients, causal chunk boundaries, output-head slicing, degeneracies and exact model/optimizer/data resume.
- Implemented conservative standard-library-only B/C submission guards, separate ledger, configurable Slurm resources, strict nine-method collector, independent full-checkpoint audit, and frozen protocol. The filesystem quota command is `/usr/lpp/mmfs/bin/mmlsquota -u "$USER" insomnia`: 36,217,120 KiB used of 52,428,800 KiB; current in-doubt blocks are also subtracted during the gate.
- Before launch: all historical IDs, including the old intentionally cancelled data-preparation attempt, were checked and are terminal. No active jobs. Numerical source and configs will freeze after the allocated verification/profile gate.

Next: publish implementation, run a 300-second maximum A6000 correctness/profile allocation, inspect actual proof and frozen full-screen estimate, then train only if all gates pass.

- Published source/configuration/tests/protocol as **1db7eff** and fast-forwarded the clean remote checkout. Submitted GPU verification **14076373**, key `direction-bc-verify-v1`, 300-second cap, edu/short/A6000/8 CPUs/32 GiB; observed PENDING (Priority). Reservations: **300/14400 GPU seconds**, **300/600 verification seconds**. No training submitted.

Next: inspect this exact job and its logs/proof; do not resubmit based on an observation timeout.

- **14076373 COMPLETED**, exit 0, 59 GPU seconds on ins093 A6000. **87 tests passed / 0 failed / 0 skipped**, 13.55 s. All five preserved checkpoints re-evaluate with maximum per-window loss difference **0.0**; initial tensor hashes and preserved batch histories match.
- Frozen profile and plan saved. Compute-only full-run estimates: B 0.1 **1057.56 s**, B 0.5 **1059.77 s**, B 1.0 **1059.86 s**, C **580.65 s**. Conservative training reservations **1740/1740/1740/1200 s**; all training plus full verification allowance **7020/14400 s**. Project peak estimate **11,041,321,929 bytes (10.28 GiB)**, within 12 GiB and current quota.
- Actual dispatch: SA/XSA/A/C use CUDA flash SDPA (C also uses Gram-matrix bmm); B uses explicit fp32 softmax/bmm. B profile peak allocation ~4.98 GiB versus C ~4.25 GiB. This is measured method+backend cost, not an isolated penalty-kernel comparison.

Next: submit B lambda 0.1 with the frozen 1740-second cap; inspect complete matched artifacts before proceeding to the next configuration.

- Submitted B lambda **0.1 / job 14076378**, 1740-second cap, same A6000 allocation. Observed RUNNING with finite early training loss/gradients and expected method metadata. Cumulative GPU reservation **2040/14400 s**; verification remains **300/600 s reserved, 59 s actual**.

Next: monitor 14076378 through terminal state and verify its 2048-update artifacts before B lambda 0.5. No duplicate/retry submitted.

- Independent fp32 rank-one/null-space stress found a real C guard defect: a computed zero written value could produce a small positive Gram denominator and a corrected norm of **313.35** from aggregate norm **0.7993** (seed 41). Reproduced with failing test in `geometry_guard_red.txt`. No C training has launched.
- Hardened C locally with an absolute plus dimension/dtype/scale-aware denominator floor, safe guarded division, and matching diagnostic guard reporting. Preserved the first gate/profile/plan under `verification_v1/` and the pre-hardening model source. Added an exact B output/gradient regression; B's operator and current running source are unchanged.

Next: finish local checks, wait for existing B 0.1 job 14076378, preserve its completed artifacts, then deploy and run the second 300-second GPU verification gate before any remaining training. Total verification reservations will be 600/600 seconds; no additional profile allocation beyond that.

- Hardened local suite: **79 passed / 13 CUDA-only skipped**, 4.05 s. New coverage includes CUDA bf16 written-space tolerance and B lambda-zero SDPA forward/gradient agreement (pending allocated execution). AST equality confirms ModelConfig, B operator, Attention.forward, GPT and operator metadata are unchanged; exact B regression also passes.
- Job **14076378** observed RUNNING at update **1382**, with every committed minibatch hash matching SA and all losses/gradients finite. No remote code changes have been made during this run.

Next: inspect 14076378 at completion, record its actual outcome, then synchronize the guarded C source and run the second bounded verification allocation.

- **B lambda 0.1 / job 14076378 COMPLETED**, exit 0, **1118 allocated GPU seconds (18m38s)**. Reached **2048 updates / 67,108,864 tokens**; validation CE **4.8848455101**, perplexity **132.2700294**. All 2048 batch hashes, finite losses/gradients, data/init/evaluation/config/software identities match the preserved baseline. Latest/previous files and final evaluation/diagnostics are present; full checkpoint/optimizer audit remains for the CPU collector.
- B 0.1 improves slightly on SA (CE 4.89046994) but is worse than XSA (4.80333272) in this single validation screen. No setting was changed in response to that result. Source commit was 1db7eff; C hardening changes no B computation.
- Stage actual GPU time so far **1177 seconds** (59 verification +1118 training), reservation **2040 seconds**. No job is active. The C fix was published separately as **4b6ceec**, but has not yet been deployed to Insomnia.

Next: preserve/reconcile the authoritative remote ledger, fast-forward the idle remote checkout, and submit the second 300-second verification gate. No additional screening run may start until it passes.

- Preserved the authoritative remote v1 gate/ledger and local progress in Git stash **0c37e7478e0e1f71c99b0b1987db5f9480cd63c5** before fast-forwarding the idle remote checkout to **a752a9d**. Completed B 0.1 results and reconciled terminal ledger are published; no data/checkpoint was removed or overwritten.
- Submitted hardened verification **14076777**, key `direction-bc-verify-v2`, 300-second A6000 cap. Total stage reservations **2340/14400 GPU seconds**, including **600/600 verification seconds**. Prior actual GPU time remains **1177 s** until this allocation is terminal.

Next: inspect 14076777, require all 92 CUDA tests, preserved baseline rechecks and updated resource fit, then continue B 0.5 only if the gate passes.

- Hardened verification **14076777 COMPLETED**, exit 0, **104 GPU seconds** on **ins083 RTX A6000**. **92 passed / zero failures or skips**, 42.10 s. All five preserved evaluations still match exactly. Both B zero-strength fp32/bf16 output/gradient comparisons and C fp32 null-space regression/bf16 written-space checks passed.
- Final frozen plan keeps B caps **1740 s each**, C **1200 s**. Current project **8,868,676,864 bytes**; conservative peak **12,238,243,995 bytes (11.40 GiB)**, below 12 GiB and quota. This estimate conservatively still includes four new pairs although B 0.1 already exists. Verification totals: **163 actual / 600 reserved seconds**, both within 600; no more profiling planned.
- Submitted **B lambda 0.5 / job 14076783**, key `direction-bc-b050-v1`, same edu/short A6000/8 CPUs/32 GiB, 1740-second cap. Source **a752a9d**. Cumulative GPU reservation **4080/14400 s**; confirmed completed actual GPU time **1281 s** plus this live job.

Next action: inspect **14076783** with squeue/sacct and its existing artifacts. Once terminal and fully verified, submit **b100** for 1740 s, then **c** for 1200 s, sequentially using `xsa.bc_submission`. Do not repeat B 0.1 or either verification job. Then run the 300-second CPU collector/independent checkpoint audit, inspect plots and final accounting, finish the nine-method report/completion audit, publish and verify GitHub. No source change or remote sync during the live run.

- **b050 / 14076783 COMPLETED**, exit 0, **1153 GPU seconds**, full 2048 updates / 67,108,864 tokens. CE **4.8804516420**, perplexity **131.6901272**. All ordered batch hashes, protocol identities and finite training metrics pass; checkpoint pairs/evaluation/diagnostics exist. Full checkpoint/optimizer audit remains for the CPU collector.

- Submitted **b100 / job 14076831**, key direction-bc-b100-v1, 1740-second cap, one A6000 on edu/short, after confirming earlier jobs terminal and complete artifacts. Next: inspect this exact job through completion; no duplicate/retry.

- Added independent CPU-only final accounting/preservation auditor (`scripts/audit_bc_accounting.py`), with real scheduler-format capture and red/green evidence: **10 tests passed**. It checks complete terminal states, exact requested/actual allocation records, one A6000, sequential GPU intervals, cumulative budgets, unchanged historical ledgers/checkpoint hashes/compact artifacts, and frozen source provenance. It does not modify the numerical pipeline or require GPU work.
- Confirmed completed B/C stage GPU usage now **2434 seconds** (163 verification +1118 B 0.1 +1153 B 0.5), plus current job 14076831. Reservations **5820/14400 GPU seconds**, verification **600/600**; C's planned 1200-second reservation still fits. CPU collector remains unsubmitted.

Next action: monitor existing **14076831** through terminal state and complete artifact checks. The active sequencing workflow will submit only **c**, with a 1200-second cap, after B 1.0 succeeds; check the authoritative ledger before any manual action. Then submit the 300-second CPU collector, complete final scheduler/storage/preservation audit, inspect both plots, finish the report and measured fresh-seed estimate, and publish/verify final results. No additional verification or training variants are planned.

- **b100 / 14076831 COMPLETED**, exit 0, **1150 GPU seconds**, full 2048 updates / 67,108,864 tokens. CE **4.9099621400**, perplexity **135.6342792**. All ordered batch hashes, protocol identities and finite training metrics pass; checkpoint pairs/evaluation/diagnostics exist. Full checkpoint/optimizer audit remains for the CPU collector.

- Submitted **c / job 14076871**, key direction-bc-c-v1, 1200-second cap, one A6000 on edu/short, after confirming earlier jobs terminal and complete artifacts. Next: inspect this exact job through completion; no duplicate/retry.

- **c / 14076871 COMPLETED**, exit 0, **726 GPU seconds**, full 2048 updates / 67,108,864 tokens. CE **4.8073909730**, perplexity **122.4118248**. All ordered batch hashes, protocol identities and finite training metrics pass; checkpoint pairs/evaluation/diagnostics exist. Full checkpoint/optimizer audit remains for the CPU collector.

- Submitted CPU collector/independent checkpoint audit **14076904**, key direction-bc-collect-v1, 300-second cap, edu/short/2 CPUs/8 GiB. All six GPU jobs are COMPLETED 0:0; actual GPU total **4310 s**, verification **163 s**, reservations **7020 GPU / 600 verification / 300 CPU seconds**.
- An ancillary readiness-print command expected an update key absent from the earlier B 0.1 readiness schema and raised KeyError. The guarded launcher independently checked every final evaluation and the frozen gate before submission; direct checks of all four evaluations confirm full 2048-update/67,108,864-token budgets. This was an observation-format error, not a training or collector failure. Use actual evaluation records for uniform budget checks.

Next action: inspect existing CPU job **14076904**; then capture final scheduler/storage accounting, independently audit preserved artifacts, inspect plots, finalize report/next-stage estimate, and publish to GitHub. No more GPU runs.

- **CPU collector 14076904 COMPLETED 0:0, 75 allocated seconds, ins024**. Combined nine-method collection and independent audit pass: all 18 latest/previous checkpoints loaded, finite model/optimizer state, exact complete batch/config/data/initialization/software identities, recomputed CE/PPL and paired intervals, diagnostic and frozen-source checks.
- Final measured GPU allocation **4310 seconds (71m50s)** including **163 verification/profile seconds**; reservations **7020 GPU / 600 verification / 300 CPU seconds**. All jobs terminal, sequential one-A6000 allocations; queue empty. Final measured storage 11035931384 bytes. No historical artifact deleted.
- C final CE **4.80739097297**, PPL **122.4118248**. Best extension, but XSA remains lower at 4.80333272368. C minus XSA **+0.0040582493**, paired-window interval **[-0.0031197863, +0.0113993311]**; no resolved superiority/equivalence claim.

Next action: complete independent local resource/preservation audit, visually inspect plots, finish report and measured next-stage estimate, publish and verify all checkouts. No further jobs.

- Independent local final audit **passed**: all seven scheduler records agree with reservations and elapsed/start/end/TRES, GPU intervals do not overlap, historical Part 1/A ledgers and checkpoint/compact-artifact identities are unchanged, both original/current source proofs match. Final project storage **11,035,931,384 bytes (10.278 GiB)**; frozen temporary-write peak estimate **11.40 GiB** below 12 GiB.
- Both full and zoomed nine-method curves visually inspected. Final report contains every loss/difference/interval, measured runtime/memory, B/C diagnostics, numerical conventions, defect/recovery, actual commands and limitations. Next-stage proposal is three fresh paired seeds of SA/XSA/frozen C: **1.68 GPU-hours measured projection**, **4h35m conservative GPU reservation**, **17.10 GiB projected peak** (requires a separately authorized storage/compute budget). Nothing further launched.

Next action: publish completed source/compact evidence/report, preserve the authoritative remote records before fast-forwarding, and verify local/Insomnia/GitHub commit equality and clean worktrees.

## Directions B/C delivery complete

- Published the complete scientific source/results/report as **c1cfc3265e62911a6d77937f9c567c3b02d3d215**. Local, Insomnia and GitHub main match. Remote working tree is clean; source hashes still match the 92-test GPU gate; no active user jobs remain.
- Preserved all 70 modified/untracked remote records before synchronization. Every artifact matched the local copy except the deliberately finalized manifest/progress notes; originals are retained in Git stash **83d2f75188db2240569b1655498714bec07a5f64**. Previous stash remains preserved. Raw log whitespace is kept byte-for-byte; authored files pass whitespace checks.
- Post-sync checks confirm all nine latest/previous checkpoint pairs and shared initialization remain. Project storage **11038641564 bytes**, below 12 GiB. Full checkpoint hashes/model/optimizer/RNG were audited in CPU job 14076904; post-sync presence/sizes verified separately. See evidence/direction_bc/delivery.json and the completed requirement audit.
- Full B/C goal achieved: correct resumable implementations, reproducible protocol, two bounded CUDA gates, four complete matched runs, nine-method collection/diagnostics/curves, independent audit, resource accounting, report and publication. No extension beats XSA by observed loss; C is closest and its window interval overlaps zero.

Next action: review DIRECTIONS_BC_REPORT.md and decide whether the proposed fresh-seed SA/XSA/C confirmation merits a separately authorized compute/storage budget. No further job is scheduled.

## Midpoint project report — November 24, 2026

- Added `MIDPOINT_REPORT.md` and linked it from README. The report covers SA/XSA and Directions A/B/C, implementation and Insomnia/data/training setup, verification and the C numerical failure/repair, all nine outcomes, uncertainty, cost and diagnostic analysis.
- Distinguished observations from hypotheses and proposed work: fresh-seed SA/XSA/C confirmation, independent evaluation data, partial-removal and magnitude controls, B penalty ablations, geometry interpolation, positive-only alignment caps, and controlled context tasks. No new variant, training job or benchmark was executed for this report.
- Checked every reported CE/PPL/delta row and selected interval against the collected JSON, diagnostic means against saved layer/head records, aggregate token/compute arithmetic, local document/image links and Markdown structure. `python3 -m xsa.direction_bc gate` still passes; frozen numerical sources and experiment evidence are unchanged. Rechecked the primary XSA and BLiMP sources for the report's background references.

Next research action: review MIDPOINT_REPORT.md and choose the follow-up question before defining a new scope and budget. No further experiment is scheduled; this documentation change includes GitHub publication and checkout verification.

## Midpoint report LaTeX compatibility fix

- Replaced all four unsupported operatorname macros in MIDPOINT_REPORT.md with standard roman math labels for softmax, clip and trace. Mathematical content and experimental results are unchanged. Checked that no prohibited macro remains and that math delimiters/braces are balanced.

Next research action remains review of the midpoint findings and selection of a separately scoped follow-up; no experiment was launched for this formatting fix.
