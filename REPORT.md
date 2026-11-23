# Part 1 implementation and verification report

This is the historical Part 1 verification report, written before the separately approved pilot. See [PILOT_REPORT.md](PILOT_REPORT.md) for the completed pilot and [PROGRESS.md](PROGRESS.md) for subsequent execution. Statements below about unrun pilot work describe that earlier snapshot.

**2026-11-23.** The matched SA/XSA experiment pipeline is implemented and verified on Insomnia. No pilot training, paper-scale study, extension, full FineWeb download or full downstream benchmark has been launched. All reported losses are from a deliberately tiny execution fixture.

## What was implemented

One nanoGPT-derived GPT supports `sa` and `xsa` with identical parameters, shapes, initialization and counts. XSA removes each causal attention aggregate's component along its own token value before head concatenation/output projection. RoPE, embedding LayerNorm, independent head count/dimension, fp32 projection under bf16, tied embeddings and baseline optimizer conventions are explicit. `REPRODUCTION_SPEC.md` maps paper targets to code/config/commands and labels every setting's provenance.

The pipeline includes deterministic GPT-2 document packing and manifests, fixed held-out windows, independent training/evaluation RNG, shared initialization artifacts, exact token/update accounting, warmup/cosine schedule, append-only logs, atomic latest/previous checkpoints, corruption fallback, signal handling and strict resume validation. It supplies single-GPU Slurm launchers, duplicate/job-budget guards, evaluation, layer/head diagnostics, synchronized profiling, plotting, paired-seed summaries and a pinned LM Evaluation Harness adapter. `README.md` contains the runnable next-stage commands.

## Actual execution and evidence

| Job | Work | Resource | Outcome | Elapsed |
|---|---|---|---|---|
| 14072260 | Reuse CUDA environment in isolated project venv | edu/short, CPU | COMPLETED, 0:0 | 46 s |
| 14072282 | Initial numerical tests, fixture pair/resume, diagnostics/plot, pilot profile | edu/short, one RTX A6000, ins093 | COMPLETED, 0:0 | 33 s |
| 14072287 | Pinned harness install and interface/task-ID checks | edu/short, CPU | COMPLETED, 0:0 | 171 s |
| 14072292 | Hardened implementation, shared init, bf16 GPU resume, repeated bounded checks/profile | edu/short, one RTX A6000, ins093 | COMPLETED, 0:0 | 16 s |

The two entire GPU allocations consumed **49 seconds**, conservatively including tests, startup, evaluation and plotting. The orchestrators measured 23.029 + 13.958 = **36.987 seconds** internally. Hard reservations totaled **480/600 seconds**; **120 seconds remain unreserved**, but no additional run is planned. CPU environment setup is not training/profiling. Local work consisted of correctness tests, data preparation and analysis; no independent local smoke/profile training was launched.

Initial GPU command:

```bash
ssh insomnia
cd ~/algollms/xsa
/usr/bin/python3 -m xsa.submission smoke --key smoke-v1 --seconds 360 --gpu l40s --cpus 8 --mem 32G
```

Read-only inspection showed an available A6000 and an approximately 40-minute L40S queue estimate. While **the same job was still PENDING**, its request was updated with:

```bash
scontrol update JobId=14072282 TresPerNode=gres/gpu:A6000:1
```

No cancellation, duplicate job or budget reset occurred. After checking all prior IDs, the final validation used:

```bash
/usr/bin/python3 -m xsa.submission smoke --key smoke-final-v2 --seconds 120 --gpu A6000 --cpus 8 --mem 32G
```

Scheduler evidence: [final accounting](evidence/cluster/final_accounting.psv), [queue](evidence/cluster/final_queue.txt), [job ledger](evidence/cluster/job_ledger.json). Logs and every checkpoint remain at `~/algollms/xsa/{logs,runs}`. Compact outputs and source snapshots are in [initial run](evidence/results/smoke-14072282/) and [final run](evidence/results/smoke-14072292/). [Checkpoint checksums](evidence/cluster/checkpoint_checksums.txt) identify preserved latest/previous files. The recorded GPU driver is in the Slurm logs; torch is 2.10.0+cu128, CUDA runtime 12.8, Python 3.11.5. Local tests used Python 3.13.5 / torch 2.9.1. Both environments are identified rather than conflated.

## Pass, fail, skip and recovery results

- **Final Insomnia suite: 26 passed, 0 failed, 0 skipped**, 6.74 s: [test output](evidence/results/smoke-14072292/tests.txt).
- **Final local suite: 23 passed, 0 failed, 3 skipped**, 5.29 s: [test output](evidence/local_tests.txt). The three skips require allocated CUDA: fp32 projection, bf16 projection and bf16 resume.
- Float32 reference versus SDPA forward/backward; fp64 gradcheck; orthogonality, zero/near-zero/collinear/already-orthogonal inputs; first-position behavior; causality; unequal attention/model widths; finite gradients; RoPE pair layout/offset; parameter identity; shared init; batch identity; token accounting; LR boundaries; accumulation equivalence; RNG isolation; deterministic resume; SIGUSR1 checkpoint; corrupted-latest fallback and preservation of the valid previous all pass.
- GPU fixture XSA resumed at update 4 versus uninterrupted 8 updates: **maximum parameter absolute difference 0.0**, tested atol 1e-6/rtol 1e-5. Separate bf16 GPU resume is part of the passing suite. CPU SA/XSA model and optimizer resume are bitwise equal.
- Initial model/pipeline/analysis/submission tests first failed before implementation. Later regression tests found two real recovery/analysis defects: a corrupt latest could overwrite the valid previous after fallback, and cross-seed summaries could mix unequal budgets. Both were fixed with passing tests. Red logs remain under `evidence/tdd_*`. The attempted initialization “red” invocation raced its implementation write and passed; it is explicitly not counted as red evidence in PROGRESS.
- A strict LR equality assertion failed on floating-point roundoff; changed to an appropriate tolerance, preserving the failure. Packaging initially failed because flat-layout auto-discovery included data directories; explicit `xsa*` package selection fixed it. `uv build --wheel --out-dir evidence/build-check` now passes ([build output](evidence/package_build_green.txt)).
- Independent fixture preparations on local/Insomnia Python versions produced the **same manifest SHA256 and all shard/document-index checksums**: [data reproducibility](evidence/data_reproducibility.json). No FineWeb data was downloaded.
- Pinned harness 0.4.9.2 adapter passes actual likelihood and rolling-likelihood calls and discovers all eight paper task IDs: [integration evidence](evidence/cluster/harness_integration.json). **Full task evaluations were not run.**
- Checkpoint loading, diagnostic output, collection CLI and plots work. The final [learning curves](evidence/results/smoke-14072292/curves.png) and independently recollected plot were visually checked. Curves versus walltime contain startup effects and are not timing benchmarks.

## Fixture results — execution only

Both arms used seed 17, one identical saved initial state, identical ordered batch hashes, 8 optimizer updates / 1024 tokens, and the same four held-out 32-token windows. The initial-state SHA256 is `d52111e2fa5ec05d6adc3afc99ceb12f130c5f6fbc6b36e8abf32160b4029782`.

| Method | Held-out CE | Perplexity |
|---|---:|---:|
| SA | 10.6815280914 | 43,544.0388 |
| XSA | 10.6818139553 | 43,556.4883 |

XSA−SA = **+0.0002858639**; the negative/inconclusive outcome for XSA is retained. The window-bootstrap interval is [0.0002207756, 0.0003321171], under an independent-window approximation; it is not seed uncertainty or evidence about real language modeling. There is one seed, so seed SD/SE are unavailable.

Final-layer XSA post-projection head alignments are approximately −8.55e-9 and 1.98e-9. The near-zero output fraction is 1/32 in both heads, consistent with the unmodified first-position formula. This construction-enforced geometry is not a quality result. Diagnostic sample size, context and pair masks differ from the paper and are recorded in each output.

## Measured next-stage estimate

The profile uses the **actual pilot architecture**, 6 layers, d=384, 6×64 heads, context 512, **29,938,944 unique parameters / 10,622,208 excluding token/tied head**. Both arms ran on the same RTX A6000, PyTorch SDPA, bf16 autocast with fp32 XSA projection, batch 8, eager execution, 3 warmup steps and 12 synchronized forward/backward measurements. Three full accumulation-8 optimizer updates were also measured. Diagnostics were disabled during timing.

| Final profile | SA | XSA |
|---|---:|---:|
| Mean forward/backward microstep | 31.031 ms | 34.581 ms |
| Forward/backward tokens/s | 131,995 | 118,447 |
| Mean full 32,768-token optimizer update | 244.296 ms | 274.970 ms |
| Peak allocated GPU memory | 3.611 GiB | 4.211 GiB |
| 2048-update compute-only projection | 0.1390 GPU-h | 0.1564 GPU-h |

Measured XSA overhead here is **11.44% forward/backward** or **12.56% per full update**, not a reproduction of the B200 block benchmark. The first profile gave 0.13835/0.15581 GPU-h; both measurements are preserved. Compilation was disabled and no compilation speedup is assumed. The native-bf16 versus fp32-projection comparison reduced worst absolute output/self-value cosine from 0.00259 to 0.000824 on the recorded random sample; both retain bf16 output quantization.

The next pair processes **67,108,864 tokens per arm** (32,768 × 2048), totaling approximately **0.2954 GPU-hours of measured compute extrapolation**. Data transfer, validation, checkpoint writes and startup add overhead. A proposed **one A6000 GPU-hour total reservation**, executed sequentially as two 30-minute jobs, is a practical starting budget. It is not a promise that both runs finish within that time; resumable checkpoints bound lost work.

For storage, one pilot Adam checkpoint is approximately 359 MB decimal; latest+previous for both arms is approximately 1.44 GB, and an atomic write temporarily adds another 359 MB. Shared initial tensors need about 120 MB. Approximately 100M uint16 tokens need 200 MB; document indices, source parquet cache and logs are extra. Proposed pilot cap: **5 GiB**, verified against actual source-shard sizes before preparation. Current project use: data 563 KiB, cache 1.5 MiB, runs 232 MiB, isolated overlay venv 622 MiB, excluding the pre-existing shared CUDA packages.

## Paper-scale storage and coarse compute bounds

These are **not measured billion-parameter timings**. The only full-model measurement is the 29.94M pilot. For planning scale only, parameter-times-token extrapolation of the measured pilot assumes constant effective FLOP efficiency and ignores context-attention cost, activation memory, kernel changes and larger-model utilization. It may be wrong by large factors; it must be replaced with a separately authorized architecture-specific profile before any large run. The paired values below are separate per-arm proxies, not a training recommendation.

| Architecture label | Actual total / non-embedding params | One checkpoint | Latest + previous + temporary peak, one arm | Coarse SA / XSA GPU-h proxy |
|---|---:|---:|---:|---:|
| 0.7B | 756,820,992 / 679,554,048 | 8.46 GiB | 25.37 GiB | ~5,500 / ~6,200 |
| 1.4B | 1,512,411,136 / 1,409,388,544 | 16.90 GiB | 50.71 GiB | ~11,000 / ~12,300 |
| 2.7B | 2,813,301,760 / 2,684,523,520 | 31.44 GiB | 94.32 GiB | ~20,400 / ~23,000 |

Each paper run is **104,857,600,000 training tokens**, exactly 2048 × 256 × 200000. Packed FineWeb-100BT is on the order of 200 GB decimal (~186 GiB), before raw parquet caches and document index. A paper pair needs both checkpoint sets plus one temporary save and shared data. The current **50 GiB home quota cannot support this**. Its inspected usage was ~27 GiB at the start; latest quota evidence is saved. Larger-memory GPU/distributed support and permitted project/scratch storage must be settled separately. [Machine-readable estimates](evidence/resource_estimates.json) contain the arithmetic.

## Remaining limitations and next action

The author's modified nanoGPT commit, exact dataset revision/order/split seed, RoPE details, bias/padding choices, evaluation batching, harness revision, prompting/few-shot settings and LAMBADA variant are unspecified. Our document-hash validation split preserves text but approximates the paper's ambiguous token split. The middle architecture is preserved despite the paper's 1.3B/1.4B labeling discrepancy. The pilot LR is a declared starting assumption, not a completed baseline LR search.

FineWeb ingestion is implemented and fixture packing/checksums are verified, but the real corpus download/preparation is unrun pending the storage budget. Full downstream datasets, longer pilot curves, three-seed confirmation, context sweeps, paper-scale memory fit, compilation and distributed training remain unrun. No KV cache or proposal extensions are implemented. A zero cosine or a passing smoke run does not establish that the paper is reproduced.

The next actionable decision is authorization for **one total A6000 GPU-hour, bounded CPU preparation, and 5 GiB under `~/algollms/xsa`** for the exploratory seed-17 SA/XSA pair. The exact shared-initialization, preparation, SA, XSA, resume and evaluation commands are in [README.md](README.md). The handoff explicitly requires a concrete GPU-hour allocation and launch authorization after profiling; nothing longer has been submitted.

## Repository delivery

Code and compact results were pushed to [ptshen/xsa-research](https://github.com/ptshen/xsa-research), branch `main`, through the existing GitHub SSH setup on Insomnia. Implementation commit: `72cb3c4802f39d54e2d971d38f2a1b68c3950ccb`; final documentation and raw Slurm logs are delivered in the following commit on the same branch. Checkpoints and token shards are preserved on Insomnia, with checksums in Git, rather than uploaded as large binaries.
