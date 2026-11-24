# SA / XSA on Insomnia

Read [MIDPOINT_REPORT.md](MIDPOINT_REPORT.md) for the project overview, research directions, experimental workflow, nine-method results, analysis and proposed next steps.

Directions B/C are complete: all four new runs and the combined nine-configuration audit passed. No extension beat XSA in this one-seed validation screen; C was closest. See [DIRECTIONS_BC_PROTOCOL.md](DIRECTIONS_BC_PROTOCOL.md), [DIRECTIONS_BC_REPORT.md](DIRECTIONS_BC_REPORT.md), and [PROGRESS.md](PROGRESS.md) for results, limitations and verified job states.

Direction A is complete: all three thresholds were trained and verified. See [DIRECTION_A_REPORT.md](DIRECTION_A_REPORT.md) for the full comparison and limitations. See [DIRECTION_A_STATUS.md](DIRECTION_A_STATUS.md) and [DIRECTION_A_PROTOCOL.md](DIRECTION_A_PROTOCOL.md).

The **approved one-seed pilot is complete**. See [PILOT_REPORT.md](PILOT_REPORT.md) for results, [PILOT_MANIFEST.json](PILOT_MANIFEST.json) for paired checks and budgets, and [PILOT_STATUS.md](PILOT_STATUS.md) for completion status. The Part 1 report records the earlier verification stage.

A matched, resumable PyTorch baseline for Part 1 of `XSA_Insomnia_complete_handoff.md`. Start with [REPORT.md](REPORT.md) for actual results and [PROGRESS.md](PROGRESS.md) for jobs, failures and the next action. [REPRODUCTION_SPEC.md](REPRODUCTION_SPEC.md) distinguishes paper settings, upstream inheritance and assumptions.

Remote root: **`~/algollms/xsa`**. GitHub: https://github.com/ptshen/xsa-research. All remote commands below start there. The exploratory pilot is complete; the paper-scale training matrix remains unlaunched. Directions A, B and C are implemented and screened. No further experiment is scheduled.

## Layout

- `xsa/model.py`: one model, `attention_type=sa|xsa|alignment_cap|value_penalty|output_geometry`, independent head width, RoPE, embedding LayerNorm, tied head.
- `xsa/data.py`, `config.py`, `initialize.py`, `train.py`: pinned document preparation, exact budgets, shared initialization, independent RNG streams, periodic/signal checkpoints and resume.
- `xsa/evaluate.py`, `diagnostics.py`, `benchmark.py`, `collect.py`, `downstream.py`: fixed held-out evaluation, bounded matrices, synchronized timing, paired summaries/plots, pinned harness adapter.
- `configs/`: smoke, 29.94M pilot, three Direction A thresholds, three B strengths and C, Table 1 architectures, three-paired-seed confirmation and constant-token context sweep.
- `slurm/`, `xsa/submission.py`: actual `edu/short` resources, one GPU, duplicate checks and persistent 600-second smoke reservation guard.
- `tests/`, `evidence/`: numerical/integration tests, red/green evidence, scheduler/environment records and compact results. Checkpoints remain under remote `runs/`.

## Environments and local correctness

Portable environment pins are in `pyproject.toml`; the recorded local checks used the existing Python 3.13.5 / torch 2.9.1 installation. A clean portable install is:

```bash
uv venv --python 3.13 .venv-local
uv pip install --python .venv-local/bin/python -e '.[test]'
OMP_NUM_THREADS=1 .venv-local/bin/python -m pytest -q
PYTHONPATH=. .venv-local/bin/python scripts/inspect_configs.py
```

Insomnia uses `.venv/bin/python`, Python 3.11.5 / torch 2.10.0+cu128, reusing the user's existing university CUDA package installation without changing it. `slurm/setup.sbatch` documents the actual setup; **do not rerun it over the verified environment**. Exact installed packages and wheel hashes are in `evidence/cluster/cluster-environment.lock.txt`; the harness extension has a separate lock. Anaconda modules are blocked on login nodes. Use Slurm for imports/tests/preprocessing/training that do substantial computation. The submission guard uses `/usr/bin/python3` and only the standard library on the login node.

```bash
ssh insomnia
cd ~/algollms/xsa
squeue -u "$USER"
sacct -j 14072260,14072282,14072287,14072292 -X -o JobID,JobName,State,Elapsed,ExitCode
cat evidence/job_ledger.json
```

Inspect all job IDs in the current ledger, not just the examples. A missing log while a job is pending is normal. Observation timeouts are not job failures. The guard refuses duplicate keys and simultaneous project jobs and records a reservation before calling `sbatch`, so uncertain submission cannot silently produce a second job.

## Reproduce the bounded checks

The completed smoke run(s) and remaining reservation are recorded in PROGRESS. **Do not automatically repeat them**. The guard accounts for the full requested walltime, including correctness, startup, evaluation and plotting, against the 600-second total. The actual initial command was:

```bash
/usr/bin/python3 -m xsa.submission smoke --key smoke-v1 --seconds 360 --gpu l40s --cpus 8 --mem 32G
# While job 14072282 was still pending, its request was changed in place:
scontrol update JobId=14072282 TresPerNode=gres/gpu:A6000:1
```

Both arms and profiling ran sequentially on the same A6000. The smoke job runs tests, prepares a small **fixture** once, trains a tiny model, checks interrupted versus uninterrupted execution, loads/evaluates checkpoints, writes diagnostics and plots, and profiles the real pilot architecture. The fixture is never accepted by a FineWeb configuration.

A short interactive profiling command for a **future authorized budget** is:

```bash
srun -A edu -p short --gres=gpu:A6000:1 -c 8 --mem=32G -t 00:02:00 --pty bash
cd ~/algollms/xsa
OMP_NUM_THREADS=1 .venv/bin/python -m xsa.benchmark --config configs/pilot.json --batch 8 --warmup 3 --iterations 12 --output runs/interactive-profile.json
```

Interactive execution bypasses the ledger; reserve its walltime before using it. It is provided as a command, not an instruction to consume more smoke budget. `--block-only` selects attention+FFN timing; `--compile` separately reports first-step/compilation cost. No compilation was used in the recorded profile.

## Prepared pilot data

The approved pilot uses `data/pilot-v2`: 99,962,616 training tokens and 36,418 validation tokens, prepared by job **14072321**. The first HTTP attempt, **14072309**, was cancelled for insufficient throughput; its logs and data directory remain. The successful retry used parallel Xet download with the chunk cache disabled. Preparation reads pinned `sample-100BT` parquet shards in filename/row order, uses GPT-2 tokenization, appends EOS, assigns whole documents to deterministic splits, and records all output checksums. A source parquet shard is downloaded in full even when only its prefix is consumed.

```bash
# Executed successfully as 14072321. Inspect the existing job; do not resubmit.
XSA_MAX_TOKENS=100000000 XSA_MAX_SHARDS=1 XSA_PREPARE_INITIAL=1 \
  /usr/bin/python3 -m xsa.submission prepare --key pilot-data-v2 \
  --data data/pilot-v2 --max-documents 150000 --walltime 00:25:00 --cpus 4 --mem 16G
# Validation and shared initialization also ran inside this CPU job:
.venv/bin/python -m xsa.data validate --output data/pilot-v2
```

150,000 documents is a bounded prefix, not an assertion of an exact token count. Read the resulting manifest. It must provide at least 64 disjoint validation windows of 513 tokens and enough training data for the intended study; the loader fails if validation is too short. The training sampler draws with replacement and the exact 67,108,864-token budget is independent of corpus size. The first pilot is a scaled-down, prefix-data approximation of FineWeb-100BT, not the paper corpus experiment. To prepare more data, select a new output directory and explicitly approve the revised cap; the writer refuses to overwrite an existing preparation.

No source dataset is bundled in Git. At roughly 100M packed tokens, uint16 requires about 200 MB plus validation and document index; reserve **5 GiB** for a pilot pair, packed prefix, initial state, checkpoints, logs and a bounded source-shard cache. Inspect quota before preparation:

```bash
/usr/lpp/mmfs/bin/mmlsquota -u "$USER" insomnia
du -sh data cache runs
```

The full ~100B-token corpus alone needs about 200 GB decimal packed storage and does not fit the current 50 GiB home quota. It requires a separately authorized project/scratch allocation, raw-cache budget and retention policy before download.

## Approved pilot execution

Approved allocation: one A6000 at a time, **one GPU-hour total reserved across the two arms**, 8 CPUs, 32 GiB RAM, plus bounded CPU preparation and 5 GiB storage. SA **14072327** and XSA **14072338** each have a 30-minute cap. These are the executed commands and script entry points, not requests to submit duplicates. Exact `sbatch` argument arrays are preserved in [the ledger](evidence/job_ledger.json); the bounded authorization is in [pilot_budget.json](evidence/pilot_budget.json).

The successful preparation job created the shared seed-17 initialization once:

```bash
mkdir -p runs/initial
.venv/bin/python -m xsa.initialize --config configs/pilot.json --seed 17 --output runs/initial/pilot-s17.pt
```

SA was submitted first, with post-training evaluation and diagnostics included:

```bash
XSA_POST_EVALUATE=1 /usr/bin/python3 -m xsa.submission train --key pilot-sa-s17-v1 \
  --config configs/pilot.json --attention sa --seed 17 \
  --initial-state runs/initial/pilot-s17.pt --data data/pilot-v2 --output runs/pilot-sa-s17 \
  --account edu --partition short --gpu A6000 --cpus 8 --mem 32G --walltime 00:30:00
```

XSA was submitted with an `afterok` dependency. Its script requires SA's final evaluation to contain **2048 updates / 67,108,864 tokens**, then uses the same data, initialization, LR and GPU type. A dependent CPU collector checks both checkpoints, all batch hashes, budgets and scheduler outcomes, then publishes compact results:

```bash
sbatch --parsable --no-requeue --kill-on-invalid-dep=yes --dependency=afterok:14072327 \
  --account=edu --partition=short --gres=gpu:A6000:1 --cpus-per-task=8 --mem=32G \
  --time=00:30:00 --job-name=xsa-pilot-xsa --output=logs/pilot-xsa-s17-v1-%j.log slurm/pilot_xsa.sbatch
sbatch --parsable --no-requeue --kill-on-invalid-dep=yes --dependency=afterok:14072338 \
  --account=edu --partition=short --cpus-per-task=2 --mem=8G --time=00:03:00 \
  --job-name=xsa-pilot-collect --output=logs/pilot-collect-%j.log slurm/pilot_collect.sbatch
```

For a stopped run, inspect its job and checkpoint before submitting a new, uniquely keyed continuation; preserve the same config/data/initial-state arguments:

```bash
/usr/bin/python3 -m xsa.submission resume --key pilot-sa-s17-resume1 \
  --config configs/pilot.json --attention sa --seed 17 \
  --initial-state runs/initial/pilot-s17.pt --data data/pilot-v2 --output runs/pilot-sa-s17 \
  --gpu A6000 --cpus 8 --mem 32G --walltime 00:30:00
```

Additional time beyond the approved pair allocation needs a revised budget. No automatic requeue is assumed. Latest and previous checkpoints are retained; interruption before an optimizer boundary rolls back data/RNG to the last saved complete update. SIGUSR1 requests a checkpoint and clean exit at an update boundary. Very long updates may require a larger warning margin than the current 120 seconds.

`configs/confirmation.json` describes three paired seeds; it is not an auto-submitter. Paper configurations contain the reported dimensions and 104,857,600,000 tokens per run but have not been memory/profile-validated. In particular, the 2.7B architecture needs larger-memory hardware or additional memory/distributed engineering; single-A6000 fit is not claimed.

## Evaluation, diagnostics, collection and downstream tasks

Run on an allocation; the following examples assume completed pilot checkpoints:

```bash
.venv/bin/python -m xsa.evaluate --checkpoint runs/pilot-sa-s17/checkpoint.pt --data data/pilot-v2 --output runs/pilot-sa-s17/evaluation.json
.venv/bin/python -m xsa.evaluate --checkpoint runs/pilot-xsa-s17/checkpoint.pt --data data/pilot-v2 --output runs/pilot-xsa-s17/evaluation.json
.venv/bin/python -m xsa.diagnostics --checkpoint runs/pilot-xsa-s17/checkpoint.pt --data data/pilot-v2 --samples 4 --context 128 --output runs/pilot-xsa-s17/diagnostics.json
.venv/bin/python -m xsa.collect \
  --results runs/pilot-sa-s17/evaluation.json runs/pilot-xsa-s17/evaluation.json \
  --runs runs/pilot-sa-s17 runs/pilot-xsa-s17 --plot runs/pilot-curves.png --output runs/pilot-paired.json
.venv/bin/python -m xsa.downstream --checkpoint runs/pilot-sa-s17/checkpoint.pt --num-fewshot 0 --output runs/pilot-sa-s17/downstream.json
```

The downstream script requests all eight paper tasks and reports the selected accuracy/normalized-accuracy metrics. Prompting, zero-shot setting and LAMBADA variant remain assumptions; this is not exact Table 2 replication. Full task downloads/evaluation need a separate evaluation budget. The adapter's local interface test does not run those datasets.

From the local repository:

```bash
bash scripts/sync_to_insomnia.sh       # additive; never deletes remote outputs
bash scripts/collect_from_insomnia.sh # compact logs/results; leaves .pt files on Insomnia
```

Do not run a source sync over an active experiment. Keep all run logs and latest/previous checkpoints until reviewed and backed up. No cleanup is automatic. Source-shard caches may be removed manually only after validating packed shards and agreeing that raw re-download is acceptable. The university's retention guarantees for this home/project storage have not been established; this work does not assume indefinite retention.
