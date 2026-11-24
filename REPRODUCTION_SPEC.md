# Reproduction specification

This document records the original **Part 1** matched SA/XSA specification. Later extensions are specified separately: [Direction A](DIRECTION_A_PROTOCOL.md) and [Directions B/C](DIRECTIONS_BC_PROTOCOL.md). Their results and deviations are in their stage reports; fixture runs alone establish execution and numerical correctness, not reproduction of language-model results.

## Authority and pins

- Primary: Shuangfei Zhai, *Exclusive Self Attention*, [arXiv:2603.09078v1](https://arxiv.org/html/2603.09078v1), dated 10 March 2026. All tables, figure captions, and experimental sections were inspected. Figure 2 and Figure 4 labels were also inspected visually.
- Architecture/training inheritance: [nanoGPT `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291). Relevant source and MIT license are in `references/`. The paper links nanoGPT but does **not** identify the author's modified training-code commit. Our pinned upstream is an explicit approximation, not a claim about the author's exact defaults.
- Secondary references inspected: [turancannb02/xsa-marimo-notebook](https://github.com/turancannb02/xsa-marimo-notebook) and [slowrun XSA commit](https://github.com/qlabs-eng/slowrun/commit/ac968a62c633d75d972afa6d86a59f89e12997b9). Neither supplies evidence for the paper's undocumented training choices; neither is the numerical reference.
- Data: [FineWeb `sample-100BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/9bb295ddab0e05d785b879661af7260fed5140fc/README.md), revision `9bb295ddab0e05d785b879661af7260fed5140fc`. This revision/configuration is our pinned resolution of the paper's dataset link; the paper does not specify a revision.
- Downstream harness: [v0.4.9.2 / `ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9`](https://github.com/EleutherAI/lm-evaluation-harness/tree/ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9). Author's harness version is unknown.
- Portable dependency pins are in `pyproject.toml`; the **actual tested cluster environment** is separately frozen in `evidence/cluster/cluster-environment.lock.txt`. Local and cluster torch/Python versions differ and are reported in results. Reusing existing cluster CUDA packages avoids replacing unrelated environments.

## Target-to-implementation map

| Target | Source/protocol | Configuration and implementation | Command |
|---|---|---|---|
| Paired learning curves | §4.1, Table 1, Figure 3 | `configs/paper_*.json`; `xsa/model.py`, `xsa/train.py`; smaller `pilot.json` | `python -m xsa.train --config configs/pilot.json --attention sa --data data/pilot --output runs/pilot-sa-s17` (then paired XSA; allocated compute only) |
| Fixed held-out CE/PPL and seed differences | §4.1 validation; evaluation batching unspecified | fixed nonoverlapping windows, token-weighted CE; `evaluate.py`, `collect.py` | `python -m xsa.evaluate --checkpoint RUN/checkpoint.pt --data DATA --output RUN/evaluation.json` |
| Similarity bias | §2, Figure 1: 1024 random training sequences, length 2048; average across heads/sequences for each layer | `diagnostics.py`: per-head data retained, explicit pair masks; bounded default 4 sequences × 32 tokens | `python -m xsa.diagnostics --checkpoint RUN/checkpoint.pt --data DATA --output RUN/diagnostics.json --samples 4 --context 32` |
| Context trend | §4.2/Figure 5: 512,1024,2048,4096,8192,16384; constant tokens/batch and same LR | `context_sweep.json`, `scripts/context_config.py`, 1.4B dimensions, LR 4e-4 | `PYTHONPATH=. python scripts/context_config.py --context 4096 --output configs/context4096.json` |
| Computation/memory overhead | §4.2/Figure 2: attention+FFN block, B200, bf16, batch 32; context or d_model sweep over 512..16384, other axis fixed at 2048 | `benchmark.py --block-only`; full pilot model is separately measured for budgeting | `python -m xsa.benchmark --config configs/paper_1.4b.json --batch 1 --block-only --output block.json` (hardware/batch deviation) |
| Downstream | Table 2, §4.2; eight tasks and metric selection below | pinned harness adapter in `downstream.py` | `python -m xsa.downstream --checkpoint RUN/checkpoint.pt --num-fewshot 0 --output RUN/downstream.json` |

Attention-sink experiments and Figure 4 LR robustness are contextual secondary material, not additional authorized training. Figure 4 labels use 1e-4, 2e-4, 4e-4, 6e-4 on the middle architecture. No grid or context sweep has been launched.

## Settings and provenance

**R** = explicitly reported in the paper. **I** = inherited from pinned nanoGPT. **A** = assumption or engineering choice. Each JSON field is covered by this table.

| Setting | Value / behavior | Status |
|---|---|---|
| Attention | scaled causal QK softmax then V aggregate; XSA subtracts projection onto current token's V independently per head before output projection | R, Algorithm 1 (equations omit scale but pseudocode uses SDPA) |
| Paper dimensions and peak LR | 24/1536/6/256/5e-4; 24/2048/24/128/4e-4; 32/2560/24/128/3e-4 | R, Table 1 |
| Position encoding | RoPE in Q/K; no learned positions | R |
| RoPE details | full head, adjacent even/odd pairs, base 10,000, positions start at zero for each sampled window | A; layout/base not reported |
| Embedding normalization | LayerNorm directly after token embedding and before embedding dropout | R for added LayerNorm; placement relative to dropout A |
| LayerNorm | eps 1e-5, learned scale, no bias | I from train.py `bias=False`; insertion at embedding is R |
| Remaining block | pre-norm residual attention and 4× GELU MLP; final LN | I |
| Projection widths | QKV: d_model→3×heads×head_dim; output: heads×head_dim→d_model | R; Algorithm 1's square shorthand generalized to Table 1 |
| Initialization | normal std .02; residual projection std .02/sqrt(2×layers); tied token/head weight; same RNG draws across methods | I |
| Vocabulary | GPT-2 tokenizer 50,257 real IDs; 50,304 model logits including padded unused IDs, as upstream model default | I/A; exact author padding not specified |
| Dropout / bias | 0.0 / false in all experiment configs | I from train.py defaults; nonzero dropout used in resume tests |
| XSA epsilon | `F.normalize(...,eps=1e-12)` | I from PyTorch default implied by R pseudocode |
| Projection precision | fp32 accumulation for bf16/fp16, cast output back; float64 kept for reference gradcheck | A, measured against native bf16 projection in profile output |
| Attention backend | PyTorch SDPA automatic kernel selection; explicit attention only in tests/diagnostics | R/I; actual kernel depends on installed torch, dtype and hardware |
| Optimizer | AdamW, betas (.9,.95), eps 1e-8, weight decay .1 for matrices, zero for vectors; grad-norm clip 1 | R for AdamW; I for details |
| Paper schedule | 200,000 optimizer updates; 2,000 warmup; cosine to 0.1× peak | R |
| Schedule indexing | zero-based, LR=0 on first update; peak at update 2000; floor at index 200000 (after last update) | I; explicitly retained boundary behavior |
| Paper batching | context 2048, global 256 sequences = 524,288 tokens/update; microbatch 1, accumulation 256, world_size 1 | R for global/context; A for single-GPU decomposition |
| Paper total tokens | **104,857,600,000** per run, not rounded to 100B | Derived exactly from R schedule |
| Precision / compilation | bf16 autocast, fp32 parameters/Adam; no scaler; eager by default; optional compile in benchmark | A; paper specifies bf16 for benchmark only; upstream compile default is true, deliberately disabled for bounded verification |
| Train seed / data seed / eval seed | 17 / 1017 / 2017; confirmations 17,29,43 with data seed = seed+1000 and common eval seed | A; author seeds not reported |
| Pilot | 6 layers, d=384, 6×64 heads, context 512; 29,938,944 unique parameters; 10,622,208 excluding token/tied head | A, explicitly scaled-down study |
| Pilot budget | 8 sequences × 8 accumulation × 512 = 32,768 tokens/update; 2048 updates = **67,108,864 tokens** | A |
| Pilot schedule | peak 6e-4, warmup 64, cosine floor 6e-5; same LR for paired XSA | A, inherited starting LR; no baseline grid search yet |
| Evaluation / checkpoints | pilot eval every 128 updates, 64 held-out windows, checkpoint every 64; paper every 2000/1000 and 1024 windows | A; these counts are not reported evaluation settings |
| Smoke | 2 layers, d=32, 2×16 heads, context 32; 8 updates ×128 =1024 tokens; 4 eval windows; eval/checkpoint every4 | A, fixture only |
| Labels, output paths, hardware | resolved manifest/config records; one active GPU experiment | A and handoff constraint |

Parameter counting uses unique learned tensors: total counts tied embeddings once; **non-embedding subtracts the token embedding/tied LM head entirely**. nanoGPT's misleading `non_embedding` helper only subtracts position embeddings; we do not use that definition. Actual counts are in `evidence/config_counts.json`. Table 1's middle model has 1,409,388,544 non-embedding parameters here; the text/plots/Table 2 say 1.3B in places. No explanation or alternative dimensions is supplied, so we preserve **24/2048/24/128** and record the naming discrepancy.

## Data and exact resume contracts

The paper describes randomly splitting 0.05% of tokens, without a seed, splitting unit, or packing algorithm. We instead hash **whole document IDs** with seed 2026 into a validation probability of 0.0005, append one GPT-2 EOS per document, and pack source-order tokens within each split. This is a documented approximation; the realized token fraction is reported and need not be exactly 0.05%. No token permutation is used. There is no additional test split. Source order is lexicographic pinned `sample/100BT/*.parquet` then row order. Downloads are bounded by an explicit document cap, but the consumed parquet shard is downloaded in full; a tiny document cap is not a byte cap.

The manifest records revision, tokenizer/version, EOS, source order, split seed, per-document identity/text hash/order/token count, split counts, little-endian uint16 shard sizes/checksums, and the document-index checksum. Shards may divide a document, dropping only candidate windows that cross a shard boundary. Training starts are uniformly sampled over valid starts across shards; training windows can overlap each other. Train/validation documents are disjoint by ID. Fixed validation windows have stride context+1 and are sampled without replacement, so even their input/target boundaries do not overlap. A short/missing validation split raises an error; there is no fallback corpus.

Training-data RNG is a dedicated NumPy generator; evaluation uses a separate fixed-seed generator and eval mode. Each optimizer update logs a combined SHA256 of its ordered microbatches. Both pilot arms load one immutable `xsa.initialize` state file with validated architecture/seed; the smoke verifies that path. Matching seeds also deterministically generate identical complete initial tensors, verified by a full-state hash and tests. Token/update counters refer only to completed optimizer updates, not microsteps. World size is not silently multiplied into accumulation; this version rejects distributed execution.

Checkpoints contain model, Adam moments, next schedule index, all Python/NumPy/torch/CUDA RNG states, training-data generator and sequence counter, batch hashes, tokens, resolved config, data/evaluation fingerprints, software and elapsed time. Writes use fsync and atomic rename, retaining latest plus previous. Resume rejects configuration/data changes. If the current file is absent/unreadable it reports fallback to previous and quarantines the damaged file before further checkpoint rotation. Saved checkpoints are trusted pickle artifacts only. No KV cache is implemented, so cache-offset tests are not applicable; RoPE offsets themselves are tested.

The Slurm trainer requests a USR1 warning 120 seconds before walltime; it stops after a completed optimizer update and saves. A killed mid-update process resumes from the last completed checkpoint, including the old data/RNG position, rather than skipping partially consumed data. No unlimited requeue policy is assumed. `--no-requeue` prevents silent duplicated bounded jobs. Logs are append-only; evaluation collection checks matching budgets and sample hashes. Smoke reservations conservatively count the entire requested allocation against 600 seconds and are never automatically reclaimed.

## Evaluation and diagnostics

Evaluation averages token cross-entropy over fixed equal-length nonoverlapping held-out windows and reports exp(CE). Paired collection rejects unequal token budgets, initialization, hyperparameters, data, or evaluation-window fingerprints. It preserves each seed, reports XSA−SA loss, and reports seed SD/SE only with multiple paired seeds. Window bootstrap intervals describe evaluation-sample uncertainty (independence approximation), not training-seed variation. Correlated windows from a shared document can make those intervals optimistic.

Diagnostics retain one result per layer/head. Both all nonself value pairs and causal nonself pairs are reported. Diagonal attention mass includes the first position. Before/after alignment excludes vectors of norm ≤1e-6 to avoid numerical cancellation determining a meaningless angle; zero/near-zero fractions are separate and include every position. Aggregate, corrected, and removed-update norms are in head-value space, before output projection. A construction-enforced zero alignment does not imply better language modeling. The paper's value-pair/self-pair selection is unspecified; this convention is an explicit approximation. Diagnostic fp32 forwards and smaller sample/context bounds are labeled.

| Paper task | Assumed pinned harness ID | Required metric |
|---|---|---|
| ARC-Easy | `arc_easy` | `acc_norm,none` |
| BoolQ | `boolq` | `acc,none` |
| HellaSwag | `hellaswag` | `acc_norm,none` |
| LAMBADA | `lambada_openai` | `acc,none` |
| OpenBookQA | `openbookqa` | `acc_norm,none` |
| PIQA | `piqa` | `acc_norm,none` |
| SocialIQA | `social_iqa` | `acc,none` |
| WinoGrande | `winogrande` | `acc,none` |

Metric types are reported in §4.2. Task IDs, LAMBADA variant, zero-shot setting, raw completion prompts, harness task defaults and seeds are assumptions. No chat template is used. The adapter scores continuations with GPT-2 tokenization, normal causal probabilities, and left truncation to the configured context; overlong continuations fail explicitly. All eight tasks use likelihood rather than free generation. Full downstream task runs await trained checkpoints and a budget; a fixture score would not replicate Table 2.
