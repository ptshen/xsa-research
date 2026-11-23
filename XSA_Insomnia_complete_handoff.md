# XSA research and reproduction: Insomnia handoff

Prepared for Patrick Shen · November 23, 2026

## How to use this file

Give this entire file to Codex and send:

> Follow Part 1 and implement the XSA reproduction setup for Insomnia. Use Part 2 as background for the later research extensions and Part 3 for source material. Complete correctness tests and bounded smoke/profile checks first. Return actual results, unresolved paper/protocol differences, and exact Slurm commands for the first paired training experiment.

**Current priority:** establish a reliable standard-attention/XSA baseline using existing university computing access, then investigate the three proposed modifications. No personal cloud spending is assumed or authorized.

**Scope:** Part 1 contains the current implementation instructions. Part 2 preserves the earlier full research proposal; its 7.95B-token matrix and five-week schedule are provisional future scope, not an instruction to launch those runs now. The economical progression is a small SA/XSA pilot, screening of the extensions, and three-seed confirmation of the strongest extension with both baselines.

**Research directions:** A caps excessive output/self-value alignment; B penalizes positive value similarity in attention logits; C projects using the geometry induced by each head's output matrix.

---

## Part 1 — Codex implementation instructions

You are implementing a research codebase to reproduce **Exclusive Self Attention (XSA)** by Shuangfei Zhai, arXiv:2603.09078. Build working code, configurations, tests, and evaluation scripts. Do not stop after writing a plan.

The immediate objective is a trustworthy **standard attention versus XSA** comparison. Part 2 contains the research proposal for later extensions; do not implement those extensions yet. Establish the baseline first.

### Scope and execution budget

- Inspect the existing repository, its instructions, and the available environment before making changes. Preserve unrelated work.
- Use the existing environment and any already allocated hardware. Do not provision cloud machines or incur external charges.
- You may run correctness tests and a bounded smoke/profile run, with at most 10 minutes of training/profiling on existing hardware. This authorizes setup and verification, not a full training study.
- If no GPU is available, complete CPU correctness and smoke tests, prepare GPU launch commands, and clearly identify checks that remain unrun.
- Prepare both paper-scale and smaller pilot configurations. Do not download the full training corpus or launch the long runs until a compute and storage budget is supplied.
- Resolve routine implementation choices yourself, recording assumptions. Ask only when a missing decision blocks useful work.


### Insomnia execution requirements

- Target Columbia's Insomnia cluster through Slurm. Use existing course, research-group, or free-tier access only where this project is permitted. No paid cloud resources are authorized.
- Inspect actual account associations, allowed partitions, GPU types, walltime limits, software modules, and storage quotas. Use current cluster documentation and read-only scheduler queries when accessible. Do not invent account names or assume that every listed GPU is available to this user.
- Default to one GPU per experiment and one active experiment while profiling. Make account, partition, GPU type, CPU count, RAM, walltime, and output directories configurable.
- Provide `sbatch` scripts and a short interactive profiling command. Heavy computation must run on allocated compute nodes, not login nodes.
- Record the exact GPU model for each run. Keep paired comparisons on the same GPU type where possible, and never treat timings across different GPU types as an attention-method speed comparison.
- Checkpoint periodically and before the walltime limit. Save atomically, keep a valid previous checkpoint, and restore model/optimizer/RNG/data progress when restarting. Handle scheduler warning signals if available; do not assume a particular preemption policy or unlimited automatic requeues.
- Store datasets and checkpoints in permitted project/scratch storage, respect quotas, and document retention/cleanup. Preprocess data once and reuse it across experiments.
- Produce a small fixed-token pilot configuration (approximately 30M parameters, context 512, 50M–100M tokens per run) for SA and XSA. Print the actual parameter count, including and excluding embeddings. Label this a scaled-down study.
- The first execution remains the bounded correctness/smoke/profile work specified above. The longer pilot needs a concrete GPU-hour allocation and launch authorization after profiling.
- Once the baseline is understood, the economical follow-up is to screen all three proposal extensions and then confirm standard attention, XSA, and the best extension across three seeds. The full five-method proposal remains a stretch scope.
- If SSH, VPN, or cluster access is unavailable, continue local code preparation and lightweight tests. Report the access blocker and runnable cluster commands without claiming GPU checks ran.

### 1. Establish the reproduction specification

Read the paper, including tables and figure captions:

- Paper: https://arxiv.org/abs/2603.09078
- Full text: https://arxiv.org/html/2603.09078v1
- PDF: https://arxiv.org/pdf/2603.09078
- Base code: https://github.com/karpathy/nanoGPT

The paper is the authority on the experiment. Use these community implementations as secondary references, not as proof of correctness:

- https://github.com/turancannb02/xsa-marimo-notebook
- https://github.com/qlabs-eng/slowrun/commit/ac968a62c633d75d972afa6d86a59f89e12997b9

Pin the upstream code commit and dependency versions. Record the paper version and whether the original training-code commit is publicly specified. Do not claim that today's nanoGPT defaults are necessarily the author's exact defaults.

Create `REPRODUCTION_SPEC.md`, mapping each target result to its source, configuration, implementation, and evaluation command. Mark every setting as explicitly reported, inherited from a pinned implementation, or assumed. Record ambiguities and deviations rather than silently filling them in.

Primary target: paired language-model training and validation curves. Secondary targets: the paper's similarity diagnostics, context-length trend, computational overhead, and downstream evaluations. Extract exact protocols from the paper; do not invent evaluation details.

### 2. Implement the matched models

Use one model implementation with `attention_type = sa | xsa`. The two variants must have identical parameter names, shapes, initialization, and parameter counts. The only experimental difference should be the XSA correction.

Match the reported nanoGPT modifications: RoPE, LayerNorm immediately after token embeddings, and independently configurable head count and head dimension. Keep the remaining baseline architecture consistent with the pinned source unless the paper specifies otherwise.

For each head, compute the ordinary causal attention aggregate `y`, then:

```python
v_hat = normalize(v, dim=-1)
z = y - (y * v_hat).sum(dim=-1, keepdim=True) * v_hat
```

Apply the correction before concatenating heads and applying the output projection. Do not replace it with diagonal masking, subtraction of only `a_ii * v_i`, projection after the output matrix, or projection against the residual-stream vector. Preserve the first-position behavior implied by the formula.

Allow `attention_width = n_heads * head_dim` to differ from `d_model`: Q/K/V map from `d_model` to `attention_width`, and the output projection maps back. Do not retain nanoGPT's dimension constraints where they conflict with this requirement. Verify scaling, RoPE layout, initialization, weight tying, and normalization against the specification.

Prepare Table 1 configurations, verifying these values against the paper:

| Paper label | Layers | d_model | Heads | Head dimension | Peak learning rate |
|---|---:|---:|---:|---:|---:|
| 0.7B | 24 | 1536 | 6 | 256 | 5e-4 |
| 1.4B | 24 | 2048 | 24 | 128 | 4e-4 |
| 2.7B | 32 | 2560 | 24 | 128 | 3e-4 |

Print actual parameter counts with explicit definitions. Investigate the paper's 1.3B/1.4B labeling discrepancy; use architecture dimensions to identify the model rather than changing dimensions to force a name.

Use a float32 reference for numerical checks. Make projection precision and normalization epsilon explicit. If a safer numerical implementation differs from the paper's pseudocode, record that engineering choice and measure its effect. Keep the normal training path compatible with efficient attention; collect attention matrices only in bounded diagnostic runs.

### 3. Make data and training reproducible

Resolve the exact FineWeb-100BT subset and revision, use GPT-2 tokenization, and record document order, EOS handling, packing, split procedure, seeds, and shard checksums in a manifest. Never silently fall back to synthetic data or another corpus.

The paper describes a 0.05% validation split but may not fully specify the splitting unit and seed. Verify the available information. If unresolved, implement a deterministic, documented split that preserves contiguous text, and label it an approximation. Do not randomly shuffle individual tokens into a new language-model corpus. Prevent overlapping training/evaluation windows and document any additional held-out test split.

Provide separate configurations for:

1. **Smoke:** tiny local fixture, a few optimization steps; verifies execution only.
2. **Pilot:** a declared smaller model and token budget; tests feasibility and qualitative behavior.
3. **Paper-scale:** reported model dimensions, data, and training schedule.

For the default paper-scale setup, verify context length 2048, global batch 256 sequences, 200,000 optimizer updates, AdamW, 2,000 warmup updates, and cosine decay to one-tenth of peak learning rate. Calculate exact tokens processed from the configuration rather than rounding the budget internally.

Define global tokens per optimizer update as:

```text
world_size * microbatch_sequences * accumulation_steps * sequence_length
```

Check divisibility and avoid double-counting world size when adapting nanoGPT's accumulation logic. Count optimizer updates separately from microsteps. For context sweeps, hold global tokens per update and total training tokens fixed; extract the paper's context lengths and learning-rate policy.

Within each seed, load the same initial state into SA and XSA and feed identical minibatch indices in identical order. Separate training-data RNG, evaluation RNG, and model RNG so evaluation does not change future training batches. Validate this with logged batch hashes. Use the same baseline-selected learning rate for its paired XSA run.

Prepare a one-seed exploratory SA/XSA pilot and a three-paired-seed confirmation configuration. One seed is for screening and feasibility, not a robust scientific conclusion. Do not automatically launch either configuration or multiply the full paper-scale compute budget by three.

Save model, optimizer, scheduler/scaler where applicable, all necessary RNG states, data position/state, resolved configuration, software versions, and tokens seen. Implement resume without resetting the schedule or replaying/skipping data unintentionally.

### 4. Verify correctness before training

Write and run meaningful tests covering:

- Reference attention versus the efficient path on small tensors with dropout disabled.
- XSA orthogonality per token and head, with dtype-appropriate absolute/relative tolerances and explicit handling of zero vectors.
- No-op behavior when the aggregate is already orthogonal to the self-value, plus collinear and near-zero cases.
- First-position attention output, without overriding it to make the test pass.
- Causality: changing future inputs leaves earlier outputs unchanged.
- Shapes and backward passes when `attention_width != d_model`.
- Finite gradients, with float64 gradient checking on a small nondegenerate reference case.
- Identical SA/XSA parameters before training and identical sampled batches within a paired run.
- Correct token accounting, LR schedule boundaries, and gradient accumulation versus an equivalent larger batch in a deterministic test.
- Checkpoint/resume versus an uninterrupted deterministic run, within documented tolerances.
- If cached decoding is implemented, equivalence with full-sequence inference, including RoPE offsets and alignment to the current token's self-value.

Separate CPU correctness results from hardware-dependent numerical and performance checks. Do not claim a test passed if it was skipped.

### 5. Implement evaluation and diagnostics

Evaluate fixed held-out token windows in evaluation mode and compute token-weighted cross-entropy. Report perplexity, paired loss differences, learning curves versus both tokens and elapsed time, and each seed's results. Avoid comparing different checkpoints, token budgets, or evaluation samples without clearly identifying that comparison.

Implement layer/head diagnostics for value-vector cosine similarities, diagonal attention mass, and aggregate/self-value cosine similarity. Record XSA alignment both before and after correction, update norms, and the fraction of zero outputs. Specify averaging conventions, including whether self-pairs and masked positions are excluded. Record deviations from the paper's diagnostic sample size or pair selection. A zero cosine caused by construction is not evidence of better language modeling.

Implement warm-up and synchronized timing for forward/backward throughput and peak memory on identical hardware, shapes, precision, and attention backends. Label hardware differences from the paper. Disable diagnostic collection during timing. Distinguish compilation time from steady-state runtime.

Provide downstream-evaluation scripts using a pinned LM Evaluation Harness version. Extract task IDs, accuracy versus normalized-accuracy metrics, prompting, and few-shot settings where reported. Mark unknown details as assumptions; do not claim exact Table 2 replication when the protocol is incomplete.

Use paired seed summaries and evaluation-sample uncertainty where appropriate. Do not substitute token-level confidence intervals for training-seed variability. Preserve negative and inconclusive results.

### 6. Deliver and run the bounded setup

Provide a concise, documented layout with:

- The shared SA/XSA implementation and pinned environment.
- Smoke, pilot, and paper-scale configurations.
- Data preparation and validation commands.
- Tests and their actual results.
- Slurm single-GPU launchers, bounded profiling jobs, and restart/resume scripts; distributed launchers only where needed.
- Evaluation, diagnostic, plotting, and benchmark scripts.
- `REPRODUCTION_SPEC.md`, a run manifest, and a README containing exact commands.

Run the tests and bounded smoke/profile checks now. Ensure plotting and checkpoint loading work on those outputs. Clearly label smoke artifacts so they cannot be mistaken for scientific results.

Estimate pilot and full-run storage and GPU-hours using measured throughput where available, stating extrapolation assumptions. Do not present throughput measured on a tiny model as a measurement for a billion-parameter model.

Finish with: what was implemented; what actually ran; pass/fail/skip results; unresolved deviations from the paper; and exact commands and resource estimates for the next SA/XSA pair. At this point, request only the compute or storage decisions needed to launch the longer experiments. Do not claim that the paper's results have been reproduced merely because the code runs.

---

## Part 2 — Research proposal: testing alternatives to exclusive self attention

*Research proposal draft*

We will test whether attention similarity bias can be reduced while preserving useful contextual information. We will compare three small modifications to attention: limiting excessive alignment, penalizing similar values before softmax, and measuring orthogonality after the output projection. Each method will be evaluated independently against standard attention and Exclusive Self Attention (XSA).

XSA removes the component of each attention output parallel to the current token’s value vector [1]. Its results motivate this study, but geometric alignment alone does not establish that a component is redundant.

$$y_{i} = \sum_{j \leq i}^{}a_{ij}v_{j},\quad\quad z_{i}^{XSA} = y_{i} - \frac{y_{i}^{\top}v_{i}}{\Vert v_{i} \Vert^{2}}v_{i}$$

### Proposed methods and hypotheses

#### A Limit excessive alignment

We will retain the parallel component until a fixed cosine-similarity limit is reached. The correction adapts to each token and head; the initial threshold will be shared across the model.

$$\begin{matrix}
u_{i} = v_{i}/ \Vert v_{i} \Vert ,\quad c_{i} = y_{i}^{\top}u_{i},\quad r_{i} = y_{i} - c_{i}u_{i},\quad b_{i} = \frac{\tau}{\sqrt{1 - \tau^{2}}} \Vert r_{i} \Vert \\
z_{i} = r_{i} + \operatorname{clip}\left( c_{i}, - b_{i},b_{i} \right)u_{i}
\end{matrix}$$

For nonzero outputs, the absolute cosine similarity with the self-value is at most τ. At τ = 0 the method reduces to XSA. We will screen τ ∈ {0.25, 0.50, 0.75}. Our hypothesis is that allowing moderate alignment preserves useful copying or contextual reinforcement without permitting strong self-alignment.

#### B Penalize similar values before softmax

We will subtract a positive value-similarity penalty from each causal attention score, then aggregate the original values using the new weights.

$$s^{\prime}_{ij} = \frac{q_{i}^{\top}k_{j}}{\sqrt{d_{h}}} - \lambda\max\left( 0,\cos\left( v_{i},v_{j} \right) \right),\quad\quad z_{i} = \sum_{j \leq i}^{}\operatorname{softmax}_{j}\left( s^{\prime}_{ij} \right)v_{j}$$

We will screen λ ∈ {0.1, 0.5, 1.0}; λ = 0 recovers standard attention. Our hypothesis is that selecting less redundant context can improve context use before information is discarded. This method encourages lower similarity but does not enforce orthogonality. It may also suppress useful similar tokens.

#### C Project using the output geometry

For each head, we will use its output-matrix block to define the geometry of projection. All vectors below belong to that head.

$$G_{h} = W_{O,h}^{\top}W_{O,h},\quad\quad z_{i} = y_{i} - \frac{v_{i}^{\top}G_{h}y_{i}}{v_{i}^{\top}G_{h}v_{i}}v_{i}$$

This makes the head’s written update orthogonal to its transformed self-value whenever the denominator is nonzero. We hypothesize that this geometry better identifies redundant updates than angles inside the head. The guarantee is per head, not relative to the entire residual stream.

### Implementation and correctness

We will use a pinned nanoGPT-based PyTorch implementation [2] with RoPE, full multi-head causal attention, tied embeddings, and a common feed-forward block. A configuration switch will select each variant. A and C modify the aggregate before the output projection; B modifies logits before softmax. None adds trainable parameters.

For C, we will recompute each head’s Gram matrix on every forward pass, with gradients through the metric. For B, explicit attention will provide a correctness reference for a memory-efficient or query-chunked implementation. Its pairwise value comparisons add computation; throughput measurements will capture this cost.

Norms and projection reductions will use float32 with near-zero guards. We will check zero values, collinear vectors, and nearly singular metrics. Zero outputs have undefined cosine similarity and will be recorded separately. We will retain each method’s native first-position behavior and report its effects separately.

#### Tests before training

| **Test**               | **Required result**                                                                                                             |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Special cases          | A at τ = 0 matches XSA; B at λ = 0 matches standard attention; C with an identity metric matches XSA.                           |
| Geometry               | A satisfies its cosine limit on nonzero outputs; C satisfies orthogonality after the output projection on nondegenerate inputs. |
| Causality and caching  | Changing future tokens leaves earlier outputs unchanged. Cached decoding matches full-sequence evaluation.                      |
| Numerics and gradients | Reference and optimized outputs agree. Gradients are finite; finite-difference checks pass away from clipping boundaries.       |

### Training design and proposed budget

We will train from scratch on a fixed FineWeb sample [3] with one tokenizer and document-level splits made before packing. Identical documents will not cross splits. Validation will guide configuration selection; test data will remain untouched until configurations and analysis are frozen.

| **Stage**          | **Model and context**               | **Runs and token budget**                                             |
|--------------------|-------------------------------------|-----------------------------------------------------------------------|
| Screening          | About 30M parameters; 512 tokens    | 9 configurations × 1 seed × 50M training tokens                       |
| Main comparison    | About 124M parameters; 1,024 tokens | 5 configurations × 3 seeds × 500M training tokens                     |
| Optional extension | Same model; 2,048 tokens            | Standard attention, XSA, and best variant; budget set after profiling |

The nine screening runs cover standard attention, XSA, C, three A thresholds, and three B penalties. One setting per method advances. These exploratory rankings require main-study confirmation. Core runs total 7.95B training tokens; controls and synthetic tasks need additional budget.

The pilot will fix a common AdamW recipe, schedule, batch size, and precision. Paired seeds share initialization and data order; further tuning allowances will be equal. Measured throughput will determine GPU-hour estimates and any uniform budget reduction before the main runs.

### Evaluation and mechanism analysis

#### Language modeling and linguistic behavior

Our primary outcome will be held-out next-token cross-entropy at the fixed training budget, with perplexity reported for readability. We will also plot validation loss against both training tokens and elapsed time. BLiMP minimal-pair accuracy [4] will provide a secondary check that changes preserve grammatical preferences; it will not serve as a direct measure of contextual reasoning.

#### Controlled context tasks

We will train small versions of all five architectures on the same procedurally generated task mixture. This separates the ability to learn a controlled behavior from whether a small web-trained model understands an instruction. Train and test sets will use disjoint random sequences and key–value assignments. Each condition will contain at least 1,000 held-out examples.

| **Task**               | **Manipulation**                                                     | **Measurement**                                           |
|------------------------|----------------------------------------------------------------------|-----------------------------------------------------------|
| Associative retrieval  | Vary number of distractors and distance to a queried key–value pair. | Answer-token accuracy and accuracy versus context length. |
| Copying and repetition | Require recovery of an earlier token span; vary helpful repetition.  | Exact span match and token accuracy.                      |
| Updating information   | Repeat a key with a new value; require the most recent assignment.   | Correct updated value versus obsolete value.              |

We will vary helpful repetition and irrelevant distractors independently. Repeated tokens are an experimental manipulation, not a guarantee of similar internal values. We will measure actual value similarities to check whether each task produces the intended geometry. Longer-than-training contexts will be reported as extrapolation tests, separately from performance within the training range.

#### Diagnostics and computational cost

On a fixed diagnostic sample, we will record signed cosine similarity and its distribution by layer and head, the fraction of high-alignment outputs, attention entropy, self-attention weight, and update norms. We will measure alignment before and after correction and after the output projection. For A, we will track how often the cap activates; for B, how attention allocation changes; for C, the magnitude of the written update and metric degeneracies.

We will benchmark forward-plus-backward throughput, peak GPU memory, and cached decoding on the same hardware and batch shapes after warm-up. Explicit-attention timings will be labeled separately from optimized timings. We will report actual compute costs and matched-time learning curves alongside matched-token comparisons.

#### Targeted controls

To distinguish A from ordinary partial removal, we will compare it with a fixed-strength XSA baseline selected using the same small validation search. To separate direction from magnitude, we will compare full XSA with a norm-restored projection and a control that only shrinks the original output. We will cap unstable rescaling and report how often this occurs. These controls will first run at pilot scale; a control that explains a main result must be repeated at the main scale before supporting a mechanistic claim.

### Analysis and criteria for success

We will compare each proposed method with both standard attention and XSA using paired seed differences. The report will show all three seed results, their mean and standard deviation, and document-level paired bootstrap intervals for loss differences. Task accuracy intervals will use paired examples. Example-level intervals measure evaluation uncertainty and will not be presented as a substitute for training-seed variability.

A promising method should improve test loss or a predefined context task consistently across seeds, with its cost stated explicitly. Lower cosine similarity alone will not count as success. We will report copying, retrieval, and updating separately. Broader claims will require a second model size or context setting.

Negative results can establish where alignment remains useful or where a correction fails to justify its cost. We will distinguish these observations from claims that the removed component is semantically redundant.

### Milestones and deliverables

| **Stage**     | **Deliverable**                                                                                                                                  |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| Week 1        | Implement the three variants, numerical tests, diagnostic logging, and document-level data splits. Profile runtime and memory.                   |
| Week 2        | Run the small-model screen, validate synthetic tasks, and freeze main-study settings and evaluation criteria.                                    |
| Weeks 3 and 4 | Run the five-condition, three-seed comparison and small-model context experiments. Add only the controls needed to explain observed differences. |
| Week 5        | Evaluate the frozen test sets, quantify uncertainty and cost, and write the report. Run a longer-context confirmation only if resources permit.  |

The schedule depends on GPU throughput and queue availability. Deliverables will include reproducible configurations, implementations and tests, logs, checkpoints, diagnostic plots, and a report explaining when each intervention helps. We will evaluate combinations only after understanding the individual methods.

### Expected contribution and related work

We aim to determine how much self-alignment is useful, whether similarity should be addressed during selection or after aggregation, and which representation space provides the most useful projection geometry. Partial-projection experiments already exist [5]; our contribution must therefore come from controlled evidence about these distinct mechanisms. A focused related-work review will precede any claim that a particular modification is new.

### References

[1] Zhai, S. (2026). Exclusive Self Attention. arXiv:2603.09078. <https://arxiv.org/abs/2603.09078>

[2] Karpathy, A. nanoGPT, model implementation. <https://github.com/karpathy/nanoGPT/blob/master/model.py>

[3] Hugging Face. FineWeb dataset and documentation. <https://huggingface.co/datasets/HuggingFaceFW/fineweb>

[4] Warstadt, A., et al. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English. TACL. <https://aclanthology.org/2020.tacl-1.25/>

[5] turancannb02. XSA interactive research notebook and partial-projection experiments. Community reproduction, not peer-reviewed evidence. <https://github.com/turancannb02/xsa-marimo-notebook>

---

## Part 3 — Reproduction resources and cluster documentation

The paper defines the target experiment. Community code provides implementation examples and related experiments; it is not evidence that the original results have been independently verified. The repositories below were inspected through documentation and selected code paths, not executed here.

| Resource | Role |
|---|---|
| [XSA paper](https://arxiv.org/abs/2603.09078) and [full text](https://arxiv.org/html/2603.09078v1) | Primary method, architecture, training, and evaluation specification. |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | Base code named in the paper; requires the paper's reported modifications. |
| [turancannb02/xsa-marimo-notebook](https://github.com/turancannb02/xsa-marimo-notebook) | Small TinyStories training comparisons, alignment diagnostics, and partial-projection experiments. |
| [Slowrun XSA commit](https://github.com/qlabs-eng/slowrun/commit/ac968a62c633d75d972afa6d86a59f89e12997b9) | Isolated XSA addition to an existing training script; a different architecture/training regime from the paper. |
| [Shwai-He/Transformer-Geometry](https://github.com/Shwai-He/Transformer-Geometry) | Related value-space/residual-space interventions and diagnostics; review for overlap before claiming novelty. |
| [jibran/attnres](https://github.com/jibran/attnres) | An XSA integration in an Attention Residuals implementation, not a matched reproduction baseline. |
| [harsh-raj-singh/xsa_marimo_competition](https://github.com/harsh-raj-singh/xsa_marimo_competition) | Small character-level training and mechanism walkthrough. |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | Dataset documentation; pin the exact subset and revision. |

### Known issues to check before reusing community code

- The inspected Marimo model enforces `d_model == n_heads * head_dim`, which cannot represent all of the paper's configurations. Fix and test independent attention dimensions.
- Its TinyStories loader can fall back to synthetic data. Research runs must fail clearly on unavailable data instead of silently changing the experiment.
- Partial projection has already been explored publicly. Distinguish your adaptive cosine cap from a fixed projection-strength multiplier, and review related work on projection spaces.
- Geometric orthogonality is a mathematical property of the intervention; improved prediction or context use must be demonstrated separately.

### Insomnia sources

- [Columbia HPC access options](https://www.cuit.columbia.edu/shared-research-computing-facility)
- [Insomnia getting started](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145140)
- [Insomnia technical information](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145136)
- [Current queue system](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/1169162241)
- [Insomnia submitting jobs](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145134)
- [Free-tier technical information](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62144315)

Verify current limits and your account permissions before submission. Hardware inventory does not guarantee access or immediate availability. If using education access, confirm the project falls within the course allocation.

### Budget context

The earlier $1,500–$2,000 Modal estimate was a conservative planning allowance for the full small-model project, including reruns; it was not a measured bill or a required expenditure. The Insomnia plan instead depends on an eligible existing university allocation and available GPU-hours. Full original-paper-scale reproduction is a separate, much larger undertaking.

If cloud resources become useful later, consult [Modal pricing](https://modal.com/pricing) and [academic grants](https://modal.com/academics). Do not launch paid jobs by default.

### Evidence status

This handoff contains a research plan and implementation instructions. No model training, cluster access verification, or reproduction experiments have been performed as part of preparing it.

