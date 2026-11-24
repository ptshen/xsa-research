# Directions B/C screening — in progress

The goal is incomplete. Both operators are implemented, the original 87-test CUDA gate passed, and B lambda 0.1 completed as job **14076378**, reaching 2048 updates / 67,108,864 tokens in 1118 allocated GPU seconds. Its validation cross-entropy is **4.8848455101** (SA 4.89046994; XSA 4.80333272). Additional numerical review exposed and locally fixed a C near-null denominator defect; the hardened source still requires its allocated GPU gate before further training. No C model has been trained. See PROGRESS.md for the latest job observations.

## Implemented methods

B subtracts a positive value-cosine penalty from causal scaled QK logits, includes the diagonal, and aggregates unmodified values. Values receive gradients through both the penalty and aggregation. Its training path uses explicit fp32 attention with 128-query chunks, with output cast to bf16. Query chunking bounds individual temporary matrices; retained backward tensors still consume memory. This implementation's overhead includes replacing flash SDPA with explicit attention.

C computes each head's metric from its current output-projection column block, then removes the component along its value using that metric. The Gram matrix remains differentiable and is recomputed every forward. Correction happens before head concatenation. A dimension/dtype/scale-aware guard skips numerically unresolved denominators; that engineering choice and its geometric limitations are specified in [the protocol](DIRECTIONS_BC_PROTOCOL.md). No trainable parameters are added. SA/XSA/A behavior is preserved.

All runs retain the seed-17, 29,938,944-parameter pilot, six layers/384 width/six 64-dimensional heads, context 512, the existing FineWeb prefix and initialization, and exactly 2048 updates/67,108,864 tokens. Optimizer, schedule, batch order and validation windows are matched. No additional data was downloaded.

## Verification and numerical failure

- Original allocated gate **14076373 COMPLETED 59 s**, **87 passed / zero failures or skips**, 13.55 s on A6000 ins093.
- All five existing models re-evaluated exactly: maximum per-window loss difference **0.0**. Initialization and ordered baseline batch histories match.
- Subsequent fp32 rank-one stress found a Gram denominator corrupted by cancellation: computed written-value norm zero, but corrected head-space norm 313.35 from an aggregate norm 0.7993. A failing test is preserved in `evidence/direction_bc/geometry_guard_red.txt`.
- C now requires its denominator to exceed both the absolute epsilon-squared floor and a dtype/dimension/value/weight-scaled roundoff threshold. The inactive branch uses a safe denominator before division. No ridge is added. B's implementation is unchanged, verified by AST equality and exact forward/gradient regression against its original source.
- The first gate/profile/plan is archived under `evidence/direction_bc/verification_v1/`. The second allocated gate and all remaining runs are pending. Current local suite: **79 passed / 13 CUDA-only skipped**.

The same initialization file and every latest/previous checkpoint are preserved. The independent collector will load both checkpoint generations and check finite model/optimizer state, complete batch histories, data/config/software identities and independently recomputed results.

## Evaluation and scope

Primary metric is held-out token-weighted next-token cross-entropy in nats/token, on the original 64 nonoverlapping windows/32,768 target tokens. Perplexity is exp(cross-entropy), the same evidence on a different scale. It is not downstream accuracy.

The final nine-configuration table, paired differences, curves, diagnostics and measured full-run costs will be added after completion. Selection uses one seed and this validation set. Window bootstrap intervals assume independent windows and do not quantify seed variability or correct selection bias. There is no untouched test, downstream evaluation, synthetic context task or causal mechanism control. KV caching remains unimplemented.

Separate stage limits are 14400 GPU seconds, including at most 600 verification/profile seconds, 1800 CPU-only allocated seconds, and 12 GiB total project storage including temporary checkpoint headroom. Historical budgets are unchanged. No further research stage is authorized or scheduled. Final resource accounting and a measured next-stage estimate are pending completion of the four runs.
