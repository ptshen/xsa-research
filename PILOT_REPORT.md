# Exploratory SA/XSA pilot results

Both arms completed 2048 updates / 67,108,864 training tokens on NVIDIA RTX A6000, with identical saved initialization and all 2048 ordered minibatch hashes. Actual allocated GPU time: **1287 seconds**, within the approved 3600 seconds. Project storage at collection: 4.430 GiB, within 5 GiB.

| Method | Held-out cross-entropy | Perplexity |
|---|---:|---:|
| SA | 4.89046994 | 133.01607 |
| XSA | 4.80333272 | 121.91605 |

XSA minus SA loss: **-0.08713721**. Paired evaluation-window bootstrap 95% interval: [-0.09983557593077422, -0.07541421875357629]. This is evaluation-sample uncertainty with an independent-window approximation; shared-document correlations may make it optimistic. There is one training seed, so no training-seed uncertainty estimate is available.

The pinned FineWeb prefix contains 99,962,616 training tokens from 146,329 documents and 36,418 validation tokens from 74 documents. Evaluation uses the same 64 nonoverlapping 512-token target windows (32,768 evaluated tokens). Tokenizer: GPT-2; document-preserving split seed 2026. This is a scaled-down prefix-data study, not paper reproduction. Downstream tasks and further seeds remain unrun.

[Learning curves](evidence/pilot/curves.png), [paired results](evidence/pilot/paired_results.json), [batch identity](evidence/pilot/pair_identity.json), [scheduler accounting](evidence/pilot/accounting.psv), and [manifest](PILOT_MANIFEST.json) contain the evidence. Diagnostics and all learning-curve rows are included in each arm's artifact directory. Checkpoints and prior checkpoints remain under `~/algollms/xsa/runs/pilot-{sa,xsa}-s17/`.

The first HTTP preparation attempt was cancelled for insufficient throughput; the parallel-download retry completed successfully. Its original log and data directory were retained. Only its abandoned partial-download cache was removed after the full pinned source was verified present.

No further training is automatically launched. Inspect these exploratory results before selecting any additional seeds, LR search, or extension experiments.
