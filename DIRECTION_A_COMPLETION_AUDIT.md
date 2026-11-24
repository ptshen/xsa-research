# Direction A requirement audit

This audits the complete goal attachment and its scoped Direction A requirements. The experiment is complete; no inference of a general quality improvement is made. Verification uses the actual allocated jobs, checkpoints, collected metrics, source, and numerical tests.

| Requirement | Evidence inspected | Result |
|---|---|---|
| Exact capped-alignment formula before concatenation/output projection | `xsa/model.py::cap_alignment`, `correct_aggregate`, `Attention.forward`; direct fp64 reference tests | PASS |
| Fixed model-wide 0 ≤ τ < 1, no new parameters, preserved SDPA | ModelConfig validation; three configs; real run manifests and checkpoint parameter counts; CUDA profile | PASS |
| τ=0 equals XSA, including gradients | `test_tau_zero_matches_xsa_forward_and_backward`; original `project` operation retained | PASS |
| Bounds, inactive outputs, signs, zero/near-zero/collinear/orthogonal inputs and invalid τ | `test_cap_geometry_reference_and_gradients`, `test_inactive_signed_and_degenerate_cases`, `test_invalid_thresholds`; documented epsilon/cancellation limits | PASS |
| Causality, reference/SDPA forward and backward, finite differences away from clipping boundaries | `test_reference_sdpa_first_position_and_causality`, parameterized gradchecks | PASS |
| fp32 reductions under bf16 with explicit output-quantization tolerance | `test_cuda_cap_precision` for both dtypes; code preserves fp64 references and casts final output | PASS |
| Unchanged SA/XSA behavior and counts; existing initialization/checkpoint loading | Frozen baseline source regression; actual baseline re-evaluation difference 0.0; all three initial states hash-identical | PASS |
| CPU/CUDA resume, optimizer/data state and configuration rejection | `test_cap_resume_and_reject_changed_tau`; baseline resume/RNG tests; unconditional complete-config equality in `Trainer.restore` also rejects other changed settings | PASS |
| Numerical correctness before training | Verification 14072510: 49 passed, no skips; scheduler end precedes every cap job; numerical/config hashes unchanged afterward | PASS |
| Matching architecture, optimizer/LR/precision/schedules, data and windows | Full checkpoint comparison hashes plus independent raw-manifest config comparison allowing only attention, τ and descriptive label; all software records identical | PASS |
| Three full seed-17 runs with original shared initialization and ordered batches | Jobs 14072518/14074753/14074780; every method has 2048 consecutive updates and 67,108,864 tokens; all 2048 batch hashes/final data states match | PASS |
| No baseline retraining or other direction/seed/downstream/matrix launch | Stage ledger and final accounting contain only one verification job, three authorized cap runs and one CPU collector | PASS |
| GPU/CPU limits, sequential allocations and actual A6000 resource | 2049 GPU seconds, 38 verification/profile seconds, 36 CPU-only seconds; 5580/180/300 respective reservations; actual TRES and nonoverlapping start/end times | PASS |
| Storage including temporary-checkpoint headroom; existing resources only | Before-launch du/quota checks, 7.06 GiB conservative transient estimate, 6.701 GiB at collection; edu/short; no additional download or paid resource | PASS |
| Duplicate prevention, existing job inspection, no restart after observation timeout | Guard tests, pre-submission scheduler snapshots, ledger reconciliation; 14072518 continued through lost SSH and was reused after successful completion | PASS |
| Reproducible configuration, data and threshold metadata | Three versioned configs, source hashes, pinned existing data manifest/checksums; thresholds in manifests/checkpoints/metrics/evaluations/diagnostics | PASS |
| Final loss/perplexity and every threshold against both baselines | Independent recomputation of all 64-window means/perplexities and complete summary, including negative comparisons to XSA | PASS |
| Learning curves versus tokens and elapsed time | Full plot and explicitly labeled second-half zoom; both visually inspected | PASS |
| Throughput and peak memory | Five-method matched profile; actual allocation times and loop throughput; actual cap-training peaks; baseline training peaks explicitly unrecorded | PASS |
| Same-sample layer/head diagnostics | All five diagnostic files: 4 samples, context 128, seed 7017, 6 layers × 6 heads; cap activations, signed alignments, norms, zero/near-zero fractions checked | PASS |
| Full checkpoint loading, finite values and collection | CPU collector 14074923 loaded all five final model/checkpoint states, checked weights/identities and latest/previous file checksums; exit 0 | PASS |
| Scope and uncertainty retained | Validation selection, one seed, small prefix/model/window sample, no untouched test claim; controls/fresh seeds remain unlaunched | PASS |
| Logs/checkpoints preserved and recovery documented | Remote paths/SHAs, source snapshot, red/green evidence, complete Slurm logs, PROGRESS chronology, backed-up ledger and checkpoint pairs | PASS |

The numerical suite's [JUnit record](evidence/direction_a/tests.xml), [CUDA output](evidence/direction_a/tests.txt), [gate proof](evidence/direction_a/verification.json), [final scheduler accounting](evidence/direction_a/final_accounting.psv), [checkpoint identities](evidence/direction_a/results/identity.json), and [independent artifact audit](evidence/direction_a/final_audit.json) are the supporting evidence. Three final collector/budget tests additionally passed locally; two overlap the CUDA suite.

The original Part 1 smoke ledger remains unchanged at 49 actual / 480 reserved seconds. The new stage has its own explicit authorization and ledger. Source data, logs and latest/previous checkpoints remain under `~/algollms/xsa`; large binary files are not placed in Git.

Final publication is checked against GitHub `main` after committing the report and compact artifacts; delivery confirmation is recorded in PROGRESS.md. There is no pending experiment after this stage.
