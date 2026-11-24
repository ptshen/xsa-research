# Direction A implementation and execution protocol

The `alignment_cap` method limits the magnitude of cosine alignment between a token's attention aggregate and its own value. Each head computes the original causal SDPA aggregate, splits it into perpendicular and parallel components, clips the parallel coefficient using the model-wide `alignment_tau`, then concatenates heads and applies the existing output projection. No trainable parameters are added.

## Numerical conventions

- Thresholds satisfy `0 <= tau < 1`; missing, nonfinite and out-of-range values fail validation. SA/XSA configurations reject an irrelevant threshold.
- Tau zero calls the original XSA projection exactly, including its gradients and clamped normalization. Existing SA/XSA formulas are unchanged; frozen-source forward/gradient regression tests pass.
- Reductions use fp32 for fp16/bf16 inputs and preserve fp64 for references. The output is cast back to the input dtype. The real-arithmetic cosine bound has dtype-dependent tolerance after that cast and becomes ill-conditioned for nearly zero outputs.
- Normalization uses the existing `projection_eps=1e-12`. Zero values leave the aggregate unchanged. Below epsilon the normalized vector is not unit length, so the strict geometric bound is not asserted. This preserves the baseline's tau-zero behavior.
- Collinear aggregates, including the first causal position, follow the formula: their perpendicular component is zero in exact arithmetic, so their output is zero for every tau below one. Floating-point cancellation may leave a small residual. Tau approaching one does not guarantee SA behavior for these cases.
- Diagnostic cosine averages exclude output/value norms at or below 1e-6; exact-zero and near-zero fractions are reported separately. Activation rates include all positions and record `abs(c) > b` using fp32 intermediates.

## Fixed experimental settings

Configs `direction_a_025.json`, `direction_a_050.json`, and `direction_a_075.json` differ from `pilot.json` only in their label and intended attention method/threshold. The model has 29,938,944 total / 10,622,208 nonembedding parameters, six layers, width 384, six 64-dimensional heads, context 512, SDPA and bf16 with fp32 correction.

Each run loads `runs/initial/pilot-s17.pt`, uses `data/pilot-v2`, seed 17/data seed 1017/evaluation seed 2017, and performs 2048 updates of 32,768 tokens. LR, optimizer, accumulation, evaluation and checkpoint schedules remain identical. The 64 validation windows and all ordered training batch hashes must match the original pilot. Existing baseline evaluations remain untouched; the verification job wrote separate re-evaluations.

Thresholds are saved in configs, manifests, metrics and checkpoints. Resume requires identical settings; changing tau is rejected. Initialization matching excludes the threshold because it changes no tensor shapes. Comparison hashes allow only the intended attention/threshold differences (plus the pre-existing descriptive-label exclusion); other protocol differences fail.

## Commands and resource accounting

All remote commands run under `~/algollms/xsa`. Read the ledger and current jobs before submission:

```bash
ssh insomnia
cd ~/algollms/xsa
cat evidence/direction_a/job_ledger.json
squeue --me
sacct -X -j 14072510,14072518 -o JobID,State,Elapsed,ExitCode
```

The executed verification command was:

```bash
/usr/bin/python3 -m xsa.stage_submission verify --key direction-a-verify-v1 --seconds 180
```

It ran the complete CUDA test suite, reloaded/re-evaluated both preserved baselines, checked initialization compatibility, and profiled SA, XSA and all three caps with the same A6000, SDPA, bf16, batch 8, context 512, three warmup microsteps, twelve measured microsteps and three accumulation-8 updates. The optimizer benchmark is a common timing harness; actual training uses the unchanged pilot optimizer groups. Timing excludes data I/O, validation, checkpoint writes and diagnostics.

Training is sequential; these commands are individually submitted only after checking the preceding job and complete final artifacts:

```bash
/usr/bin/python3 -m xsa.stage_submission train --tag 025 --key direction-a-tau025-v1 --seconds 1800
/usr/bin/python3 -m xsa.stage_submission train --tag 050 --key direction-a-tau050-v1 --seconds 1800
/usr/bin/python3 -m xsa.stage_submission train --tag 075 --key direction-a-tau075-v1 --seconds 1800
# After all three are complete:
/usr/bin/python3 -m xsa.stage_submission collect --key direction-a-collect-v1 --seconds 300
```

The launcher checks prior IDs, refuses duplicate keys/active project jobs/uncertain submissions, and reserves full requested walltime before submitting. Separate stage caps are 7200 GPU seconds total, 300 verification/profile GPU seconds within that, 900 CPU-only seconds and 8 GiB project storage. The three initial training caps plus verification reserve 5580 GPU seconds. No automatic requeue or retry occurs. The original Part 1 smoke accounting is unchanged.

For an interrupted run, inspect its job and checkpoint, then use the same tag with a new submission key and `--resume`; the continuation must fit the remaining cumulative reservation. Never use a new threshold to resume an existing directory. Logs and latest/previous checkpoints remain preserved.

## Analysis limits

All thresholds use the existing validation set for selection. They are not independent test results. The collector reports every threshold against both baselines, checks full checkpoint/config/data identities and committed batch histories, and records runtime and memory. Validation-window bootstrap intervals assume independent windows and do not quantify training-seed variability.

Diagnostics use the same four 128-token training windows as the original pilot; they measure activation, geometry and update magnitudes, not language-model quality. A fixed-strength partial-removal control with equal tuning allowance is needed before claiming that adaptive clipping itself helps. Fresh-seed confirmation and untouched evaluation data are separate future work; none is launched by this stage.
