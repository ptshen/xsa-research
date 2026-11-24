# Matched nine-configuration profile

Verification job 14076777; RTX A6000 on ins083; 29.94M parameters, batch 8, context 512, bf16 with fp32 operator reductions, no compilation. Three warmup microsteps, twelve measured forward/backward microsteps, three accumulation-8 updates, and an extra untimed dispatch-identification step.

| Method | Forward/backward microstep (ms) | Full update (ms) | Update overhead vs XSA | Profile peak allocated (GiB) | Observed attention path |
|---|---:|---:|---:|---:|---|
| SA | 31.735 | 245.166 | -11.38% | 3.872 | Flash SDPA |
| XSA | 36.330 | 276.660 | +0.00% | 4.205 | Flash SDPA |
| A τ=0.25 | 38.288 | 290.919 | +5.15% | 4.246 | Flash SDPA |
| A τ=0.5 | 38.529 | 291.168 | +5.24% | 4.244 | Flash SDPA |
| A τ=0.75 | 38.301 | 291.305 | +5.29% | 4.246 | Flash SDPA |
| B λ=0.1 | 64.480 | 518.353 | +87.36% | 4.987 | Explicit fp32 softmax/bmm |
| B λ=0.5 | 64.502 | 518.437 | +87.39% | 4.982 | Explicit fp32 softmax/bmm |
| B λ=1.0 | 64.516 | 517.717 | +87.13% | 4.980 | Explicit fp32 softmax/bmm |
| C | 38.526 | 286.987 | +3.73% | 4.247 | Flash SDPA + Gram bmm |

Timing excludes data I/O, evaluation, checkpoint writes and diagnostics. Peak memory covers the profile process, including optimizer state and the untimed dispatch probe; it is separate from peak memory recorded during full training. B’s cost includes switching from flash SDPA to explicit fp32 attention. Query chunking is not evidence of lower total backward memory. These are short within-allocation measurements, not a repeated end-to-end speed study.
