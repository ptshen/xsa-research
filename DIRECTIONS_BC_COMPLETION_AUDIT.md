# Directions B/C completion audit — in progress

The goal is not complete. Source and GPU numerical verification are complete. B 0.1 finished; B 0.5 is job 14076783; B 1.0, C and final collection remain.

| Requirement | Required authoritative evidence | Current status |
|---|---|---|
| Read objective/current repository and cluster state | Preserved objective, Git equality, queue/quota/accounting | Inspected; recorded in PROGRESS |
| Correct B penalty including diagonal, gradients and causal chunks | Independent per-query reference, finite differences, causality and lambda-zero tests | CPU/CUDA passed, final gate 14076777 |
| Correct C head-block geometry and differentiable metric | Written-space reference, head-width mismatch, gradients, guard/degeneracy tests | CPU/CUDA passed, final gate 14076777 |
| Unchanged SA/XSA/A and no new parameters | Frozen-source output/gradient tests, parameter/initialization hashes | Frozen-source tests pass; all five real checkpoints re-evaluate exactly |
| Deterministic resume and setting rejection | CPU/CUDA model/optimizer/data comparisons and strict config checks | CPU/CUDA passed |
| All required CUDA tests and actual preserved checkpoint re-evaluation | JUnit with zero failures/skips; five per-window recheck records | Final job 14076777: 92 passed, no skips; first gate archived |
| Measured profile and frozen complete-screen resource estimate | Nine-method timing/memory/kernel trace and DIRECTIONS_BC_PLAN.json | Final gate passed; complete screen fits |
| Four complete matched screening runs | Four final 2048-update/67,108,864-token checkpoints and evaluations | B 0.1 complete; B 0.5 submitted; two remain |
| Combined nine-method evaluation | Exact config/data/init/window/software identity; all batch hashes; summary and curves | Pending |
| Method-specific diagnostics | Actual B weights/penalties; C written geometry and denominator/guard summaries | Implemented; final samples pending |
| Resumable latest/previous checkpoint pairs | Independent loading, finite model/optimizer tensors and identity audit | Pending |
| Budgets and preservation | All job IDs, starts/ends/TRES, reservations, quota/storage; original ledger and checkpoint checksums | 4080 GPU seconds reserved; final audit pending |
| Actual commands, failures, results, limits and next-stage estimate | Final report, manifests, PROGRESS and raw evidence | Pending |
| GitHub publication | Published source and final evidence commits verified against origin and Insomnia | Implementation 1db7eff published; final evidence pending |
