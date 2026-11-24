# Directions B/C completion audit — in progress

The goal is not complete. Source and local numerical verification are implemented; GPU verification and all four screening runs remain unverified at this snapshot.

| Requirement | Required authoritative evidence | Current status |
|---|---|---|
| Read objective/current repository and cluster state | Preserved objective, Git equality, queue/quota/accounting | Inspected; recorded in PROGRESS |
| Correct B penalty including diagonal, gradients and causal chunks | Independent per-query reference, finite differences, causality and lambda-zero tests | Local passed; CUDA pending |
| Correct C head-block geometry and differentiable metric | Written-space reference, head-width mismatch, gradients, guard/degeneracy tests | Local passed; CUDA pending |
| Unchanged SA/XSA/A and no new parameters | Frozen-source output/gradient tests, parameter/initialization hashes | Local passed; real checkpoints pending |
| Deterministic resume and setting rejection | CPU/CUDA model/optimizer/data comparisons and strict config checks | CPU passed; CUDA pending |
| All required CUDA tests and actual preserved checkpoint re-evaluation | JUnit with zero failures/skips; five per-window recheck records | Job 14076373 pending/running |
| Measured profile and frozen complete-screen resource estimate | Nine-method timing/memory/kernel trace and DIRECTIONS_BC_PLAN.json | Pending |
| Four complete matched screening runs | Four final 2048-update/67,108,864-token checkpoints and evaluations | Not launched |
| Combined nine-method evaluation | Exact config/data/init/window/software identity; all batch hashes; summary and curves | Pending |
| Method-specific diagnostics | Actual B weights/penalties; C written geometry and denominator/guard summaries | Implemented; final samples pending |
| Resumable latest/previous checkpoint pairs | Independent loading, finite model/optimizer tensors and identity audit | Pending |
| Budgets and preservation | All job IDs, starts/ends/TRES, reservations, quota/storage; original ledger and checkpoint checksums | Stage reservation 300 GPU seconds; final audit pending |
| Actual commands, failures, results, limits and next-stage estimate | Final report, manifests, PROGRESS and raw evidence | Pending |
| GitHub publication | Published source and final evidence commits verified against origin and Insomnia | Implementation 1db7eff published; final evidence pending |
