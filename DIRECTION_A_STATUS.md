# Direction A screen in progress

Implementation commit: `57aa6e2`. GPU correctness/profile job **14072510 completed in 38 s**, with **49 passing tests and zero skips**. The preserved SA/XSA evaluations and initial state passed rechecks.

| Threshold | Job | State |
|---|---|---|
| 0.25 | 14072518 | Last observed running; SSH currently unreachable |
| 0.50 | Not submitted | Waiting for preceding run |
| 0.75 | Not submitted | Waiting for preceding run |

The stage ledger is `evidence/direction_a/job_ledger.json`. Inspect those IDs before submitting anything again. Existing seed-17 baselines are reused. No other research direction, extra seed or downstream benchmark is launched.

Next action: verify full-budget completion, evaluation and diagnostics for 0.25, then submit 0.50 within the same cumulative budget.
