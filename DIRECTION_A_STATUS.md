# Direction A screen in progress

Implementation commit: `57aa6e2`. GPU correctness/profile job **14072510 completed in 38 s**, with **49 passing tests and zero skips**. The preserved SA/XSA evaluations and initial state passed rechecks.

| Threshold | Job | State |
|---|---|---|
| 0.25 | 14072518 | COMPLETED; CE 4.82149275; 663 seconds |
| 0.50 | 14074753 | COMPLETED; CE 4.81451356; 690 seconds |
| 0.75 | 14074780 | Submitted; inspect scheduler for current state |

The stage ledger is `evidence/direction_a/job_ledger.json`. Inspect those IDs before submitting anything again. Existing seed-17 baselines are reused. No other research direction, extra seed or downstream benchmark is launched.

Next action: verify full completion/evaluation/diagnostics for job 14074780, then run the bounded CPU collector and independent final audit. No additional training is planned.
