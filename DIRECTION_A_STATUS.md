# Direction A screen complete

All three thresholds completed 2048 updates / 67,108,864 tokens and passed the full checkpoint/pairing audit. No tested cap improved on XSA in this one-seed validation screen.

| Work | Job | Final state | Allocated time |
|---|---|---|---:|
| CUDA verification/profile | 14072510 | COMPLETED | 38 s GPU |
| τ=0.25 | 14072518 | COMPLETED | 663 s GPU |
| τ=0.50 | 14074753 | COMPLETED | 690 s GPU |
| τ=0.75 | 14074780 | COMPLETED | 658 s GPU |
| Checkpoint validation/collection | 14074923 | COMPLETED | 36 s CPU |

All exit codes are 0:0. Stage GPU allocation: **2049 seconds**, within 7200. Storage at collection: **6.701 GiB**, within 8 GiB. Logs and latest/previous checkpoints remain on Insomnia under `~/algollms/xsa`.

See [DIRECTION_A_REPORT.md](DIRECTION_A_REPORT.md), [DIRECTION_A_MANIFEST.json](DIRECTION_A_MANIFEST.json), and [the completion audit](DIRECTION_A_COMPLETION_AUDIT.md).

Next action: review the exploratory findings before choosing any separately authorized control or fresh-seed study. No further experiment is scheduled.
