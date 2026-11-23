"""Strict paired comparisons and window uncertainty, distinct from seed variation."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired_summary(records):
    if not records:
        raise ValueError('no evaluation records')
    for key in ('tokens', 'data_manifest_sha256', 'evaluation_windows_sha256', 'study_config_sha256', 'scientific_result'):
        if len({r[key] for r in records}) != 1:
            raise ValueError(f'unmatched {key} across seeds')
    seeds = sorted({r["seed"] for r in records})
    pairs = []
    for seed in seeds:
        by_kind = {r["attention"]: r for r in records if r["seed"] == seed}
        if set(by_kind) != {"sa", "xsa"} or sum(r["seed"] == seed for r in records) != 2:
            raise ValueError(f"seed {seed} needs exactly one SA and one XSA result")
        a,b = by_kind["sa"], by_kind["xsa"]
        for key in ("tokens", "data_manifest_sha256", "evaluation_windows_sha256", "initial_state_sha256", "comparison_config_sha256", "scientific_result"):
            if a[key] != b[key]:
                raise ValueError(f"unmatched {key} for seed {seed}")
        diff = np.asarray(b["window_losses"]) - np.asarray(a["window_losses"])
        boot = np.random.default_rng(2026).choice(diff, (2000, len(diff)), replace=True).mean(1)
        pairs.append({"seed": seed, "tokens": a["tokens"], "sa_cross_entropy": a["cross_entropy"],
                      "xsa_cross_entropy": b["cross_entropy"], "xsa_minus_sa": b["cross_entropy"]-a["cross_entropy"],
                      "window_bootstrap_95ci": np.quantile(boot,[.025,.975]).tolist()})
    diffs = np.asarray([p["xsa_minus_sa"] for p in pairs])
    return {"pairs": pairs, "mean_xsa_minus_sa": float(diffs.mean()), "paired_seeds": len(seeds),
            "seed_standard_deviation": float(diffs.std(ddof=1)) if len(seeds)>1 else None,
            "seed_standard_error": float(diffs.std(ddof=1)/np.sqrt(len(seeds))) if len(seeds)>1 else None,
            "scientific_result": all(r["scientific_result"] for r in records),
            "uncertainty_note": "window bootstrap assumes independent windows; same-document correlations may narrow intervals. This is evaluation-sample uncertainty, not training-seed variability. One seed supports feasibility only."}


def committed_evaluations(events, max_update):
    evals = {}
    for row in events:
        if row["type"] == "resume":
            evals = {step: value for step, value in evals.items() if step <= row["update"]}
        elif row["type"] == "eval":
            evals[row["update"]] = row
    return [evals[step] for step in sorted(evals) if step <= max_update]


def plot_curves(run_dirs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    fixture = False
    for directory in run_dirs:
        rows = [json.loads(s) for s in (Path(directory)/"metrics.jsonl").read_text().splitlines()]
        evaluation = json.loads((Path(directory)/"evaluation.json").read_text())
        fixture |= not evaluation['scientific_result']
        rows = committed_evaluations(rows, evaluation['update'])
        for ax,key in zip(axes,("tokens","elapsed_seconds")):
            ax.plot([r[key] for r in rows],[r["cross_entropy"] for r in rows],marker="o",label=Path(directory).name)
            ax.set(xlabel="Training tokens" if key=="tokens" else "Elapsed training/evaluation seconds",ylabel="Held-out cross-entropy")
            ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("Execution fixture only — no scientific conclusion" if fixture else "Paired held-out learning curves")
    fig.savefig(output,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument("--results",nargs="+",required=True);p.add_argument("--output",required=True)
    p.add_argument("--runs",nargs="+");p.add_argument("--plot")
    a=p.parse_args(); result=paired_summary([json.loads(Path(f).read_text()) for f in a.results])
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    if a.runs and a.plot: plot_curves(a.runs,a.plot)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
