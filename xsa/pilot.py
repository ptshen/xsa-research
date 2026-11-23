"""Completion gate and automatic collection for the authorized one-seed pilot."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def require_completed_pair_member(record, attention):
    expected={'tokens':67108864,'update':2048,'attention':attention,'scientific_result':True}
    for key,value in expected.items():
        if record.get(key)!=value:
            raise ValueError(f'incomplete/mismatched pilot {attention}: {key}={record.get(key)} expected {value}')


def finish():
    from .collect import paired_summary, plot_curves
    from .train import load_checkpoint
    from .data import checksum
    records=[];states=[]
    paths=[Path('runs/pilot-sa-s17'),Path('runs/pilot-xsa-s17')]
    for kind,path in zip(('sa','xsa'),paths):
        record=json.loads((path/'evaluation.json').read_text())
        require_completed_pair_member(record,kind)
        records.append(record)
        states.append(load_checkpoint(path/'checkpoint.pt'))
    if states[0]['batch_hashes']!=states[1]['batch_hashes'] or states[0]['data_state']!=states[1]['data_state']:
        raise ValueError('paired training batch order or final data state differs')
    if records[0]['software']['gpu']!=records[1]['software']['gpu']:
        raise ValueError('paired GPU types differ')
    summary=paired_summary(records)
    out=Path('evidence/pilot');out.mkdir(exist_ok=True)
    for path in paths:
        shutil.copytree(path,out/path.name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.pt','*.tmp'))
    (out/'paired_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot_curves(paths,out/'curves.png')
    identity={'passed':True,'updates':2048,'tokens_per_arm':67108864,'initial_state_sha256':states[0]['initial_state_sha256'],
              'all_batch_hashes_identical':True,'final_data_state_identical':True,
              'checkpoint_sha256':{str(p/'checkpoint.pt'):checksum(p/'checkpoint.pt') for p in paths}}
    (out/'pair_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    budget=json.loads(Path('evidence/pilot_budget.json').read_text())
    job_ids=','.join([budget['preparation_job'],budget['cancelled_preparation_job']]+[j['job_id'] for j in budget['training_jobs']])
    accounting=subprocess.check_output(['sacct','-X','-P','-j',job_ids,'-o','JobID,JobName,State,ElapsedRaw,ExitCode,NodeList,AllocTRES'],text=True)
    (out/'accounting.psv').write_text(accounting)
    for job in budget['training_jobs']:
        rows=[line.split('|') for line in accounting.splitlines() if line.startswith(job['job_id']+'|')]
        if len(rows)!=1 or rows[0][2]!='COMPLETED' or rows[0][4]!='0:0':
            raise ValueError(f'training accounting not successfully terminal: {job}')
    gpu_seconds=sum(int(line.split('|')[3]) for line in accounting.splitlines() if any(line.startswith(j['job_id']+'|') for j in budget['training_jobs']))
    if gpu_seconds>3600: raise ValueError('approved GPU budget exceeded')
    disk=int(subprocess.check_output(['du','-sb','.'],text=True).split()[0])
    if disk>budget['storage_bytes']: raise ValueError('approved project storage exceeded')
    for log in Path('logs').glob('pilot-*.log'):
        shutil.copy2(log,out/log.name)
    shutil.copy2('data/pilot-v2/manifest.json',out/'data_manifest.json')
    manifest={'stage':'approved exploratory pilot','completed':True,'seed':17,'gpu_seconds':gpu_seconds,'project_bytes':disk,
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'budget':budget,'summary':summary,'identity':identity,'limitations':['one training seed','FineWeb bounded prefix','document-level split approximates paper','no downstream task run','not a paper-scale reproduction']}
    Path('PILOT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pair=summary['pairs'][0]
    report=f'''# Exploratory SA/XSA pilot results

Both arms completed 2048 updates / 67,108,864 training tokens on {records[0]['software']['gpu']}, with identical saved initialization and all 2048 ordered minibatch hashes. Actual allocated GPU time: **{gpu_seconds} seconds**, within the approved 3600 seconds. Project storage at collection: {disk/2**30:.3f} GiB, within 5 GiB.

| Method | Held-out cross-entropy | Perplexity |
|---|---:|---:|
| SA | {records[0]['cross_entropy']:.8f} | {records[0]['perplexity']:.5f} |
| XSA | {records[1]['cross_entropy']:.8f} | {records[1]['perplexity']:.5f} |

XSA minus SA loss: **{pair['xsa_minus_sa']:+.8f}**. Paired evaluation-window bootstrap 95% interval: {pair['window_bootstrap_95ci']}. This is evaluation-sample uncertainty with an independent-window approximation; shared-document correlations may make it optimistic. There is one training seed, so no training-seed uncertainty estimate is available.

The pinned FineWeb prefix contains 99,962,616 training tokens from 146,329 documents and 36,418 validation tokens from 74 documents. Evaluation uses the same 64 nonoverlapping 512-token target windows (32,768 evaluated tokens). Tokenizer: GPT-2; document-preserving split seed 2026. This is a scaled-down prefix-data study, not paper reproduction. Downstream tasks and further seeds remain unrun.

[Learning curves](evidence/pilot/curves.png), [paired results](evidence/pilot/paired_results.json), [batch identity](evidence/pilot/pair_identity.json), [scheduler accounting](evidence/pilot/accounting.psv), and [manifest](PILOT_MANIFEST.json) contain the evidence. Diagnostics and all learning-curve rows are included in each arm's artifact directory. Checkpoints and prior checkpoints remain under `~/algollms/xsa/runs/pilot-{{sa,xsa}}-s17/`.

The first HTTP preparation attempt was cancelled for insufficient throughput; the parallel-download retry completed successfully. Its original log and data directory were retained. Only its abandoned partial-download cache was removed after the full pinned source was verified present.

No further training is automatically launched. Inspect these exploratory results before selecting any additional seeds, LR search, or extension experiments.
'''
    Path('PILOT_REPORT.md').write_text(report)
    Path('PILOT_STATUS.md').write_text('# Pilot complete\n\nBoth approved training arms finished and passed the paired checks. See [PILOT_REPORT.md](PILOT_REPORT.md) for results and limitations. No further training is scheduled.\n')
    # Include status even when the already-spooled collector stages older paths.
    subprocess.run(['git', 'add', 'PILOT_STATUS.md'], check=True)
    with Path('PROGRESS.md').open('a') as f:
        f.write(f'\n## Pilot finished automatically\n\nBoth training jobs completed the full budget; identical initialization/batch order verified. SA CE {records[0]["cross_entropy"]:.8f}, XSA CE {records[1]["cross_entropy"]:.8f}, XSA−SA {pair["xsa_minus_sa"]:+.8f}. GPU allocation {gpu_seconds}/3600 seconds; disk {disk/2**30:.3f}/5 GiB. See PILOT_REPORT.md. Next action: review the one-seed findings; no further training authorized or scheduled.\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['gate','finish']);a=p.parse_args()
    if a.action=='gate':
        require_completed_pair_member(json.loads(Path('runs/pilot-sa-s17/evaluation.json').read_text()),'sa')
    else: finish()


if __name__=='__main__':main()
