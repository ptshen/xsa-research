"""Direction A verification, strict screen comparison and artifact collection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT = Path('evidence/direction_a')
TAGS = ('025', '050', '075')
TAUS = (.25, .5, .75)
BASELINES = [Path(f'runs/pilot-{kind}-s17') for kind in ('sa', 'xsa')]
RUNS = [Path(f'runs/direction-a-tau{tag}-s17') for tag in TAGS]


def source_hashes():
    names = ('model','train','data','config','initialize','evaluate','diagnostics','benchmark')
    paths = [Path(f'xsa/{name}.py') for name in names] + [Path(f'configs/direction_a_{tag}.json') for tag in TAGS]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def screen_summary(records):
    import numpy as np
    expected = {('sa',None), ('xsa',None)} | {('alignment_cap',t) for t in TAUS}
    if len(records) != 5 or {(r['attention'],r.get('alignment_tau')) for r in records} != expected:
        raise ValueError('screen requires SA, XSA and exactly three distinct thresholds')
    for r in records:
        if (r['tokens'],r['update'],r['seed'],r['scientific_result']) != (67108864,2048,17,True):
            raise ValueError('incomplete or mismatched screen budget/seed/data kind')
        if len(r['window_losses']) != 64 or not all(math.isfinite(v) for v in r['window_losses']):
            raise ValueError('invalid evaluation windows')
        if not math.isclose(r['cross_entropy'],float(np.mean(r['window_losses'])),abs_tol=1e-10):
            raise ValueError('loss does not match window results')
    for key in ('comparison_config_sha256','study_config_sha256','data_manifest_sha256',
                'evaluation_windows_sha256','initial_state_sha256','evaluated_tokens'):
        if len({r[key] for r in records}) != 1: raise ValueError(f'unmatched {key}')
    if len({r['software']['gpu'] for r in records}) != 1: raise ValueError('unmatched GPU type')
    baselines = {r['attention']:r for r in records if r['attention'] in ('sa','xsa')}
    rows = []
    for r in records:
        row = {k:r.get(k) for k in ('attention','alignment_tau','cross_entropy','perplexity','tokens','seed')}
        for kind, baseline in baselines.items():
            delta = np.asarray(r['window_losses']) - np.asarray(baseline['window_losses'])
            boot = np.random.default_rng(2026).choice(delta,(2000,len(delta)),replace=True).mean(1)
            row[f'minus_{kind}'] = r['cross_entropy'] - baseline['cross_entropy']
            row[f'minus_{kind}_window_bootstrap_95ci'] = np.quantile(boot,[.025,.975]).tolist()
        rows.append(row)
    return {'results':rows,'selection_data':'validation','training_seeds':1,
            'uncertainty_note':'Exploratory threshold selection. Window bootstrap assumes independent windows; shared-document correlations can narrow intervals. No training-seed uncertainty or independent test result.'}


def gate():
    proof = json.loads((ROOT/'verification.json').read_text())
    if not proof['passed'] or proof['source_sha256'] != source_hashes():
        raise ValueError('missing verification or numerical source/config changed since verification')
    if not proof['screen_fits_time_and_storage']: raise ValueError('screen does not fit approved resources')


def verify(test_xml):
    import xml.etree.ElementTree as ET
    import torch
    from .benchmark import benchmark
    from .config import load_config
    from .evaluate import evaluate
    from .initialize import architecture
    from .model import GPT, ModelConfig
    from .train import load_checkpoint, state_hash
    suites = list(ET.parse(test_xml).getroot().iter('testsuite'))
    counts = {key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    if counts['tests'] < 40 or any(counts[k] for k in ('failures','errors','skipped')):
        raise ValueError(f'CUDA correctness gate failed: {counts}')
    identities = []
    for path in BASELINES:
        saved = json.loads((path/'evaluation.json').read_text())
        actual = evaluate(path/'checkpoint.pt','data/pilot-v2','cuda')
        for key in ('tokens','update','seed','data_manifest_sha256','evaluation_windows_sha256',
                    'initial_state_sha256','comparison_config_sha256','checkpoint_sha256'):
            if actual[key] != saved[key]: raise ValueError(f'baseline changed: {key}')
        if max(abs(a-b) for a,b in zip(actual['window_losses'],saved['window_losses'])) > 1e-6:
            raise ValueError('baseline evaluation no longer agrees')
        state = load_checkpoint(path/'checkpoint.pt')
        identities.append((state['batch_hashes'],state['data_state']))
        (ROOT/f'rechecked-{path.name}.json').write_text(json.dumps(actual,indent=2)+'\n')
        del state
    if identities[0] != identities[1]: raise ValueError('baseline batch/data identity mismatch')
    initial = load_checkpoint('runs/initial/pilot-s17.pt')
    for tag in TAGS:
        cfg = load_config(f'configs/direction_a_{tag}.json')
        if architecture(cfg) != initial['architecture'] or cfg['seed'] != initial['seed']:
            raise ValueError('initialization architecture mismatch')
        model = GPT(ModelConfig(**cfg['model'])); model.load_state_dict(initial['model'])
        if state_hash(model) != actual['initial_state_sha256']: raise ValueError('initial tensor identity mismatch')
        del model
    del initial
    variants = [('sa',None),('xsa',None)] + [('alignment_cap',t) for t in TAUS]
    profile = benchmark(load_config('configs/pilot.json'),8,3,12,variants=variants)
    (ROOT/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    disk = int(subprocess.check_output(['du','-sb','.'],text=True).split()[0])
    checkpoint_bytes = max((p/'checkpoint.pt').stat().st_size for p in BASELINES)
    # Three new latest/previous pairs and one atomic temporary checkpoint.
    peak = disk + 7 * checkpoint_bytes + 32 * 1024**2
    times = [r['estimated_run_gpu_hours']*3600 for r in profile['results'] if r['attention']=='alignment_cap']
    fits = peak < 8*1024**3 and all(t*1.5+90 < 1800 for t in times)
    proof = {'passed':True,'tests':counts,'baseline_recheck_passed':True,'initialization_passed':True,
             'baseline_batches_passed':True,'source_sha256':source_hashes(),'project_bytes':disk,
             'project_peak_estimate_bytes':peak,'compute_only_run_seconds':times,
             'screen_fits_time_and_storage':fits,'job_id':__import__('os').environ.get('SLURM_JOB_ID')}
    (ROOT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    if not fits: raise ValueError('measured time/storage estimate does not fit authorization')
    print(json.dumps(proof),flush=True)


def committed_training(events):
    rows = {}
    for event in events:
        if event['type'] == 'resume':
            rows = {step:row for step,row in rows.items() if step <= event['update']}
        elif event['type'] == 'train':
            rows[event['update']] = event
    return [rows[step] for step in sorted(rows)]


def collect():
    import torch
    from .collect import plot_curves
    from .data import checksum, validate_manifest
    from .evaluate import comparison_hash, load_model
    gate()
    validate_manifest('data/pilot-v2')
    paths = BASELINES + RUNS
    records, identities, timing = [], [], []
    expected_reference = None
    for path in paths:
        record = json.loads((path/'evaluation.json').read_text())
        model, state = load_model(path/'checkpoint.pt','cpu')
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError(f'nonfinite checkpoint: {path}')
        if checksum(path/'checkpoint.pt') != record['checkpoint_sha256']:
            raise ValueError(f'checkpoint/evaluation mismatch: {path}')
        if comparison_hash(state['config']) != record['comparison_config_sha256']:
            raise ValueError(f'checkpoint/config mismatch: {path}')
        if state['update'] != 2048 or state['tokens_seen'] != 67108864:
            raise ValueError(f'incomplete checkpoint: {path}')
        reference = (state['batch_hashes'],state['data_state'],state['initial_state_sha256'],
                     state['data_manifest_sha256'],state['evaluation_windows_sha256'],comparison_hash(state['config']))
        if expected_reference is None: expected_reference = reference
        if reference != expected_reference: raise ValueError(f'unmatched checkpoint identity: {path}')
        events = [json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
        rows = committed_training(events)
        if [r['update'] for r in rows] != list(range(1,2049)):
            raise ValueError(f'incomplete committed metrics: {path}')
        if [r['batch_sha256'] for r in rows] != state['batch_hashes']:
            raise ValueError(f'metric/checkpoint batch mismatch: {path}')
        if not all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in rows):
            raise ValueError(f'nonfinite training metrics: {path}')
        if path in RUNS and not all(r['alignment_tau'] == record['alignment_tau'] for r in rows):
            raise ValueError(f'threshold metadata mismatch: {path}')
        runtimes = [json.loads(p.read_text()) for p in path.glob('runtime-*.json')]
        timing.append({'run':str(path),'training_loop_seconds':state['elapsed_seconds'],
                       'training_loop_tokens_per_second':state['tokens_seen']/state['elapsed_seconds'],
                       'peak_training_allocated_bytes':max((r['peak_allocated_bytes'] or 0 for r in runtimes),default=None),
                       'runtime_sessions':runtimes})
        identities.append({'run':str(path),'checkpoint_sha256':record['checkpoint_sha256'],
                           'previous_checkpoint_sha256':checksum(path/'checkpoint.previous.pt'),
                           'initial_state_sha256':state['initial_state_sha256'],'updates':state['update'],
                           'all_batch_hashes_match_baseline':True,'final_data_state_matches_baseline':True,
                           'parameters':model.parameter_counts()})
        records.append(record)
        del model, state
    summary = screen_summary(records)
    jobs = json.loads((ROOT/'job_ledger.json').read_text())
    ids = ','.join(j['job_id'] for j in jobs)
    accounting = subprocess.check_output(['sacct','-X','-P','-j',ids,'-o','JobID,JobName,State,ElapsedRaw,ExitCode,NodeList,AllocTRES'],text=True)
    (ROOT/'accounting.psv').write_text(accounting)
    by_id = {r.split('|')[0]:r.split('|') for r in accounting.splitlines()[1:] if r}
    for tag in TAGS:
        job = [j for j in jobs if j['kind']=='train' and j['tag']==tag][-1]
        row = by_id[job['job_id']]
        if row[2] != 'COMPLETED' or row[4] != '0:0': raise ValueError(f'training job not successful: {row}')
    gpu_seconds = sum(int(by_id[j['job_id']][3]) for j in jobs if j['kind'] in ('verify','train'))
    verification_seconds = sum(int(by_id[j['job_id']][3]) for j in jobs if j['kind']=='verify')
    if gpu_seconds > 7200 or verification_seconds > 300: raise ValueError('stage actual GPU budget exceeded')
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    for path in paths:
        shutil.copytree(path,out/path.name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.pt','*.tmp'))
    plot_curves(paths,out/'curves.png')
    for name, value in [('summary.json',summary),('identity.json',identities),('timing.json',timing)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    for log in Path('logs').glob('direction-a-*.log'): shutil.copy2(log,ROOT/log.name)
    disk = int(subprocess.check_output(['du','-sb','.'],text=True).split()[0])
    if disk > 8*1024**3: raise ValueError('project storage cap exceeded')
    manifest = {'completed':True,'source_sha256':source_hashes(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'summary':summary,'identities':identities,'timing':timing,'stage_gpu_seconds':gpu_seconds,
                'verification_gpu_seconds':verification_seconds,'project_bytes':disk,'jobs':jobs,
                'selection_data':'validation','next_stage_launched':False}
    Path('DIRECTION_A_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines = ['# Direction A exploratory screen', '',
             'All three caps completed 2048 updates / 67,108,864 tokens with the preserved seed-17 initialization, ordered training batches and validation windows. Existing SA/XSA runs were reused after numerical and checkpoint checks.', '',
             '| Method | Threshold | Validation loss | Perplexity | Minus SA | Minus XSA |',
             '|---|---:|---:|---:|---:|---:|']
    for r in summary['results']:
        lines.append(f"| {r['attention']} | {r['alignment_tau']} | {r['cross_entropy']:.8f} | {r['perplexity']:.5f} | {r['minus_sa']:+.8f} | {r['minus_xsa']:+.8f} |")
    lines += ['',f'Stage allocated GPU time: **{gpu_seconds} seconds**, including **{verification_seconds} seconds** of verification/profile, within 7200/300 seconds. Project storage: **{disk/2**30:.3f} GiB**, within 8 GiB.', '',
              'This is one-seed validation selection. All thresholds are reported; the winning validation score is not independent confirmation. Window-bootstrap intervals in the summary describe evaluation-sample uncertainty under an independent-window approximation, not training-seed variation. Shared-document correlations may narrow them.', '',
              'Numerical tests, preserved-source regression tests, actual baseline checkpoint re-evaluation, and the measured five-method profile are under `evidence/direction_a/`. Full source hashes, checkpoint identities, runtime records and job IDs are in [DIRECTION_A_MANIFEST.json](DIRECTION_A_MANIFEST.json). [Learning curves](evidence/direction_a/results/curves.png) and [all results](evidence/direction_a/results/summary.json) include both baselines and every threshold.', '',
              'The implementation clips the parallel coefficient token by token before head concatenation. It adds no parameters and retains SDPA. Projection reductions use fp32 under bf16; the final cast can slightly relax the numerical cosine bound. Below the existing normalization epsilon the vector is not unit length, so no cosine-bound claim is made. Tau zero uses the original XSA operation exactly. First-position and collinear behavior follows the formula; no special-case copying path was added.', '',
              'Diagnostics use the original fixed four 128-token training windows, float32 forward, per-layer/head activation, alignment and update norms, with zero/near-zero output fractions separate. They are small-sample geometric measurements, not evidence of language-model quality.', '',
              'Before attributing improvement specifically to adaptive clipping, compare against a fixed-strength partial-removal control with an equal tuning allowance. Freeze the selected threshold and evaluate fresh paired seeds and untouched evaluation data. These controls and follow-up runs are unlaunched.']
    Path('DIRECTION_A_REPORT.md').write_text('\n'.join(lines)+'\n')
    with Path('PROGRESS.md').open('a') as f:
        f.write(f'\n## Direction A collected\n\nAll three thresholds completed and passed checkpoint, configuration, initialization, batch-order and evaluation identity checks. GPU allocation {gpu_seconds}/7200 s, verification/profile {verification_seconds}/300 s, project {disk/2**30:.3f}/8 GiB. Full outcomes in DIRECTION_A_REPORT.md. Next: independent artifact/plot audit, final accounting and GitHub publication; no further experiment scheduled.\n')
    print(json.dumps(summary),flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action',choices=['verify','gate','collect'])
    p.add_argument('--test-xml'); a = p.parse_args()
    if a.action == 'verify': verify(a.test_xml)
    elif a.action == 'gate': gate()
    else: collect()


if __name__ == '__main__': main()
