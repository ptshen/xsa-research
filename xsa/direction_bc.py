"""B/C verification, fixed resource plan, and strict nine-configuration collection."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from .direction_a import committed_training

ROOT = Path('evidence/direction_bc')
TAGS = ('b010', 'b050', 'b100', 'c')
OLD_RUNS = [Path(f'runs/pilot-{k}-s17') for k in ('sa', 'xsa')] + [
    Path(f'runs/direction-a-tau{t}-s17') for t in ('025', '050', '075')]
NEW_RUNS = [Path(f'runs/direction-{tag}-s17') for tag in TAGS]
METHOD_KEYS = ('attention', 'alignment_tau', 'similarity_lambda', 'query_chunk_size')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_hashes():
    paths = list(Path('xsa').glob('*.py')) + list(Path('tests').glob('*.py'))
    paths += [Path(f'configs/direction_{tag}.json') for tag in TAGS]
    paths += list(Path('slurm').glob('direction_bc_*.sbatch'))
    paths += [Path('slurm/train.sbatch'), ROOT/'budget.json', ROOT/'audit_remote.py', Path('DIRECTIONS_BC_PROTOCOL.md')]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def allocation_plan(compute_seconds, project_bytes, checkpoint_bytes):
    # 50% compute contingency plus 120s for I/O/evaluation/diagnostics/startup.
    seconds = [max(1200, math.ceil((t * 1.5 + 120) / 60) * 60) for t in compute_seconds]
    peak = project_bytes + 9 * checkpoint_bytes + 128 * 2**20
    return {'train_seconds': seconds, 'compute_only_seconds': compute_seconds,
            'project_peak_bytes': peak, 'verification_reserved_allowance': 600,
            'fits': len(seconds) == 4 and sum(seconds)+600 <= 14400 and peak < 12*2**30,
            'assumptions': 'Same A6000, batch/accumulation/precision; 50% compute margin +120s per run; four latest/previous pairs, one atomic temporary, 128MiB metadata/test allowance. Queue time excluded; peak disk is a conservative estimate, not continuous sampling.'}


def validate_record_config(record, config):
    m = config['model']
    expected = (m['attention_type'], m.get('alignment_tau'), m.get('similarity_lambda'), m.get('query_chunk_size'))
    if tuple(record.get(k) for k in METHOD_KEYS) != expected:
        raise ValueError('method metadata does not match checkpoint/config')


def screen_summary(records):
    import numpy as np
    expected = {('sa',None,None,None), ('xsa',None,None,None), ('output_geometry',None,None,None)}
    expected |= {('alignment_cap',t,None,None) for t in (.25,.5,.75)}
    expected |= {('value_penalty',None,s,128) for s in (.1,.5,1.)}
    if len(records) != 9 or {tuple(r.get(k) for k in METHOD_KEYS) for r in records} != expected:
        raise ValueError('screen requires all nine distinct frozen configurations')
    for r in records:
        if (r['tokens'],r['update'],r['seed'],r['scientific_result'],r['evaluated_tokens']) != (67108864,2048,17,True,32768):
            raise ValueError('incomplete or unmatched budget/seed/evaluation')
        if len(r['window_losses']) != 64 or not all(math.isfinite(v) for v in r['window_losses']):
            raise ValueError('invalid window losses')
        ce=float(np.mean(r['window_losses']))
        if not math.isclose(ce,r['cross_entropy'],rel_tol=0,abs_tol=1e-10) or not math.isclose(math.exp(ce),r['perplexity'],rel_tol=1e-10):
            raise ValueError('reported metrics disagree with per-window losses')
    for key in ('comparison_config_sha256','study_config_sha256','data_manifest_sha256',
                'evaluation_windows_sha256','initial_state_sha256'):
        if len({r[key] for r in records}) != 1: raise ValueError(f'unmatched {key}')
    if len({json.dumps(r['software'],sort_keys=True) for r in records}) != 1:
        raise ValueError('unmatched software/GPU')
    baselines={r['attention']:r for r in records if r['attention'] in ('sa','xsa')}
    rows=[]
    for r in records:
        row={k:r.get(k) for k in (*METHOD_KEYS,'cross_entropy','perplexity','tokens','seed')}
        for kind,baseline in baselines.items():
            delta=np.asarray(r['window_losses'])-np.asarray(baseline['window_losses'])
            boot=np.random.default_rng(2026).choice(delta,(2000,64),replace=True).mean(1)
            row[f'minus_{kind}']=float(delta.mean())
            row[f'minus_{kind}_window_bootstrap_95ci']=np.quantile(boot,[.025,.975]).tolist()
        rows.append(row)
    return {'results':rows,'primary_metric':'token-weighted held-out cross-entropy (nats/token)',
            'perplexity':'exp(cross_entropy); same ranking, not independent evidence',
            'selection_data':'validation','training_seeds':1,
            'uncertainty_note':'2000 paired window resamples, RNG 2026; independence approximation can be optimistic for shared documents. Does not quantify training-seed variability or adjust for configuration selection. No untouched test.'}


def gate():
    proof=json.loads((ROOT/'verification.json').read_text())
    if not proof['passed'] or proof['source_sha256'] != source_hashes():
        raise ValueError('GPU verification absent/failed or frozen sources changed')
    plan=json.loads(Path('DIRECTIONS_BC_PLAN.json').read_text())
    if not plan['fits'] or plan != proof['plan']:
        raise ValueError('resource plan changed or does not fit')
    return proof


def verify(test_xml):
    import time
    import xml.etree.ElementTree as ET
    import torch
    from .benchmark import benchmark
    from .config import load_config
    from .data import checksum
    from .evaluate import evaluate, comparison_hash
    from .initialize import architecture
    from .model import GPT, ModelConfig
    from .train import load_checkpoint, state_hash
    suites=list(ET.parse(test_xml).getroot().iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts['tests'] < 87 or any(counts[k] for k in ('failures','errors','skipped')):
        raise ValueError(f'CUDA correctness gate failed: {counts}')
    identities=[]; rechecks=[]
    for path in OLD_RUNS:
        started=time.perf_counter()
        saved=json.loads((path/'evaluation.json').read_text())
        actual=evaluate(path/'checkpoint.pt','data/pilot-v2','cuda')
        for key in ('tokens','update','seed','data_manifest_sha256','evaluation_windows_sha256',
                    'initial_state_sha256','comparison_config_sha256','checkpoint_sha256','software'):
            if actual[key] != saved[key]: raise ValueError(f'preserved run changed: {path}: {key}')
        delta=max(abs(a-b) for a,b in zip(actual['window_losses'],saved['window_losses']))
        if delta > 1e-6: raise ValueError(f'preserved evaluation changed: {path}: {delta}')
        state=load_checkpoint(path/'checkpoint.pt')
        identities.append((state['batch_hashes'],state['data_state']))
        write_json(ROOT/f'rechecked-{path.name}.json',actual)
        rechecks.append({'run':str(path),'max_window_delta':delta,'evaluation_and_loading_seconds':time.perf_counter()-started})
        del state
    if any(row != identities[0] for row in identities): raise ValueError('preserved data-order mismatch')
    initial=load_checkpoint('runs/initial/pilot-s17.pt')
    baseline_config=load_checkpoint(OLD_RUNS[0]/'checkpoint.pt')['config']
    configs=[]
    for tag in TAGS:
        cfg=load_config(f'configs/direction_{tag}.json')
        if architecture(cfg) != initial['architecture'] or cfg['seed'] != initial['seed']:
            raise ValueError('initialization architecture mismatch')
        cfg['initial_state_path']='runs/initial/pilot-s17.pt'
        if comparison_hash(cfg) != comparison_hash(baseline_config): raise ValueError('pilot recipe changed')
        model=GPT(ModelConfig(**cfg['model'])); model.load_state_dict(initial['model'])
        if state_hash(model) != actual['initial_state_sha256']: raise ValueError('initial state mismatch')
        configs.append(cfg); del model
    del initial
    variants=[('sa',None),('xsa',None)] + [('alignment_cap',t) for t in (.25,.5,.75)]
    variants += [dict(attention_type='value_penalty',similarity_lambda=s,query_chunk_size=128) for s in (.1,.5,1.)]
    variants += [dict(attention_type='output_geometry')]
    profile=benchmark(load_config('configs/pilot.json'),8,3,12,variants=variants)
    write_json(ROOT/'profile.json',profile)
    compute=[r['estimated_run_gpu_hours']*3600 for r in profile['results'][-4:]]
    disk=int(subprocess.check_output(['du','-sb','.'],text=True).split()[0])
    checkpoint=max((p/'checkpoint.pt').stat().st_size for p in OLD_RUNS)
    plan=allocation_plan(compute,disk,checkpoint)
    quota=subprocess.check_output(['/usr/lpp/mmfs/bin/mmlsquota','-u',os.environ['USER'],'insomnia'],text=True)
    (ROOT/'quota-at-verification.txt').write_text(quota)
    home=next(line.split() for line in quota.splitlines() if line.split()[:2]==['insomnia','home'])
    quota_free=(int(home[5])-int(home[3])-int(home[6]))*1024
    plan['home_quota_free_bytes']=quota_free
    plan['fits'] = plan['fits'] and plan['project_peak_bytes']-disk < quota_free
    plan['configs']=configs
    write_json('DIRECTIONS_BC_PLAN.json',plan)
    proof={'passed':True,'tests':counts,'preserved_run_rechecks':rechecks,'shared_initialization_verified':True,
           'all_preserved_batches_equal':True,'source_sha256':source_hashes(),'plan':plan,'project_bytes':disk,
           'initial_file_sha256':checksum('runs/initial/pilot-s17.pt'),'job_id':os.environ.get('SLURM_JOB_ID')}
    write_json(ROOT/'verification.json',proof)
    if not plan['fits']: raise ValueError('complete screen exceeds time/storage/quota')
    print(json.dumps(proof),flush=True)


def plot_curves(paths, output, minimum_update=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .collect import committed_evaluations
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    labels=['SA','XSA','A τ=.25','A τ=.50','A τ=.75','B λ=.1','B λ=.5','B λ=1','C']
    for path,label in zip(paths,labels):
        events=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
        rows=[r for r in committed_evaluations(events,2048) if r['update']>=minimum_update]
        for ax,key in zip(axes,('tokens','elapsed_seconds')):
            ax.plot([r[key] for r in rows],[r['cross_entropy'] for r in rows],label=label,marker='.',markersize=3)
            ax.set(xlabel='Processed training tokens' if key=='tokens' else 'Training/evaluation loop seconds',ylabel='Validation cross-entropy (nats/token)')
            ax.grid(alpha=.2)
    axes[1].legend(fontsize=9,ncol=2)
    fig.suptitle('Seed 17 validation screen — all nine configurations' + ('; second half enlarged' if minimum_update else ''))
    fig.savefig(output,dpi=170); plt.close(fig)


def collect():
    import torch
    from .config import tokens_per_update
    from .data import checksum, validate_manifest
    from .evaluate import comparison_hash, load_model
    gate(); validate_manifest('data/pilot-v2')
    paths=OLD_RUNS+NEW_RUNS
    records=[]; identities=[]; timing=[]; reference=None
    for path in paths:
        record=json.loads((path/'evaluation.json').read_text())
        model,state=load_model(path/'checkpoint.pt','cpu')
        validate_record_config(record,state['config'])
        if not all(torch.isfinite(p).all() for p in model.parameters()): raise ValueError(f'nonfinite weights: {path}')
        if checksum(path/'checkpoint.pt') != record['checkpoint_sha256']: raise ValueError('checkpoint/evaluation mismatch')
        if comparison_hash(state['config']) != record['comparison_config_sha256']: raise ValueError('config/evaluation mismatch')
        if (state['update'],state['tokens_seen'],tokens_per_update(state['config'])) != (2048,67108864,32768):
            raise ValueError('incomplete checkpoint/token budget')
        identity=(state['batch_hashes'],state['data_state'],state['initial_state_sha256'],
                  state['data_manifest_sha256'],state['evaluation_windows_sha256'],comparison_hash(state['config']))
        if reference is None: reference=identity
        if identity != reference: raise ValueError(f'checkpoint identity mismatch: {path}')
        events=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
        rows=committed_training(events)
        if [r['update'] for r in rows] != list(range(1,2049)): raise ValueError('incomplete committed history')
        if [r['batch_sha256'] for r in rows] != state['batch_hashes']: raise ValueError('batch history differs from checkpoint')
        if not all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) and r['tokens']==r['update']*32768 for r in rows):
            raise ValueError('invalid training metrics')
        if path in NEW_RUNS:
            for r in rows: validate_record_config(r,state['config'])
        runtimes=[json.loads(p.read_text()) for p in path.glob('runtime-*.json')]
        timing.append({'run':str(path),'training_loop_seconds':state['elapsed_seconds'],
                       'training_loop_tokens_per_second':67108864/state['elapsed_seconds'],
                       'peak_training_allocated_bytes':max((r['peak_allocated_bytes'] or 0 for r in runtimes),default=None),
                       'runtime_sessions':runtimes})
        identities.append({'run':str(path),'checkpoint_sha256':record['checkpoint_sha256'],
                           'previous_checkpoint_sha256':checksum(path/'checkpoint.previous.pt'),
                           'initial_state_sha256':state['initial_state_sha256'],'parameters':model.parameter_counts(),
                           'updates':state['update'],'all_batch_hashes_match_baseline':True,'final_data_state_matches_baseline':True})
        records.append(record); del model,state
    summary=screen_summary(records)
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    for path in paths: shutil.copytree(path,out/path.name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.pt','*.tmp'))
    for name,value in [('summary',summary),('identity',identities),('timing',timing)]: write_json(out/f'{name}.json',value)
    plot_curves(paths,out/'curves.png'); plot_curves(paths,out/'curves_zoom.png',1024)
    for log in Path('logs').glob('direction-bc-*.log'): shutil.copy2(log,ROOT/log.name)
    disk=int(subprocess.check_output(['du','-sb','.'],text=True).split()[0])
    if disk > 12*2**30: raise ValueError('project storage exceeded')
    manifest={'screen_collected':True,'final_accounting_pending':True,'source_sha256':source_hashes(),
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'summary':summary,'identities':identities,'timing':timing,'project_bytes':disk,
              'jobs':json.loads((ROOT/'job_ledger.json').read_text()),'selection_data':'validation','next_stage_launched':False}
    write_json('DIRECTIONS_BC_MANIFEST.json',manifest)
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['verify','gate','collect'])
    parser.add_argument('--test-xml'); args=parser.parse_args()
    if args.action=='verify': verify(args.test_xml)
    elif args.action=='gate': gate()
    else: collect()


if __name__=='__main__': main()
