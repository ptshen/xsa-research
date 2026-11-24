"""Conservative B/C reservations; no torch import or computation on login nodes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .direction_bc import ROOT, TAGS, OLD_RUNS, NEW_RUNS, gate
from .stage_submission import check_budget


def check_prior_states(ids, accounting):
    states={row.split('|')[0]:row.split('|')[1] for row in accounting.splitlines()[1:] if row}
    terminal=('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED')
    for job in ids:
        if not states.get(job,'').startswith(terminal):
            raise ValueError(f'job {job} not confirmed terminal: {states.get(job)}')


def check_training_target(path, resume):
    has_checkpoint=any(path.glob('checkpoint*.pt'))
    if has_checkpoint != resume:
        raise ValueError('existing run requires --resume; resume requires preserved checkpoint')
    if not resume and path.exists() and any(path.iterdir()):
        raise ValueError('nonempty run directory; inspect before starting')
    if resume and (path/'evaluation.json').exists():
        result=json.loads((path/'evaluation.json').read_text())
        if result['update']==2048 and result['tokens']==67108864:
            raise ValueError('completed run must not be resubmitted')


def save_ledger(path, jobs):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(jobs,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['verify','train','collect']); p.add_argument('--key',required=True)
    p.add_argument('--seconds',type=int,required=True); p.add_argument('--tag',choices=TAGS)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--account',default='edu'); p.add_argument('--partition',default='short')
    p.add_argument('--gpu',default='A6000'); p.add_argument('--cpus',type=int)
    p.add_argument('--memory'); a=p.parse_args()
    if not a.key.replace('-','').replace('_','').isalnum(): p.error('simple alphanumeric submission key required')
    ROOT.mkdir(exist_ok=True); Path('logs').mkdir(exist_ok=True)
    with (ROOT/'submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'job_ledger.json'; jobs=json.loads(path.read_text()) if path.exists() else []
        prior=json.loads(Path('evidence/job_ledger.json').read_text())['jobs']
        prior+=json.loads(Path('evidence/direction_a/job_ledger.json').read_text())
        queue=subprocess.check_output(['squeue','--me','-h','-o','%i|%j|%T'],text=True)
        ids=[str(j['job_id']) for j in prior+jobs if j.get('job_id')]
        accounting=subprocess.check_output(['sacct','-X','-P','-j',','.join(ids),'-o','JobID,State,ElapsedRaw,ExitCode'],text=True)
        (ROOT/f'pre-submit-{a.key}.txt').write_text(queue+'\n'+accounting)
        check_prior_states(ids,accounting)
        budget=json.loads((ROOT/'budget.json').read_text())
        check_budget(jobs,budget,a.key,a.kind,a.seconds,queue.splitlines())
        disk=int(subprocess.check_output(['du','-sb','.'],text=True).split()[0])
        if disk >= budget['project_bytes']: raise ValueError('project storage cap reached')
        if a.kind=='train':
            if not a.tag: p.error('training requires --tag')
            proof=gate(); i=TAGS.index(a.tag)
            check_training_target(NEW_RUNS[i],a.resume)
            remaining=[tag for tag in TAGS if tag!=a.tag and not any(j['kind']=='train' and j['tag']==tag for j in jobs)]
            reserved=sum(j['seconds'] for j in jobs if j['kind'] in ('verify','train'))+a.seconds
            if reserved+sum(proof['plan']['train_seconds'][TAGS.index(tag)] for tag in remaining)>budget['gpu_seconds']:
                raise ValueError('submission leaves insufficient reservation for remaining screen')
            if not a.resume and a.seconds<proof['plan']['train_seconds'][i]:
                raise ValueError('walltime below frozen conservative run estimate')
            # Retained pairs for unfinished runs plus one atomic-write temporary.
            checkpoint=max((run/'checkpoint.pt').stat().st_size for run in OLD_RUNS)
            unfinished=sum(not (run/'evaluation.json').exists() for run in NEW_RUNS)
            if disk+(2*unfinished+1)*checkpoint+128*2**20 > budget['project_bytes']:
                raise ValueError('remaining screen lacks conservative storage headroom')
        elif a.kind=='collect':
            gate()
            for run in NEW_RUNS:
                record=json.loads((run/'evaluation.json').read_text())
                if (record['update'],record['tokens'])!=(2048,67108864): raise ValueError('screen incomplete')
        hours,rest=divmod(a.seconds,3600); minutes,seconds=divmod(rest,60)
        cmd=['sbatch','--parsable','--no-requeue',f'--account={a.account}',f'--partition={a.partition}',
             f'--job-name=xsa-bc-{a.kind}',f'--time={hours:02d}:{minutes:02d}:{seconds:02d}',
             f'--output=logs/{a.key}-%j.log',f'--cpus-per-task={a.cpus or (2 if a.kind=="collect" else 8)}',
             f'--mem={a.memory or ("8G" if a.kind=="collect" else "32G")}']
        if a.kind!='collect': cmd.append(f'--gres=gpu:{a.gpu}:1')
        cmd.append(f'slurm/direction_bc_{a.kind}.sbatch')
        env=dict(os.environ,XSA_BC_TAG=a.tag or '',XSA_RESUME='1' if a.resume else '0')
        entry={'key':a.key,'kind':a.kind,'seconds':a.seconds,'tag':a.tag,'resume':a.resume,
               'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               'job_id':None,'submitted_at':time.time(),'command':cmd,'state':'submission_pending'}
        jobs.append(entry); save_ledger(path,jobs)
        result=subprocess.check_output(cmd,env=env,text=True).strip()
        entry.update(job_id=result.split(';')[0],state='submitted')
        save_ledger(path,jobs); print(json.dumps(entry,indent=2))


if __name__=='__main__': main()
