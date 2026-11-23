"""Direction A reservations, duplicate prevention and read-only scheduler checks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def check_budget(jobs, budget, key, kind, seconds, active):
    if seconds <= 0 or kind not in ('verify', 'train', 'collect'):
        raise ValueError('invalid stage allocation')
    if any(j['key'] == key for j in jobs):
        raise ValueError('existing key; inspect its job instead of resubmitting')
    if any(j.get('job_id') is None for j in jobs):
        raise ValueError('uncertain submission; reconcile it before another submission')
    if active:
        raise ValueError(f'active project jobs: {active}')
    proposed = jobs + [{'kind':kind, 'seconds':seconds}]
    gpu = sum(j['seconds'] for j in proposed if j['kind'] in ('verify','train'))
    verify = sum(j['seconds'] for j in proposed if j['kind'] == 'verify')
    cpu = sum(j['seconds'] for j in proposed if j['kind'] == 'collect')
    if gpu > budget['gpu_seconds'] or verify > budget['verification_gpu_seconds'] or cpu > budget['cpu_seconds']:
        raise ValueError('stage reservation budget exceeded')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=['verify','train','collect']); p.add_argument('--key', required=True)
    p.add_argument('--seconds', type=int, required=True); p.add_argument('--tag', choices=['025','050','075'])
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    root = Path('evidence/direction_a'); root.mkdir(exist_ok=True)
    with (root/'submission.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root/'job_ledger.json'
        jobs = json.loads(path.read_text()) if path.exists() else []
        budget = json.loads((root/'budget.json').read_text())
        queue = subprocess.check_output(['squeue','--me','-h','-o','%i|%j|%T'], text=True)
        prior = json.loads(Path('evidence/job_ledger.json').read_text())['jobs']
        ids = ','.join(str(j['job_id']) for j in prior + jobs if j.get('job_id'))
        accounting = subprocess.check_output(['sacct','-X','-P','-j',ids,'-o','JobID,State,ElapsedRaw,ExitCode'], text=True)
        (root/f'pre-submit-{a.key}.txt').write_text(queue + '\n' + accounting)
        check_budget(jobs, budget, a.key, a.kind, a.seconds, [r for r in queue.splitlines() if '|xsa-' in r])
        if a.kind == 'train':
            if not a.tag: p.error('training requires --tag')
            from .direction_a import gate
            gate()  # Standard-library-only gate on login nodes.
        hours, rest = divmod(a.seconds, 3600); minutes, seconds = divmod(rest, 60)
        cmd = ['sbatch','--parsable','--no-requeue','--account=edu','--partition=short',
               f'--job-name=xsa-a-{a.kind}', f'--time={hours:02d}:{minutes:02d}:{seconds:02d}',
               f'--output=logs/{a.key}-%j.log']
        if a.kind == 'collect': cmd += ['--cpus-per-task=2','--mem=8G']
        else: cmd += ['--gres=gpu:A6000:1','--cpus-per-task=8','--mem=32G']
        cmd.append(f'slurm/direction_a_{a.kind}.sbatch')
        env = dict(os.environ, XSA_A_TAG=a.tag or '', XSA_RESUME='1' if a.resume else '0')
        entry = {'key':a.key,'kind':a.kind,'seconds':a.seconds,'tag':a.tag,'resume':a.resume,
                 'job_id':None,'submitted_at':time.time(),'command':cmd,'state':'submission_pending'}
        jobs.append(entry); path.write_text(json.dumps(jobs, indent=2)+'\n')
        result = subprocess.check_output(cmd, env=env, text=True).strip()
        entry.update(job_id=result.split(';')[0], state='submitted')
        path.write_text(json.dumps(jobs, indent=2)+'\n'); print(json.dumps(entry, indent=2))


if __name__ == '__main__': main()
