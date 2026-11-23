"""Conservative smoke budget and duplicate-job checks; no automatic retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def remaining_budget(ledger):
    return 600 - sum(j.get("reserved_seconds", 0) for j in ledger["jobs"] if j["kind"] == "smoke")


def check_submission(ledger, key, kind, seconds, active):
    if any(j["key"] == key for j in ledger["jobs"]):
        raise ValueError("existing submission key; inspect its squeue/sacct state instead of submitting again")
    if active:
        raise ValueError(f"active project jobs: {active}; only one experiment at a time")
    if kind == "smoke" and (seconds <= 0 or seconds > remaining_budget(ledger)):
        raise ValueError("smoke/profile budget exceeded")


def command(args):
    return subprocess.check_output(args,text=True).strip()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("kind",choices=("smoke","train","resume","prepare"));p.add_argument("--key",required=True)
    p.add_argument("--seconds",type=int,default=360);p.add_argument("--account",default="edu");p.add_argument("--partition",default="short")
    p.add_argument("--gpu",default="l40s");p.add_argument("--cpus",type=int,default=8);p.add_argument("--mem",default="32G")
    p.add_argument("--walltime",default="04:00:00");p.add_argument("--config",default="configs/pilot.json")
    p.add_argument("--data",default="data/pilot");p.add_argument("--output");p.add_argument("--attention",choices=("sa","xsa"),default="sa")
    p.add_argument("--seed",type=int,default=17);p.add_argument("--max-documents",type=int)
    p.add_argument('--initial-state')
    a=p.parse_args()
    root=Path.cwd();ledger_path=root/'evidence/job_ledger.json';ledger_path.parent.mkdir(exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    with (root/'evidence/submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'jobs':[]}
        queue=command(['squeue','--me','-h','-o','%i|%j|%T'])
        active=[row for row in queue.splitlines() if '|xsa-' in row]
        # Always inspect known IDs; failed observation is not permission to rerun.
        ids=','.join(str(j['job_id']) for j in ledger['jobs'] if j.get('job_id'))
        accounting=command(['sacct','-X','-n','-P','-j',ids,'-o','JobID,State,Elapsed,ExitCode']) if ids else ''
        print('Existing queue:',queue or '(empty)');print('Existing accounting:',accounting or '(empty)')
        check_submission(ledger,a.key,a.kind,a.seconds,active)
        if a.kind in ('train','resume') and not a.output:
            p.error('--output is required for training/resume')
        if a.kind=='prepare' and (not a.max_documents or a.max_documents<1):
            p.error('--max-documents is required for preparation')
        script='smoke.sbatch' if a.kind=='smoke' else 'prepare.sbatch' if a.kind=='prepare' else 'train.sbatch'
        env=os.environ.copy();env.update(XSA_ROOT=str(root),XSA_CONFIG=a.config,XSA_DATA=a.data,XSA_OUTPUT=a.output or '',
                                        XSA_ATTENTION=a.attention,XSA_SEED=str(a.seed),XSA_RESUME='1' if a.kind=='resume' else '0',
                                        XSA_SMOKE_SECONDS=str(a.seconds),XSA_MAX_DOCUMENTS=str(a.max_documents or 0))
        env['XSA_INITIAL_STATE'] = a.initial_state or ''
        cmd=['sbatch','--parsable','--no-requeue','--account',a.account,'--partition',a.partition,
             '--cpus-per-task',str(a.cpus),'--mem',a.mem,'--job-name',f'xsa-{a.kind}']
        if a.kind != 'prepare': cmd += ['--gres',f'gpu:{a.gpu}:1']
        # Whole smoke allocation is capped, including correctness/evaluation/startup.
        cmd += ['--time',f'00:{a.seconds//60:02d}:{a.seconds%60:02d}' if a.kind=='smoke' else a.walltime,
                '--output',f'logs/{a.key}-%j.log',str(root/'slurm'/script)]
        entry={'key':a.key,'kind':a.kind,'job_id':None,'submitted_at':time.time(),'reserved_seconds':a.seconds if a.kind=='smoke' else 0,
               'command':cmd,'state':'submission_pending'}
        ledger['jobs'].append(entry)
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
        # Keep reservation even if sbatch or connection fails: inspect before recovery.
        submitted=subprocess.check_output(cmd,env=env,text=True).strip()
        entry.update(job_id=submitted.split(';')[0],state='submitted')
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
        print(json.dumps(entry,indent=2))


if __name__=='__main__':
    main()
