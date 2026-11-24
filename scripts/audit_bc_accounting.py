"""Audit final Slurm accounting and preserved compact artifacts; no GPU work."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def audit_accounting(jobs, rows):
    by_id={row['JobID']:row for row in rows}
    if len(by_id)!=len(rows) or len({j['job_id'] for j in jobs})!=len(jobs):
        raise ValueError('duplicate scheduler/ledger identities')
    reserved={'gpu':0,'verify':0,'cpu':0}; actual=dict(reserved)
    intervals=[]; details=[]; failures=[]; latest={}
    for job in jobs:
        if job['job_id'] not in by_id: raise ValueError('missing scheduler job')
        row=by_id[job['job_id']]; kind=job['kind']
        if kind not in ('verify','train','collect'): raise ValueError('unexpected stage job kind')
        state=row['State'].split()[0]
        if state not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'):
            raise ValueError('job is not terminal')
        elapsed=int(row['ElapsedRaw']); requested=int(row['TimelimitRaw'])*60
        if requested!=job['seconds'] or elapsed<0 or elapsed>requested:
            raise ValueError('requested/actual allocation mismatch or overrun')
        start,end=[datetime.fromisoformat(row[k]) for k in ('Start','End')]
        if (end-start).total_seconds()!=elapsed: raise ValueError('timestamps disagree with elapsed allocation')
        tres=dict(item.split('=',1) for item in row['AllocTRES'].split(','))
        gpu=int(tres.get('gres/gpu','0'))
        if kind=='collect':
            if gpu: raise ValueError('CPU collector unexpectedly allocated a GPU')
            reserved['cpu']+=requested; actual['cpu']+=elapsed
        else:
            if gpu!=1 or tres.get('gres/gpu:a6000')!='1': raise ValueError('expected one A6000')
            reserved['gpu']+=requested; actual['gpu']+=elapsed
            intervals.append((start,end,job['job_id']))
            if kind=='verify': reserved['verify']+=requested; actual['verify']+=elapsed
        if row['Account']!='edu' or row['Partition']!='short': raise ValueError('unexpected allocation')
        success=state=='COMPLETED' and row['ExitCode']=='0:0'
        if not success: failures.append(job['job_id'])
        latest[(kind,job.get('tag') if kind=='train' else None)]=success
        details.append({'job_id':job['job_id'],'kind':kind,'tag':job.get('tag'),'state':row['State'],
                        'exit_code':row['ExitCode'],'node':row['NodeList'],'allocated_seconds':elapsed,
                        'reserved_seconds':requested,'start':row['Start'],'end':row['End'],'alloc_tres':row['AllocTRES']})
    required={('verify',None),('collect',None)} | {('train',t) for t in ('b010','b050','b100','c')}
    if set(latest)!=required or not all(latest.values()): raise ValueError('missing or unsuccessful final stage')
    intervals.sort()
    if any(b[0]<a[1] for a,b in zip(intervals,intervals[1:])): raise ValueError('GPU allocations overlap')
    for key,cap in (('gpu',14400),('verify',600),('cpu',1800)):
        if max(reserved[key],actual[key])>cap: raise ValueError(f'{key} budget exceeded')
    return {'gpu_seconds':actual['gpu'],'verification_gpu_seconds':actual['verify'],'cpu_seconds':actual['cpu'],
            'reserved_gpu_seconds':reserved['gpu'],'reserved_verification_gpu_seconds':reserved['verify'],
            'reserved_cpu_seconds':reserved['cpu'],'sequential_gpu_allocations':True,
            'failed_or_cancelled_jobs':failures,'jobs':details}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path('evidence/direction_bc')
    jobs=json.loads((root/'job_ledger.json').read_text())
    with (root/'final_accounting.psv').open() as f: rows=list(csv.DictReader(f,delimiter='|'))
    result=audit_accounting(jobs,rows)
    if (root/'final_queue.txt').read_text().strip(): raise ValueError('user queue is not empty')
    storage=json.loads((root/'final_storage.json').read_text())
    if storage['project_bytes']>=12*2**30: raise ValueError('project storage cap exceeded')
    result['project_bytes']=storage['project_bytes']
    proof=json.loads((root/'verification.json').read_text())
    manifest=json.loads(Path('DIRECTIONS_BC_MANIFEST.json').read_text())
    independent=json.loads((root/'independent_checkpoint_audit.json').read_text())
    assert proof['passed'] and proof['tests']=={'tests':92,'failures':0,'errors':0,'skipped':0}
    assert independent['passed'] and independent['pairs_of_checkpoints_loaded']==9
    assert manifest['source_sha256']==proof['source_sha256']
    for path,digest in proof['source_sha256'].items(): assert sha(path)==digest,path
    assert next(j for j in jobs if j['job_id']==proof['job_id'])['state']=='completed'
    # Earlier budgets and completed checkpoints are immutable historical evidence.
    preserved=[]
    for path in ('evidence/job_ledger.json','evidence/direction_a/job_ledger.json'):
        old=subprocess.check_output(['git','show',f'8a1093a:{path}'])
        assert hashlib.sha256(old).hexdigest()==sha(path),path
        preserved.append(path)
    old_identities=json.loads(Path('evidence/direction_a/results/identity.json').read_text())
    new_identities={r['run']:r for r in manifest['identities']}
    for old in old_identities:
        new=new_identities[old['run']]
        for key in ('checkpoint_sha256','previous_checkpoint_sha256','initial_state_sha256','parameters'):
            assert old[key]==new[key],(old['run'],key)
        directory=Path(old['run']).name
        for path in (Path('evidence/direction_a/results')/directory).iterdir():
            if path.is_file(): assert sha(path)==sha(root/'results'/directory/path.name),str(path)
        preserved.append(old['run'])
    # The first B run predates C hardening; its actual verified source is retained.
    first=json.loads((root/'verification_v1/verification.json').read_text())
    for path,digest in first['source_sha256'].items():
        old=subprocess.check_output(['git','show',f'1db7eff:{path}'])
        assert hashlib.sha256(old).hexdigest()==digest,path
    result.update(passed=True,frozen_sources_match=True,preserved_historical_artifacts=preserved,
                  original_b_source_verified=True,independent_checkpoint_audit_passed=True,
                  note='Storage peak is estimated separately; end-of-stage storage and actual scheduler usage are measured. GitHub delivery is verified separately.')
    (root/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
