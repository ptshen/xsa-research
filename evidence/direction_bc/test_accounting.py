"""CPU-only accounting tests, outside the frozen numerical test suite."""
import copy
from datetime import datetime, timedelta

import pytest

from scripts.audit_bc_accounting import audit_accounting


def fixtures():
    jobs=[]; rows=[]
    stages=[('verify',None),('train','b010'),('verify',None),('train','b050'),('train','b100'),('train','c'),('collect',None)]
    for i,(kind,tag) in enumerate(stages):
        start=datetime(2026,11,24)+timedelta(minutes=10*i)
        seconds=300 if kind!='train' else (1200 if tag=='c' else 1740)
        jobs.append(dict(job_id=str(i+1),kind=kind,tag=tag,seconds=seconds))
        rows.append(dict(JobID=str(i+1),State='COMPLETED',ElapsedRaw='10',ExitCode='0:0',
                         Start=start.isoformat(),End=(start+timedelta(seconds=10)).isoformat(),
                         AllocTRES='cpu=8,gres/gpu=1,gres/gpu:a6000=1' if kind!='collect' else 'cpu=2',
                         TimelimitRaw=str(seconds//60),NodeList='ins093',Account='edu',Partition='short'))
    return jobs,rows


def test_complete_sequential_bounded_accounting():
    jobs,rows=fixtures(); result=audit_accounting(jobs,rows)
    assert result['gpu_seconds']==60 and result['verification_gpu_seconds']==20
    assert result['cpu_seconds']==10 and result['reserved_gpu_seconds']==7020
    assert result['sequential_gpu_allocations']


@pytest.mark.parametrize('key,value',[('State','RUNNING'),('ExitCode','1:0'),('AllocTRES','cpu=8,gres/gpu=2,gres/gpu:a6000=2'),('TimelimitRaw','999')])
def test_reject_unverified_or_mismatched_allocations(key,value):
    jobs,rows=fixtures(); rows[-2][key]=value
    with pytest.raises(ValueError): audit_accounting(jobs,rows)


def test_reject_missing_jobs_overlap_excess_and_incomplete_screen():
    jobs,rows=fixtures()
    with pytest.raises(ValueError): audit_accounting(jobs,rows[:-1])
    bad=copy.deepcopy(rows);bad[1].update(Start=rows[0]['Start'],End=rows[0]['End'])
    with pytest.raises(ValueError): audit_accounting(jobs,bad)
    bad=copy.deepcopy(jobs);bad[0]['seconds']=14400
    with pytest.raises(ValueError): audit_accounting(bad,rows)
    bad=copy.deepcopy(jobs);bad[4]['tag']='b050'
    with pytest.raises(ValueError): audit_accounting(bad,rows)


def test_failed_attempt_is_counted_when_successful_retry_exists():
    jobs,rows=fixtures()
    failed=copy.deepcopy(jobs[1]);failed['job_id']='failed';failed['seconds']=60
    failed_row=copy.deepcopy(rows[1]);failed_row.update(JobID='failed',State='FAILED',ExitCode='1:0',TimelimitRaw='1',Start='2026-11-23T23:59:00',End='2026-11-23T23:59:10')
    result=audit_accounting([failed]+jobs,[failed_row]+rows)
    assert result['gpu_seconds']==70 and result['reserved_gpu_seconds']==7080
    assert result['failed_or_cancelled_jobs']==['failed']


@pytest.mark.parametrize('index,seconds,budget',[(0,360,'verify'),(1,15000,'gpu'),(6,1860,'cpu')])
def test_matching_scheduler_request_still_must_fit_cumulative_budget(index,seconds,budget):
    jobs,rows=fixtures(); jobs[index]['seconds']=seconds; rows[index]['TimelimitRaw']=str(seconds//60)
    with pytest.raises(ValueError,match=f'{budget} budget exceeded'): audit_accounting(jobs,rows)
