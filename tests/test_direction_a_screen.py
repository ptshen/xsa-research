import copy
import json
from pathlib import Path

import pytest

from xsa.direction_a import screen_summary, committed_training
from xsa.stage_submission import check_budget


def records():
    root = Path('evidence/pilot')
    rows = [json.loads((root / f'pilot-{kind}-s17/evaluation.json').read_text()) for kind in ('sa', 'xsa')]
    for tau in (.25, .5, .75):
        row = copy.deepcopy(rows[1]); row.update(attention='alignment_cap', alignment_tau=tau)
        rows.append(row)
    return rows


def test_screen_requires_exact_matching_complete_set():
    result = screen_summary(records())
    assert len(result['results']) == 5 and result['selection_data'] == 'validation'
    for key, value in [('tokens', 1), ('update', 1), ('seed', 29), ('comparison_config_sha256', 'wrong'),
                       ('data_manifest_sha256', 'wrong'), ('evaluation_windows_sha256', 'wrong'),
                       ('initial_state_sha256', 'wrong'), ('scientific_result', False)]:
        rows = records(); rows[-1][key] = value
        with pytest.raises(ValueError): screen_summary(rows)
    for rows in (records()[:-1], records() + [records()[-1]]):
        with pytest.raises(ValueError): screen_summary(rows)
    rows = records(); rows[-1]['alignment_tau'] = .25
    with pytest.raises(ValueError): screen_summary(rows)
    rows = records(); rows[-1]['software']['gpu'] = 'different GPU'
    with pytest.raises(ValueError): screen_summary(rows)


def test_stage_budget_and_duplicates():
    budget = {'gpu_seconds':7200, 'verification_gpu_seconds':300, 'cpu_seconds':900}
    jobs = [{'key':'verify1', 'kind':'verify', 'seconds':180, 'job_id':'123'}]
    check_budget(jobs, budget, 'train1', 'train', 1800, [])
    for key, kind, seconds, active in [('verify1','verify',120,[]), ('verify2','verify',121,[]),
                                       ('collect','collect',901,[]), ('train','train',1800,['123']),
                                       ('bad','train',0,[])]:
        with pytest.raises(ValueError): check_budget(jobs,budget,key,kind,seconds,active)
    reserved = jobs + [{'key':'long','kind':'train','seconds':5400,'job_id':'124'}]
    with pytest.raises(ValueError): check_budget(reserved,budget,'extra','train',1800,[])
    uncertain = jobs + [{'key':'uncertain','kind':'train','seconds':1800,'job_id':None}]
    with pytest.raises(ValueError): check_budget(uncertain,budget,'next','train',1800,[])


def test_training_collection_rolls_back_uncommitted_rows():
    rows = [{'type':'train','update':1,'loss':2}, {'type':'train','update':2,'loss':99},
            {'type':'resume','update':1}, {'type':'eval','update':1}, {'type':'train','update':2,'loss':3}]
    assert [r['loss'] for r in committed_training(rows)] == [2,3]
