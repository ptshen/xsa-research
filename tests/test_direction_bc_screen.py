import copy
import json
from pathlib import Path

import pytest

from xsa.direction_bc import screen_summary, allocation_plan, validate_record_config
from xsa.bc_submission import check_prior_states, check_training_target


def records():
    root=Path('evidence/direction_a/results')
    rows=[json.loads(p.read_text()) for p in sorted(root.glob('*/evaluation.json'))]
    baseline=next(r for r in rows if r['attention']=='sa')
    for strength in (.1,.5,1.):
        r=copy.deepcopy(baseline); r.update(attention='value_penalty',similarity_lambda=strength,query_chunk_size=128)
        rows.append(r)
    r=copy.deepcopy(baseline); r.update(attention='output_geometry'); rows.append(r)
    return rows


def test_nine_method_summary_checks_complete_matched_metrics():
    rows=records(); assert len(screen_summary(rows)['results'])==9
    for key,value in [('tokens',123),('perplexity',1),('initial_state_sha256','wrong'),('query_chunk_size',64)]:
        bad=copy.deepcopy(rows); bad[-2][key]=value
        with pytest.raises(ValueError): screen_summary(bad)
    with pytest.raises(ValueError): screen_summary(rows[:-1])
    with pytest.raises(ValueError): screen_summary(rows[:-1]+[rows[0]])


def test_gate_resource_plan_includes_overhead_temporary_storage_and_caps():
    plan=allocation_plan([600,700,700,650],7_200_000_000,360_000_000)
    assert sum(plan['train_seconds'])+600 <= 14400
    assert plan['project_peak_bytes'] > 7_200_000_000 + 8*360_000_000
    assert plan['fits']
    assert not allocation_plan([4000]*4,7_200_000_000,360_000_000)['fits']
    assert not allocation_plan([600]*4,11*2**30,360_000_000)['fits']


def test_scheduler_and_target_guards(tmp_path):
    check_prior_states(['1'],'JobID|State|ElapsedRaw|ExitCode\n1|COMPLETED|42|0:0\n')
    for state in ('RUNNING','PENDING','COMPLETING',''):
        with pytest.raises(ValueError): check_prior_states(['1'],f'JobID|State|ElapsedRaw|ExitCode\n1|{state}|0|0:0\n')
    with pytest.raises(ValueError): check_prior_states(['1'],'JobID|State|ElapsedRaw|ExitCode\n')
    run=tmp_path/'run'; check_training_target(run,False)
    run.mkdir(); (run/'checkpoint.pt').write_text('preserved')
    with pytest.raises(ValueError): check_training_target(run,False)
    check_training_target(run,True)
    with pytest.raises(ValueError): check_training_target(tmp_path/'absent',True)


def test_record_method_settings_must_match_checkpoint():
    record=records()[-2]
    cfg={'model':{'attention_type':'value_penalty','similarity_lambda':1.,'query_chunk_size':128}}
    validate_record_config(record,cfg)
    altered=copy.deepcopy(cfg); altered['model']['similarity_lambda']=.1
    with pytest.raises(ValueError): validate_record_config(record,altered)
