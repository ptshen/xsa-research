import pytest
from xsa.pilot import require_completed_pair_member


def test_pilot_gate_requires_full_budget_and_fineweb():
    record={'tokens':67108864,'update':2048,'attention':'sa','scientific_result':True}
    require_completed_pair_member(record,'sa')
    for key,value in [('tokens',32768),('update',1),('attention','xsa'),('scientific_result',False)]:
        wrong=dict(record);wrong[key]=value
        with pytest.raises(ValueError):require_completed_pair_member(wrong,'sa')
