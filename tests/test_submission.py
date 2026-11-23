import pytest
from xsa.submission import remaining_budget, check_submission


def test_budget_is_conservative_and_repeats_are_rejected():
    ledger={'jobs':[{'key':'first','job_id':'123','kind':'smoke','reserved_seconds':360}]}
    assert remaining_budget(ledger)==240
    with pytest.raises(ValueError,match='budget'): check_submission(ledger,'second','smoke',241,[])
    with pytest.raises(ValueError,match='existing'): check_submission(ledger,'first','smoke',120,[])
    with pytest.raises(ValueError,match='active'): check_submission(ledger,'second','smoke',120,['123'])
    check_submission(ledger,'second','smoke',240,[])
