import numpy as np
import pytest
import torch
from xsa.diagnostics import diagnostic_stats
from xsa.collect import paired_summary
from xsa.downstream import score_tokens
from xsa.model import ModelConfig, GPT


def test_diagnostics_known_geometry():
    v=torch.tensor([[[[1.,0.],[0.,1.]]]])
    q=k=torch.zeros_like(v)
    y=torch.tensor([[[[1.,0.],[.5,.5]]]])
    z=torch.tensor([[[[0.,0.],[.5,0.]]]])
    s=diagnostic_stats(q,k,v,y,z)
    assert s['value_cosine_off_diagonal']==[0.]
    assert s['diagonal_attention_mass']==[.75]
    assert s['alignment_after']==[0.]
    assert s['zero_output_fraction']==[.5]


def record(kind,seed=17):
    return {'attention':kind,'seed':seed,'tokens':128,'data_manifest_sha256':'data','evaluation_windows_sha256':'eval',
            'initial_state_sha256':'same','comparison_config_sha256':'cfg','study_config_sha256':'study','scientific_result':False,
            'window_losses':[2.,3.,4.] if kind=='sa' else [1.9,2.9,3.9],'cross_entropy':3. if kind=='sa' else 2.9}


def test_paired_summary_and_mismatch_guards():
    result=paired_summary([record('sa'),record('xsa')])
    assert result['mean_xsa_minus_sa']==pytest.approx(-.1)
    assert result['seed_standard_error'] is None
    assert result['pairs'][0]['window_bootstrap_95ci']==pytest.approx([-.1,-.1])
    bad=record('xsa');bad['tokens']+=1
    with pytest.raises(ValueError,match='tokens'): paired_summary([record('sa'),bad])
    bad=record('xsa');bad['evaluation_windows_sha256']='wrong'
    with pytest.raises(ValueError,match='evaluation'): paired_summary([record('sa'),bad])


def test_downstream_continuation_scoring():
    torch.manual_seed(1)
    m=GPT(ModelConfig(vocab_size=31,n_layers=1,d_model=8,n_heads=1,head_dim=8,seq_len=8)).eval()
    context,continuation=[2,3,4],[5,6,7]
    score,greedy=score_tokens(m,context,continuation,'cpu')
    logits=m(torch.tensor([[2,3,4,5,6]]))[0,2:].log_softmax(-1)
    expected=logits.gather(1,torch.tensor(continuation)[:,None]).sum().item()
    assert score==pytest.approx(expected,abs=1e-6)
    assert greedy==bool((logits.argmax(-1)==torch.tensor(continuation)).all())
    with pytest.raises(ValueError,match='continuation'): score_tokens(m,[2],list(range(10)),'cpu')


def test_resumed_curve_discards_abandoned_future_rows():
    from xsa.collect import committed_evaluations
    rows=[{'type':'eval','update':0},{'type':'eval','update':4},{'type':'eval','update':8},
          {'type':'resume','update':4},{'type':'eval','update':6}]
    assert [r['update'] for r in committed_evaluations(rows,6)]==[0,4,6]


def test_summary_rejects_different_budgets_across_seeds():
    records=[record('sa'),record('xsa'),record('sa',29),record('xsa',29)]
    for r in records[2:]:r['tokens']=256
    with pytest.raises(ValueError,match='across seeds'): paired_summary(records)
