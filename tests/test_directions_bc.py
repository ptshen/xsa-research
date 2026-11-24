import copy
import importlib.util
import math
import sys

import pytest
import torch
from torch.nn import functional as F

from xsa.model import GPT, ModelConfig, attention, project, value_penalty_attention, geometry_project
from xsa.evaluate import comparison_hash, evaluate
from xsa.initialize import architecture, create_initial_state
from xsa.train import Trainer
from test_pipeline import config, tiny_data


def b_reference(q, k, v, strength, eps=1e-12):
    # Independent per-query reference, with no future keys in the computation.
    rows = []
    for i in range(q.size(-2)):
        cosine = F.cosine_similarity(v[..., i:i+1, :], v[..., :i+1, :], dim=-1, eps=eps)
        score = (q[..., i:i+1, :] * k[..., :i+1, :]).sum(-1) / math.sqrt(q.size(-1))
        weights = (score - strength * cosine.clamp_min(0)).softmax(-1)
        rows.append((weights.unsqueeze(-1) * v[..., :i+1, :]).sum(-2))
    return torch.stack(rows, dim=-2)


def c_reference(y, v, weight, eps=1e-12):
    # Written-space calculation, independent of the implementation's Gram matrix.
    rows = []
    for h in range(y.size(1)):
        w = weight[:, h*y.size(-1):(h+1)*y.size(-1)]
        written_y, written_v = y[:, h] @ w.T, v[:, h] @ w.T
        denominator = written_v.square().sum(-1, keepdim=True)
        scale = (written_y * written_v).sum(-1, keepdim=True) / denominator.clamp_min(eps**2)
        rows.append(y[:, h] - torch.where(denominator > eps**2, scale, 0) * v[:, h])
    return torch.stack(rows, dim=1)


@pytest.mark.parametrize('strength', [0., .1, .5, 1.])
@pytest.mark.parametrize('chunk', [1, 3, 16])
def test_b_reference_chunks_outputs_gradients_and_causality(strength, chunk):
    torch.manual_seed(41)
    q, k, v = [torch.randn(1, 2, 7, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    actual = value_penalty_attention(q, k, v, strength, chunk_size=chunk)
    expected = b_reference(q, k, v, strength)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    for a, b in zip(torch.autograd.grad(actual.square().sum(), (q,k,v)),
                    torch.autograd.grad(expected.square().sum(), (q,k,v))):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(actual[:, :, 0], v[:, :, 0], atol=0, rtol=0)
    changed = [x.detach().clone() for x in (q,k,v)]
    for x in changed: x[:, :, 4:] += 12
    torch.testing.assert_close(value_penalty_attention(*changed, strength, chunk_size=chunk)[:, :, :4], actual[:, :, :4])
    if strength == 0:
        torch.testing.assert_close(actual, attention(q,k,v), atol=1e-12, rtol=1e-11)
        rerun=value_penalty_attention(q,k,v,0.,chunk_size=chunk)
        for a,b in zip(torch.autograd.grad(rerun.square().sum(),(q,k,v)),
                       torch.autograd.grad(attention(q,k,v).square().sum(),(q,k,v))):
            torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-10)


def test_b_penalizes_positive_including_diagonal_not_negative_similarity():
    q = k = torch.zeros(1,1,3,2,dtype=torch.float64)
    v = torch.tensor([[[[1.,0.],[-1.,0.],[1.,0.]]]],dtype=torch.float64)
    z = value_penalty_attention(q,k,v,1.,chunk_size=2)
    weights = torch.tensor([-1.,0.,-1.],dtype=torch.float64).softmax(0)
    torch.testing.assert_close(z[0,0,2], (weights[:,None]*v[0,0]).sum(0))
    assert z[0,0,2,0] < attention(q,k,v)[0,0,2,0]


def test_b_gradcheck_and_zero_values():
    torch.manual_seed(6)
    q,k,v = [torch.randn(1,1,3,2,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    assert torch.autograd.gradcheck(lambda q,k,v: value_penalty_attention(q,k,v,.5,chunk_size=2),(q,k,v))
    for magnitude in (0.,1e-14,1e-10):
        vv = (v.detach()*magnitude).requires_grad_()
        z = value_penalty_attention(q,k,vv,.5,chunk_size=2)
        assert torch.isfinite(z).all()
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(z.sum(),(q,k,vv)))


def test_c_reference_head_slicing_geometry_and_gradcheck():
    torch.manual_seed(32)
    y,v = [torch.randn(1,2,3,4,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    w = torch.randn(7,8,dtype=torch.float64,requires_grad=True)
    actual, expected = geometry_project(y,v,w), c_reference(y,v,w)
    torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-11)
    for a,b in zip(torch.autograd.grad(actual.square().sum(),(y,v,w),retain_graph=True),
                   torch.autograd.grad(expected.square().sum(),(y,v,w))):
        torch.testing.assert_close(a,b,atol=2e-11,rtol=2e-10)
    for h in range(2):
        wh = w[:,h*4:(h+1)*4]
        zy,vw = actual[:,h]@wh.T,v[:,h]@wh.T
        torch.testing.assert_close((zy*vw).sum(-1),torch.zeros(1,3,dtype=torch.float64),atol=2e-12,rtol=0)
    assert torch.autograd.gradcheck(geometry_project,(y,v,w))
    assert not torch.allclose(actual,geometry_project(y,v,w.detach()+.3))


def test_c_identity_collinear_rank_deficient_and_guard():
    torch.manual_seed(27)
    y,v = [torch.randn(1,2,5,4,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    w = torch.eye(8,dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(geometry_project(y,v,w),project(y,v),atol=1e-12,rtol=1e-11)
    torch.testing.assert_close(geometry_project(v,v,w),torch.zeros_like(v),atol=1e-12,rtol=0)
    for value in (torch.zeros_like(v),v.detach()*1e-14,v.detach()*1e-10):
        value.requires_grad_(); z=geometry_project(y,value,w)
        assert torch.isfinite(z).all()
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(z.sum(),(y,value,w),retain_graph=True))
    torch.testing.assert_close(geometry_project(y,torch.zeros_like(v),w),y,atol=0,rtol=0)
    null_w=torch.zeros_like(w,requires_grad=True)
    torch.testing.assert_close(geometry_project(y,v,null_w),y,atol=0,rtol=0)
    deficient=w.detach().clone(); deficient[:,::2]=0
    torch.testing.assert_close(geometry_project(y,v,deficient),c_reference(y,v,deficient))
    nearly_singular=torch.diag(torch.tensor([1.,1e-8,1e-12,0.]*2,dtype=torch.float64))
    torch.testing.assert_close(geometry_project(y,v,nearly_singular),c_reference(y,v,nearly_singular))


@pytest.mark.parametrize('strength', [-1.,float('nan'),float('inf'),None])
def test_invalid_b_strength(strength):
    with pytest.raises(ValueError,match='lambda'):
        ModelConfig(attention_type='value_penalty',similarity_lambda=strength)


def settings(kind):
    return dict(attention_type=kind,similarity_lambda=.5,query_chunk_size=3) if kind=='value_penalty' else dict(attention_type=kind)


@pytest.mark.parametrize('kind',['value_penalty','output_geometry'])
def test_new_model_causality_shared_parameters_and_initialization(tmp_path,kind):
    c=config(); data=tiny_data(tmp_path/'data')
    initial=tmp_path/'initial.pt'; create_initial_state(c,initial); c['initial_state_path']=str(initial)
    baseline=Trainer(c,data,tmp_path/'sa')
    changed=copy.deepcopy(c); changed['model'].update(settings(kind))
    trainer=Trainer(changed,data,tmp_path/'new')
    assert architecture(c)==architecture(changed)
    assert comparison_hash(c)==comparison_hash(changed)
    assert baseline.initial_hash==trainer.initial_hash
    assert baseline.model.parameter_counts()==trainer.model.parameter_counts()
    assert list(baseline.model.state_dict())==list(trainer.model.state_dict())
    # Width 2*8 != model width 12 catches output-block orientation mistakes.
    mc=ModelConfig(vocab_size=31,n_layers=2,d_model=12,n_heads=2,head_dim=8,seq_len=8,**settings(kind))
    model=GPT(mc).eval(); x=torch.randint(31,(2,8)); altered=x.clone(); altered[:,4:]=(altered[:,4:]+1)%31
    logits=model(x); torch.testing.assert_close(model(altered)[:,:4],logits[:,:4],atol=0,rtol=0)
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('kind',['value_penalty','output_geometry'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_bc_resume_and_setting_rejection(tmp_path,kind,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('allocated CUDA GPU required')
    c=config(); c['model'].update(settings(kind))
    if device=='cuda': c['train']['precision']='bfloat16'
    data=tiny_data(tmp_path/'data')
    full=Trainer(c,data,tmp_path/'full',device); full.run()
    split=Trainer(c,data,tmp_path/'split',device); split.run(stop_after=2)
    resumed=Trainer(c,data,tmp_path/'split',device,resume=True); resumed.run()
    assert full.batch_hashes==resumed.batch_hashes
    assert full.stream.state_dict()==resumed.stream.state_dict()
    for k,v in full.model.state_dict().items(): torch.testing.assert_close(v,resumed.model.state_dict()[k],atol=0,rtol=0)
    for pid,state in full.optimizer.state_dict()['state'].items():
        for k,v in state.items(): torch.testing.assert_close(v,resumed.optimizer.state_dict()['state'][pid][k],atol=0,rtol=0)
    changed=copy.deepcopy(c)
    changed['model']['similarity_lambda' if kind=='value_penalty' else 'projection_eps']=.25
    with pytest.raises(ValueError,match='configuration'): Trainer(changed,data,tmp_path/'split',device,resume=True)


@pytest.mark.parametrize('kind',['value_penalty','output_geometry'])
def test_cuda_bf16_matches_fp32_reductions_and_has_finite_gradients(kind):
    if not torch.cuda.is_available(): pytest.skip('allocated CUDA GPU required')
    torch.manual_seed(9)
    q,k,v=[torch.randn(2,3,9,8,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    w=torch.randn(10,24,device='cuda',requires_grad=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        z=value_penalty_attention(q,k,v,.5,chunk_size=4) if kind=='value_penalty' else geometry_project(q,v,w)
    ref=value_penalty_attention(q.float(),k.float(),v.float(),.5,chunk_size=4) if kind=='value_penalty' else geometry_project(q.float(),v.float(),w)
    torch.testing.assert_close(z,ref.to(torch.bfloat16),atol=0,rtol=0)
    inputs=(q,k,v) if kind=='value_penalty' else (q,v,w)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(z.float().square().sum(),inputs))


def test_sa_xsa_a_unchanged_against_frozen_source():
    spec=importlib.util.spec_from_file_location('bc_baseline','evidence/direction_bc/baseline_model.py')
    old=importlib.util.module_from_spec(spec); sys.modules[spec.name]=old; spec.loader.exec_module(old)
    for kind,tau in [('sa',None),('xsa',None),('alignment_cap',.25),('alignment_cap',.5),('alignment_cap',.75)]:
        kw=dict(vocab_size=31,n_layers=2,d_model=12,n_heads=2,head_dim=8,seq_len=8,attention_type=kind,alignment_tau=tau)
        torch.manual_seed(17); before=old.GPT(old.ModelConfig(**kw))
        torch.manual_seed(17); after=GPT(ModelConfig(**kw))
        x=torch.randint(31,(2,8)); a,b=before(x),after(x)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        a.square().mean().backward(); b.square().mean().backward()
        for p,q in zip(before.parameters(),after.parameters()): torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)


def test_bc_diagnostics_use_actual_weights_and_written_space():
    from xsa.diagnostics import diagnostic_stats
    q=k=torch.zeros(1,1,3,2)
    v=torch.tensor([[[[1.,0.],[-1.,0.],[1.,0.]]]])
    y=attention(q,k,v)
    z=value_penalty_attention(q,k,v,.5,chunk_size=2)
    b=diagnostic_stats(q,k,v,y,z,strength=.5)
    expected=torch.tensor([-.5,0.,-.5]).softmax(0)
    assert b['diagonal_attention_mass'][0] == pytest.approx((1+torch.tensor([0.,-.5]).softmax(0)[1]+expected[-1])/3)
    assert b['positive_similarity_weighted_after'][0] < b['positive_similarity_weighted_before'][0]
    assert b['applied_penalty_causal_mean'][0] > 0
    w=torch.tensor([[2.,1.],[0.,1.],[1.,0.]])
    z=geometry_project(y,v,w)
    c=diagnostic_stats(q,k,v,y,z,output_weight=w)
    assert c['geometry_guard_fraction']==[0.]
    assert c['written_alignment_after']==[None] # All nonzero values are collinear here.
    assert c['geometry_denominator_q50'][0]==pytest.approx(5.)


@pytest.mark.parametrize('kind',['value_penalty','output_geometry'])
def test_bc_metadata_and_comparison_rejects_unrelated_changes(tmp_path,kind):
    import json
    c=config(); c['model'].update(settings(kind)); data=tiny_data(tmp_path/'data')
    trainer=Trainer(c,data,tmp_path/'run'); trainer.run(stop_after=1)
    rows=[json.loads(s) for s in (tmp_path/'run'/'metrics.jsonl').read_text().splitlines()]
    record=evaluate(tmp_path/'run'/'checkpoint.pt',data)
    for row in rows+[record]:
        assert row['similarity_lambda']==c['model'].get('similarity_lambda')
        assert row['query_chunk_size']==c['model'].get('query_chunk_size')
        assert row['attention_backend']
    altered=copy.deepcopy(c); altered['train']['learning_rate']*=2
    assert comparison_hash(c)!=comparison_hash(altered)
    altered=copy.deepcopy(c); altered['model']['projection_eps']*=10
    assert comparison_hash(c)!=comparison_hash(altered)


def test_reject_irrelevant_settings_and_invalid_chunk():
    with pytest.raises(ValueError): ModelConfig(similarity_lambda=.5)
    for chunk in (0,-1,1.5,True):
        with pytest.raises(ValueError): ModelConfig(attention_type='value_penalty',similarity_lambda=.5,query_chunk_size=chunk)
