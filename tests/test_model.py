import copy
import pytest
import torch
from xsa.model import ModelConfig, GPT, attention, project, rope


def test_reference_sdpa_and_gradients():
    torch.manual_seed(7)
    q, k, v = [torch.randn(2, 3, 5, 4, requires_grad=True) for _ in range(3)]
    for kind in ('sa', 'xsa'):
        ref = attention(q, k, v, kind, backend='reference')
        fast = attention(q, k, v, kind, backend='sdpa')
        torch.testing.assert_close(ref, fast, atol=3e-7, rtol=2e-5)
        a = torch.autograd.grad(ref.square().sum(), (q,k,v), retain_graph=True)
        b = torch.autograd.grad(fast.square().sum(), (q,k,v), retain_graph=True)
        for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=3e-6,rtol=3e-5)


def test_projection_geometry_and_degeneracies():
    torch.manual_seed(9)
    y, v = torch.randn(2,3,7,8), torch.randn(2,3,7,8)
    z = project(y,v)
    assert ((z*v).sum(-1).abs() < 2e-6).all()
    torch.testing.assert_close(project(torch.tensor([1.,0.]),torch.tensor([0.,2.])),torch.tensor([1.,0.]))
    torch.testing.assert_close(project(v,v),torch.zeros_like(v),atol=8e-7,rtol=0)
    torch.testing.assert_close(project(y,torch.zeros_like(v)),y)
    # normalize's clamp means values below epsilon are not fully projected out.
    small = torch.tensor([1e-14,0.],dtype=torch.float64)
    torch.testing.assert_close(project(torch.tensor([1.,0.],dtype=torch.float64),small,eps=1e-12),torch.tensor([.9999,0.],dtype=torch.float64))


def test_gradcheck():
    q,k,v = [torch.randn(1,1,3,4,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    assert torch.autograd.gradcheck(lambda q,k,v: attention(q,k,v,'xsa',backend='reference'),(q,k,v))


@pytest.mark.parametrize('kind',['sa','xsa'])
def test_first_position_and_causality(kind):
    q,k,v = [torch.randn(1,2,5,4) for _ in range(3)]
    y = attention(q,k,v,kind)
    target = v[:,:,0] if kind=='sa' else torch.zeros_like(v[:,:,0])
    torch.testing.assert_close(y[:,:,0],target,atol=5e-7,rtol=1e-5)
    q2,k2,v2 = [t.clone() for t in (q,k,v)]
    for t in (q2,k2,v2): t[:,:,3:]+=20
    torch.testing.assert_close(attention(q2,k2,v2,kind)[:,:,:3],y[:,:,:3])


def test_rope_layout_and_offsets():
    x = torch.tensor([[[[1.,2.,3.,4.]]]])
    torch.testing.assert_close(rope(x),x)
    # Adjacent pairs; angles at position one are 1 and .01.
    a=torch.tensor([1.,.01])
    even=x[...,::2]*a.cos()-x[...,1::2]*a.sin()
    odd=x[...,::2]*a.sin()+x[...,1::2]*a.cos()
    expected=torch.stack((even,odd),-1).flatten(-2)
    torch.testing.assert_close(rope(x,offset=1),expected)


def test_matched_parameters_unequal_width_and_backward():
    cfg=ModelConfig(vocab_size=31,n_layers=2,d_model=12,n_heads=2,head_dim=8,seq_len=8)
    torch.manual_seed(3); sa=GPT(cfg)
    cfg2=copy.deepcopy(cfg); cfg2.attention_type='xsa'
    torch.manual_seed(3); xsa=GPT(cfg2)
    assert sa.state_dict().keys()==xsa.state_dict().keys()
    for a,b in zip(sa.parameters(),xsa.parameters()): assert torch.equal(a,b)
    assert sa.lm_head.weight is sa.wte.weight
    assert sa.parameter_counts()==xsa.parameter_counts()
    x=torch.randint(31,(2,8))
    for m in (sa,xsa):
        logits=m(x)
        assert logits.shape==(2,8,31)
        logits.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        m.eval(); altered=x.clone(); altered[:,5:]=0
        torch.testing.assert_close(m(x)[:,:5],m(altered)[:,:5])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires allocated CUDA GPU')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_cuda_projection(dtype):
    torch.manual_seed(3)
    y=torch.randn(2,3,9,64,device='cuda',dtype=dtype)
    v=torch.randn_like(y)
    z=project(y,v,precision='float32')
    rel=(z.float()*v.float()).sum(-1).abs()/(z.float().norm(dim=-1)*v.float().norm(dim=-1)).clamp_min(1e-12)
    assert rel.max() < (1e-6 if dtype==torch.float32 else .002)
    q,k=[torch.randn_like(v) for _ in range(2)]
    torch.testing.assert_close(attention(q,k,v,'xsa',backend='sdpa'),attention(q,k,v,'xsa',backend='reference'),atol=.02 if dtype==torch.bfloat16 else 2e-6,rtol=.03 if dtype==torch.bfloat16 else 2e-5)
