import copy
import json
import numpy as np
import pytest
import torch
from xsa.config import load_config, tokens_per_update, learning_rate
from xsa.data import prepare_documents, TokenData, BatchStream, validate_manifest
from xsa.model import GPT, ModelConfig
from xsa.train import Trainer


def tiny_data(path):
    docs=[{'id':str(i),'text':f'Document {i}. The river flows through a green valley. '*12} for i in range(160)]
    prepare_documents(docs,path,source={'kind':'fixture'},val_fraction=.2,shard_tokens=1000)
    return path


def config():
    c=load_config('configs/smoke.json')
    c['model'].update(n_layers=1,d_model=16,n_heads=2,head_dim=8,seq_len=8,dropout=.1)
    c['train'].update(updates=4,microbatch_sequences=2,accumulation_steps=2,eval_interval=2,eval_windows=2,checkpoint_interval=2,warmup_updates=1)
    return c


def test_data_manifest_split_and_sampler(tmp_path):
    path=tiny_data(tmp_path/'data')
    m=validate_manifest(path)
    assert m['tokenizer']['name']=='gpt2' and m['tokenizer']['eos']==50256
    assert m['splits']['train']['documents']+m['splits']['val']['documents']==160
    d=TokenData(path,'train',8)
    a,b=BatchStream(d,123),BatchStream(d,123)
    xa,ya,ha=a.next(3); xb,yb,hb=b.next(3)
    assert ha==hb and torch.equal(xa,xb) and torch.equal(ya,yb)
    state=a.state_dict(); expected=a.next(3)
    b.load_state_dict(state); actual=b.next(3)
    assert expected[2]==actual[2]
    val=TokenData(path,'val',8)
    fixed=val.fixed_windows(5,seed=91)
    assert len(set(fixed))==5
    for shard in {s for s,_ in fixed}:
        starts=sorted(p for s,p in fixed if s==shard)
        assert all(y-x>=9 for x,y in zip(starts,starts[1:]))
    shard=path/m['splits']['train']['shards'][0]['path']
    with shard.open('r+b') as f: f.write(b'xx')
    with pytest.raises(ValueError,match='checksum'): validate_manifest(path)


def test_token_accounting_and_schedule():
    c=config(); t=c['train']
    assert tokens_per_update(c)==32
    assert tokens_per_update(c,world_size=2)==64
    assert learning_rate(0,t)==0
    assert learning_rate(1,t)==pytest.approx(t['learning_rate'])
    assert learning_rate(t['updates'],t)==pytest.approx(t['learning_rate']*.1)
    assert learning_rate(t['updates']+10,t)==pytest.approx(t['learning_rate']*.1)
    for name in ['paper_0.7b','paper_1.4b','paper_2.7b']:
        p=load_config(f'configs/{name}.json')
        assert tokens_per_update(p)==524288
        assert tokens_per_update(p)*p['train']['updates']==104857600000


def test_gradient_accumulation_equivalence():
    torch.manual_seed(8)
    cfg=ModelConfig(vocab_size=23,n_layers=1,d_model=8,n_heads=1,head_dim=8,seq_len=4)
    a=GPT(cfg).double(); b=copy.deepcopy(a)
    x=torch.randint(23,(4,4)); y=torch.randint(23,(4,4))
    loss=torch.nn.functional.cross_entropy(a(x).flatten(0,1),y.flatten()); loss.backward()
    for xx,yy in zip(x.chunk(2),y.chunk(2)):
        (torch.nn.functional.cross_entropy(b(xx).flatten(0,1),yy.flatten())/2).backward()
    for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p.grad,q.grad,atol=2e-12,rtol=2e-10)


@pytest.mark.parametrize('kind',['sa','xsa'])
def test_resume_matches_uninterrupted_and_eval_rng(tmp_path,kind):
    path=tiny_data(tmp_path/'data'); c=config(); c['model']['attention_type']=kind
    full=Trainer(c,path,tmp_path/'full',device='cpu'); full.run()
    split=Trainer(c,path,tmp_path/'split',device='cpu'); split.run(stop_after=2)
    assert (tmp_path/'split'/'checkpoint.previous.pt').exists()
    resumed=Trainer(c,path,tmp_path/'split',device='cpu',resume=True)
    before=resumed.stream.state_dict(); rng=torch.get_rng_state().clone()
    resumed.evaluate()
    assert before==resumed.stream.state_dict() and torch.equal(rng,torch.get_rng_state())
    resumed.run()
    assert resumed.update==full.update==4 and resumed.tokens_seen==128
    for key,value in full.model.state_dict().items(): torch.testing.assert_close(value,resumed.model.state_dict()[key],atol=0,rtol=0)
    assert full.batch_hashes==resumed.batch_hashes
    assert full.stream.state_dict()==resumed.stream.state_dict()
    assert full.optimizer.state_dict()['param_groups']==resumed.optimizer.state_dict()['param_groups']
    for pid,state in full.optimizer.state_dict()['state'].items():
        for key,value in state.items(): torch.testing.assert_close(value,resumed.optimizer.state_dict()['state'][pid][key],atol=0,rtol=0)
    changed=copy.deepcopy(c); changed['train']['learning_rate']*=2
    with pytest.raises(ValueError,match='configuration'): Trainer(changed,path,tmp_path/'split',device='cpu',resume=True)


def test_pair_identical_initial_state_and_batches(tmp_path):
    path=tiny_data(tmp_path/'data'); c=config()
    a=Trainer(c,path,tmp_path/'sa',device='cpu')
    c2=copy.deepcopy(c); c2['model']['attention_type']='xsa'
    b=Trainer(c2,path,tmp_path/'xsa',device='cpu')
    assert a.initial_hash==b.initial_hash
    for _ in range(3): assert a.stream.next(2)[2]==b.stream.next(2)[2]


def test_corrupt_latest_checkpoint_uses_preserved_previous(tmp_path):
    path=tiny_data(tmp_path/'data'); c=config()
    t=Trainer(c,path,tmp_path/'run',device='cpu'); t.run(stop_after=2)
    (tmp_path/'run'/'checkpoint.pt').write_bytes(b'corrupt checkpoint')
    restored=Trainer(c,path,tmp_path/'run',device='cpu',resume=True)
    assert restored.update==2
    restored.save()
    from xsa.train import load_checkpoint
    assert load_checkpoint(tmp_path/'run'/'checkpoint.previous.pt')['update']==2
    restored.run()
    assert restored.update==4


def test_source_kind_and_invalid_accumulation(tmp_path):
    from xsa.config import accumulation_for
    path=tiny_data(tmp_path/'data'); c=config(); c['data_kind']='fineweb'
    with pytest.raises(ValueError,match='source kind'): Trainer(c,path,tmp_path/'run',device='cpu')
    with pytest.raises(ValueError,match='divide'): accumulation_for(100,3,8)
    assert accumulation_for(524288,1,16384)==32


def test_shared_initial_checkpoint_loads_into_both_variants(tmp_path):
    from xsa.initialize import create_initial_state
    path=tiny_data(tmp_path/'data'); c=config()
    initial=tmp_path/'initial.pt';create_initial_state(c,initial)
    c['initial_state_path']=str(initial)
    a=Trainer(c,path,tmp_path/'sa',device='cpu')
    other=copy.deepcopy(c);other['model']['attention_type']='xsa'
    b=Trainer(other,path,tmp_path/'xsa',device='cpu')
    assert a.initial_hash==b.initial_hash
    wrong=copy.deepcopy(c);wrong['seed']+=1
    with pytest.raises(ValueError,match='initialization'): Trainer(wrong,path,tmp_path/'bad',device='cpu')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires allocated CUDA GPU')
def test_cuda_bfloat16_resume(tmp_path):
    path=tiny_data(tmp_path/'data'); c=config();c['model']['attention_type']='xsa';c['train']['precision']='bfloat16'
    full=Trainer(c,path,tmp_path/'full',device='cuda');full.run()
    interrupted=Trainer(c,path,tmp_path/'split',device='cuda');interrupted.run(stop_after=2)
    resumed=Trainer(c,path,tmp_path/'split',device='cuda',resume=True);resumed.run()
    for key,value in full.model.state_dict().items():
        torch.testing.assert_close(value,resumed.model.state_dict()[key],atol=1e-6,rtol=1e-5)
    assert full.batch_hashes==resumed.batch_hashes


def test_preparation_token_cap_preserves_complete_documents(tmp_path):
    docs=[{'id':str(i),'text':'A small document. '*10} for i in range(10)]
    m=prepare_documents(docs,tmp_path/'bounded',{'kind':'fixture'},val_fraction=.2,max_tokens=100)
    total=sum(s['tokens'] for s in m['splits'].values())
    assert 0<total<=100
    assert sum(s['documents'] for s in m['splits'].values())<10
    assert m['max_tokens']==100
