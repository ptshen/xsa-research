"""Independent artifact/checkpoint audit; allocated CPU node only, no training."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

root=Path('evidence/direction_bc')
manifest=json.loads(Path('DIRECTIONS_BC_MANIFEST.json').read_text())
summary=json.loads((root/'results/summary.json').read_text())
assert len(summary['results'])==9
reference=None
audited=[]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): digest.update(block)
    return digest.hexdigest()


for identity,row in zip(manifest['identities'],summary['results']):
    run=Path(identity['run'])
    evaluation=json.loads((run/'evaluation.json').read_text())
    ce=float(np.mean(evaluation['window_losses']))
    assert len(evaluation['window_losses'])==64 and evaluation['evaluated_tokens']==32768
    assert abs(ce-row['cross_entropy'])<1e-12
    assert math.isclose(math.exp(ce),row['perplexity'],rel_tol=1e-12)
    current=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert current['update']==2048 and current['tokens_seen']==67108864
    assert sha(run/'checkpoint.pt')==evaluation['checkpoint_sha256']==identity['checkpoint_sha256']
    config=current['config']; model=dict(config['model'])
    method=model.pop('attention_type')
    for key in ('alignment_tau','similarity_lambda','query_chunk_size'):
        assert model.pop(key,None)==row.get(key)
    assert method==row['attention']
    common=dict(config,model=model); common.pop('label',None)
    shared=(common,current['batch_hashes'],current['data_state'],current['initial_state_sha256'],
            current['data_manifest_sha256'],current['evaluation_windows_sha256'],current['software'])
    if reference is None: reference=shared
    assert shared==reference
    for checkpoint in ('checkpoint.pt','checkpoint.previous.pt'):
        state=current if checkpoint=='checkpoint.pt' else torch.load(run/checkpoint,map_location='cpu',weights_only=False)
        assert state['config']==config
        assert 0<state['update']<=2048 and state['tokens_seen']==state['update']*32768
        assert len(state['batch_hashes'])==state['update']
        assert state['batch_hashes']==current['batch_hashes'][:state['update']]
        assert all(torch.isfinite(t).all() for t in state['model'].values())
        for values in state['optimizer']['state'].values():
            assert all(torch.isfinite(t).all() for t in values.values() if isinstance(t,torch.Tensor))
        assert set(state['rng'])=={'python','numpy','torch','cuda'}
        assert state['data_state'] and state['evaluation_windows_sha256']==current['evaluation_windows_sha256']
        if checkpoint.endswith('previous.pt'): assert sha(run/checkpoint)==identity['previous_checkpoint_sha256']
        del state
    events=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    committed={}
    for event in events:
        if event['type']=='resume': committed={i:r for i,r in committed.items() if i<=event['update']}
        if event['type']=='train': committed[event['update']]=event
    assert sorted(committed)==list(range(1,2049))
    assert [committed[i]['batch_sha256'] for i in range(1,2049)]==current['batch_hashes']
    for i,r in committed.items():
        assert r['tokens']==i*32768 and math.isfinite(r['loss']) and math.isfinite(r['grad_norm'])
    diagnostic=json.loads((run/'diagnostics.json').read_text())
    assert diagnostic['checkpoint_tokens']==67108864
    assert (diagnostic['samples'],diagnostic['context'],diagnostic['seed'])==(4,128,7017)
    assert len(diagnostic['layers'])==6
    for layer in diagnostic['layers']:
        assert all(len(v)==6 for v in layer.values())
        assert all(v is None or math.isfinite(v) for values in layer.values() for v in values)
        assert all(0<=v<=1 for k,values in layer.items() if 'fraction' in k or 'diagonal_attention_mass' in k for v in values)
    audited.append({'run':str(run),'latest_and_previous_valid':True,'full_batch_history_verified':True,
                    'config_and_software_match':True,'metrics_recomputed':True,'diagnostics_checked':True})
    del current

for row,identity in zip(summary['results'],manifest['identities']):
    ev=json.loads((Path(identity['run'])/'evaluation.json').read_text())
    for index,kind in ((0,'sa'),(1,'xsa')):
        baseline=json.loads((Path(manifest['identities'][index]['run'])/'evaluation.json').read_text())
        differences=np.asarray(ev['window_losses'])-np.asarray(baseline['window_losses'])
        assert abs(float(differences.mean())-row[f'minus_{kind}'])<1e-12
        samples=np.random.default_rng(2026).choice(differences,(2000,64),replace=True).mean(1)
        np.testing.assert_allclose(np.quantile(samples,[.025,.975]),row[f'minus_{kind}_window_bootstrap_95ci'],atol=1e-12,rtol=0)

proof=json.loads((root/'verification.json').read_text())
assert sha('runs/initial/pilot-s17.pt')==proof['initial_file_sha256']
assert sha('data/pilot-v2/manifest.json')==reference[4]
for path,digest in proof['source_sha256'].items(): assert sha(path)==digest
result={'passed':True,'runs':audited,'pairs_of_checkpoints_loaded':9,
        'comparisons_and_bootstrap_reproduced':True,'frozen_sources_match':True,
        'resource_accounting':'separate final scheduler audit after collector exits'}
(root/'independent_checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
