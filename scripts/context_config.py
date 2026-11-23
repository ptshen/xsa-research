"""Materialize a context sweep member without changing either token budget."""
import argparse
import json
from pathlib import Path

from xsa.config import accumulation_for, load_config, tokens_per_update

p=argparse.ArgumentParser()
p.add_argument('--context',type=int,choices=[512,1024,2048,4096,8192,16384],required=True)
p.add_argument('--microbatch',type=int,default=1);p.add_argument('--output',required=True)
a=p.parse_args()
cfg=load_config('configs/paper_1.4b.json')
cfg['model']['seq_len']=a.context;cfg['train']['microbatch_sequences']=a.microbatch
cfg['train']['accumulation_steps']=accumulation_for(524288,a.microbatch,a.context)
assert tokens_per_update(cfg)*cfg['train']['updates']==104857600000
Path(a.output).write_text(json.dumps(cfg,indent=2)+'\n')
