"""One bounded allocated-GPU verification; all artifacts explicitly fixture-only."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from .benchmark import benchmark
from .collect import paired_summary, plot_curves
from .config import load_config
from .data import prepare_documents
from .diagnostics import diagnose
from .evaluate import evaluate
from .train import Trainer, software
from .initialize import create_initial_state


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    write(output/'hardware.json',software())
    with (output/'tests.txt').open('w') as log:
        result=subprocess.run([os.sys.executable,'-m','pytest','-q'],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'correctness failed; see {output}/tests.txt')
    data=Path('data/fixture')
    if not data.exists():
        docs=({'id':f'fixture-{i}','text':f'Record {i}. '+'The careful researcher checks each result, preserves the evidence, and repeats the measurement. '*16} for i in range(800))
        prepare_documents(docs,data,{'kind':'fixture','scientific_result':False},val_fraction=.2,shard_tokens=100000)
    cfg=load_config('configs/smoke.json')
    initial=output/'shared_initial.pt'
    create_initial_state(cfg,initial)
    cfg['initial_state_path']=str(initial)
    results=[]; hashes=[]
    for kind in ('sa','xsa'):
        c=copy.deepcopy(cfg);c['model']['attention_type']=kind
        run=output/kind
        trainer=Trainer(c,data,run,device='cuda')
        if kind=='xsa':
            trainer.run(stop_after=4)
            del trainer
            trainer=Trainer(c,data,run,device='cuda',resume=True)
        trainer.run()
        hashes.append((trainer.initial_hash,trainer.batch_hashes))
        if kind=='xsa':
            continued={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
            del trainer
            full=Trainer(c,data,output/'xsa-uninterrupted',device='cuda');full.run()
            max_error=max(float((continued[k]-v.detach().cpu()).abs().max()) for k,v in full.model.state_dict().items())
            for k,v in full.model.state_dict().items():
                torch.testing.assert_close(continued[k],v.detach().cpu(),atol=1e-6,rtol=1e-5)
            write(output/'cuda_resume.json',{'passed':True,'max_parameter_absolute_error':max_error,'atol':1e-6,'rtol':1e-5})
            del full
        else:
            del trainer
        r=evaluate(run/'checkpoint.pt',data,'cuda');results.append(r);write(run/'evaluation.json',r)
        write(run/'diagnostics.json',diagnose(run/'checkpoint.pt',data,'cuda',samples=4,context=32))
    assert hashes[0]==hashes[1], 'paired initial state or batch order differs'
    write(output/'pair_identity.json',{'passed':True,'initial_state_sha256':hashes[0][0],'batch_sha256':hashes[0][1]})
    write(output/'paired_results.json',paired_summary(results))
    plot_curves([output/'sa',output/'xsa'],output/'curves.png')
    torch.cuda.empty_cache()
    profile=benchmark(load_config('configs/pilot.json'),batch=8,warmup=3,iterations=12)
    write(output/'pilot_profile.json',profile)
    write(output/'completion.json',{'passed':True,'scientific_result':False,'total_wall_seconds':time.monotonic()-start,
                                   'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'budget_policy':'whole job hard walltime reserved in evidence/job_ledger.json'})
    print(f'Bounded verification finished: {output}',flush=True)


if __name__=='__main__':
    main()
