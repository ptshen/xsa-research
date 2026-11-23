"""Exercise the same process signal used by the Slurm walltime warning."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from xsa.train import load_checkpoint
from test_pipeline import config, tiny_data


def test_usr1_saves_a_resumable_completed_update(tmp_path):
    data=tiny_data(tmp_path/'data'); cfg=config();cfg['train']['updates']=1000
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(cfg))
    output=tmp_path/'run'; env=os.environ.copy();env['OMP_NUM_THREADS']='1'
    with (tmp_path/'process.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-m','xsa.train','--config',str(config_path),'--data',str(data),
                                  '--output',str(output),'--device','cpu'],stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            deadline=time.monotonic()+25
            metrics=output/'metrics.jsonl'
            while not metrics.exists() or '"type": "train"' not in metrics.read_text():
                assert process.poll() is None, (tmp_path/'process.log').read_text()
                assert time.monotonic()<deadline, 'trainer did not reach update'
                time.sleep(.02)
            process.send_signal(signal.SIGUSR1)
            assert process.wait(timeout=15)==0, (tmp_path/'process.log').read_text()
        finally:
            if process.poll() is None: process.kill();process.wait()
    state=load_checkpoint(output/'checkpoint.pt')
    assert 1<=state['update']<1000
    assert state['tokens_seen']==state['update']*32
    assert state['data_state']['sequences_seen']==state['update']*4
