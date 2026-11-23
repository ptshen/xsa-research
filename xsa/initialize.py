"""Create one immutable initialization file to load into both paired arms."""
import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .model import GPT, ModelConfig


def architecture(config):
    model = dict(config['model'])
    model.pop('attention_type')
    model.pop('alignment_tau', None)  # Operator setting does not change tensor shapes.
    return model


def create_initial_state(config, path):
    torch.manual_seed(config['seed'])
    model = GPT(ModelConfig(**config['model']))
    with Path(path).open('xb') as f:
        torch.save({'model':model.state_dict(),'architecture':architecture(config),'seed':config['seed']},f)
    return model


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int)
    a=p.parse_args();c=load_config(a.config)
    if a.seed is not None: c['seed']=a.seed
    model=create_initial_state(c,a.output)
    print(json.dumps(model.parameter_counts()))


if __name__=='__main__':
    main()
