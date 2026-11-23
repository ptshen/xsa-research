"""Evaluate a trusted checkpoint on the exact saved held-out window protocol."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import TokenData, checksum, validate_manifest
from .model import GPT, ModelConfig
from .train import load_checkpoint, software


def load_model(checkpoint, device):
    state = load_checkpoint(checkpoint)
    model = GPT(ModelConfig(**state["config"]["model"])).to(device)
    model.load_state_dict(state["model"]); model.eval()
    return model, state


def comparison_hash(cfg, across_seeds=False):
    cfg = json.loads(json.dumps(cfg))
    ModelConfig(**cfg['model'])
    if cfg['model']['attention_type'] == 'alignment_cap':
        cfg['model'].pop('alignment_tau')
    cfg["model"].pop("attention_type")
    cfg.pop("label", None)
    if across_seeds:
        for key in ('seed', 'data_seed', 'initial_state_path'):
            cfg.pop(key, None)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def evaluate(checkpoint, data, device="cpu"):
    model, state = load_model(checkpoint, device)
    cfg = state["config"]
    validate_manifest(data)
    if checksum(Path(data) / "manifest.json") != state["data_manifest_sha256"]:
        raise ValueError("evaluation data differs from training manifest")
    tokens = TokenData(data, "val", cfg["model"]["seq_len"])
    locations = tokens.fixed_windows(cfg["train"]["eval_windows"], cfg["eval_seed"])
    evaluation_hash = hashlib.sha256(json.dumps(locations).encode()).hexdigest()
    if evaluation_hash != state["evaluation_windows_sha256"]:
        raise ValueError("evaluation windows differ from checkpoint")
    losses = []
    for loc in locations:
        x,y = tokens.batch([loc])
        ctx = torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16) if cfg["train"]["precision"] == "bfloat16" else nullcontext()
        with ctx:
            logits = model(x.to(device))
            nll = F.cross_entropy(logits.float().flatten(0, 1), y.to(device).flatten(), reduction="sum")
        losses.append(float(nll) / y.numel())
    ce = float(np.mean(losses))
    return {"attention": cfg["model"]["attention_type"], "alignment_tau": cfg['model'].get('alignment_tau'),
            "seed": cfg["seed"], "tokens": state["tokens_seen"],
            "update": state["update"], "cross_entropy": ce, "perplexity": math.exp(ce), "window_losses": losses,
            "evaluated_tokens": len(losses) * cfg["model"]["seq_len"], "checkpoint_sha256": checksum(checkpoint),
            "data_manifest_sha256": state["data_manifest_sha256"], "evaluation_windows_sha256": evaluation_hash,
            "initial_state_sha256": state["initial_state_sha256"], "comparison_config_sha256": comparison_hash(cfg),
            "study_config_sha256": comparison_hash(cfg, across_seeds=True),
            "scientific_result": cfg["data_kind"] != "fixture", "software": software()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--data", required=True)
    p.add_argument("--output", required=True); p.add_argument("--device", default="cuda")
    a = p.parse_args()
    result = evaluate(a.checkpoint, a.data, a.device)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
