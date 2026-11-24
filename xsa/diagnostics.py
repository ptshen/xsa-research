"""Explicit bounded attention matrices for diagnostics only, never timing."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import TokenData
from .evaluate import load_model
from .model import operator_metadata


def diagnostic_stats(q, k, v, y, z, tau=None, eps=1e-12, strength=None, output_weight=None):
    q,k,v,y,z = [a.float() for a in (q,k,v,y,z)]
    t = v.size(-2)
    mask = torch.ones(t, t, dtype=torch.bool, device=v.device).tril()
    weights = ((q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))).masked_fill(~mask, -float("inf")).softmax(-1)
    vn = F.normalize(v, dim=-1, eps=1e-12)
    cos = vn @ vn.transpose(-1, -2)
    off_diagonal = ~torch.eye(t, dtype=torch.bool, device=v.device)
    causal_pairs = off_diagonal & mask

    def pair_mean(pair_mask):
        if not pair_mask.any():
            return [None] * v.size(1)
        return cos[..., pair_mask].mean((0, 2)).tolist()

    def alignment(a, value=v):
        valid = (a.norm(dim=-1) > 1e-6) & (value.norm(dim=-1) > 1e-6)
        c = F.cosine_similarity(a, value, dim=-1, eps=1e-12)
        count = valid.sum((0, 2))
        mean = (c * valid).sum((0, 2)) / count.clamp_min(1)
        return [float(x) if n else None for x,n in zip(mean, count)]

    extra = {}
    if strength is not None:
        positive = (F.normalize(v, dim=-1, eps=eps) @ F.normalize(v, dim=-1, eps=eps).transpose(-1,-2)).clamp_min(0)
        scores = (q @ k.transpose(-1,-2)) / math.sqrt(q.size(-1)) - strength * positive
        penalized = scores.masked_fill(~mask,-float('inf')).softmax(-1)
        extra['positive_similarity_weighted_before'] = (weights*positive).sum(-1).mean((0,2)).tolist()
        extra['positive_similarity_weighted_after'] = (penalized*positive).sum(-1).mean((0,2)).tolist()
        extra['applied_penalty_causal_mean'] = (strength*positive[...,mask]).mean((0,2)).tolist()
        extra['diagonal_attention_mass_before'] = weights.diagonal(dim1=-2,dim2=-1).mean((0,2)).tolist()
        weights = penalized
    if output_weight is not None:
        w=output_weight.float().reshape(-1,v.size(1),v.size(-1)).permute(1,0,2)
        written_y, written_v, written_z = [a @ w.transpose(-1,-2) for a in (y,v,z)]
        denominator=(v*(v @ (w.transpose(-1,-2) @ w))).sum(-1)
        extra['geometry_guard_fraction']=(denominator <= eps**2).float().mean((0,2)).tolist()
        samples=denominator.transpose(0,1).flatten(1)
        for name, quantile in [('min',0.),('q05',.05),('q50',.5),('q95',.95),('max',1.)]:
            extra[f'geometry_denominator_{name}']=samples.quantile(quantile,dim=1).tolist()
        extra['written_alignment_before']=alignment(written_y,written_v)
        extra['written_alignment_after']=alignment(written_z,written_v)
        extra['written_correction_norm']=(written_y-written_z).norm(dim=-1).mean((0,2)).tolist()
        extra['written_output_norm']=written_z.norm(dim=-1).mean((0,2)).tolist()
        extra['written_zero_output_fraction']=(written_z.norm(dim=-1)==0).float().mean((0,2)).tolist()
        extra['written_near_zero_output_fraction']=(written_z.norm(dim=-1)<=1e-6).float().mean((0,2)).tolist()
    if tau is not None:
        u = F.normalize(v, dim=-1, eps=eps)
        c = (y * u).sum(-1, keepdim=True)
        r = y - c * u
        b = tau / math.sqrt(1 - tau * tau) * r.norm(dim=-1, keepdim=True)
        extra['cap_activation_fraction'] = (c.abs() > b).float().mean((0, 2, 3)).tolist()
    return {**extra, "value_cosine_off_diagonal": pair_mean(off_diagonal),
            "value_cosine_causal_nonself": pair_mean(causal_pairs),
            "diagonal_attention_mass": weights.diagonal(dim1=-2, dim2=-1).mean((0, 2)).tolist(),
            "alignment_before": alignment(y), "alignment_after": alignment(z),
            "aggregate_norm": y.norm(dim=-1).mean((0, 2)).tolist(),
            "corrected_norm": z.norm(dim=-1).mean((0, 2)).tolist(),
            "removed_update_norm": (y-z).norm(dim=-1).mean((0, 2)).tolist(),
            "zero_output_fraction": (z.norm(dim=-1) == 0).float().mean((0, 2)).tolist(),
            "near_zero_output_fraction": (z.norm(dim=-1) <= 1e-6).float().mean((0, 2)).tolist()}


@torch.no_grad()
def diagnose(checkpoint, data, device, samples=4, context=128):
    model, state = load_model(checkpoint, device)
    if not 1 <= context <= min(model.cfg.seq_len, 512) or not 1 <= samples <= 1024:
        raise ValueError("bounded diagnostic requires context <=512 and samples <=1024")
    d = TokenData(data, "train", context)
    locations = d.fixed_windows(samples, seed=7017)
    collected = [[] for _ in model.blocks]
    for i, block in enumerate(model.blocks):
        block.attn.diagnostic_sink = lambda q,k,v,y,z,i=i,w=block.attn.c_proj.weight: collected[i].append(
            diagnostic_stats(q,k,v,y,z,model.cfg.alignment_tau,model.cfg.projection_eps,
                             model.cfg.similarity_lambda,w if model.cfg.attention_type=='output_geometry' else None))
    for loc in locations:
        x,_ = d.batch([loc]); model(x.to(device))
    layers = []
    for rows in collected:
        reduced = {}
        for key in rows[0]:
            reduced[key] = []
            for head in range(model.cfg.n_heads):
                vals = [r[key][head] for r in rows if r[key][head] is not None]
                reduced[key].append(sum(vals) / len(vals) if vals else None)
        layers.append(reduced)
    return {"samples": samples, "context": context, "split": "train", "seed": 7017,
            **operator_metadata(model.cfg), "locations": locations,
            "attention": model.cfg.attention_type, "alignment_tau": model.cfg.alignment_tau,
            "precision": "float32 diagnostic forward", "checkpoint_tokens": state["tokens_seen"],
            "scientific_result": state["config"]["data_kind"] != "fixture", "layers": layers,
            "averaging": "per head; equal weight per sequence and token; value pairs exclude self; causal metric also excludes masked pairs; alignment excludes norms <=1e-6; exact and near-zero fractions include all positions",
            "bc_averaging": "B penalties include all causal pairs including self; weighted similarities sum over keys then average queries. C denominator quantiles/min/max are computed per sequence, then averaged across sequences; written quantities are per head, before summing heads. B alignment_before uses the ordinary aggregate with the same current weights.",
            "paper_deviation": "paper uses 1024 random training sequences at length 2048, averaged over heads; this bounded command caps context at 512"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True);p.add_argument("--data",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--samples",type=int,default=4);p.add_argument("--context",type=int,default=32)
    a=p.parse_args(); r=diagnose(a.checkpoint,a.data,a.device,a.samples,a.context)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n')


if __name__=="__main__":
    main()
