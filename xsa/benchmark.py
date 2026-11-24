"""Synchronized end-to-end pilot and block forward/backward measurements."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .config import load_config, tokens_per_update
from .model import GPT, ModelConfig, Block, project, operator_metadata
from .train import software


def benchmark(cfg, batch, warmup, iterations, compile_model=False, block_only=False, variants=None):
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires an allocated CUDA GPU")
    if min(batch, warmup, iterations) < 1:
        raise ValueError("positive batch, warmup and iteration counts required")
    results = []
    for variant in (variants or [("sa", None), ("xsa", None)]):
        torch.manual_seed(cfg["seed"])
        overrides = variant if isinstance(variant,dict) else dict(attention_type=variant[0],alignment_tau=variant[1])
        settings = dict(cfg['model']); settings.update(overrides)
        mc = ModelConfig(**settings)
        model = (Block(mc) if block_only else GPT(mc)).cuda().train()
        params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW([
            {'params':[p for p in model.parameters() if p.ndim>=2], 'weight_decay':cfg['train']['weight_decay']},
            {'params':[p for p in model.parameters() if p.ndim<2], 'weight_decay':0.}],
            lr=cfg["train"]["learning_rate"], betas=tuple(cfg["train"]["betas"]), eps=1e-8)
        x = torch.randn(batch, mc.seq_len, mc.d_model, device="cuda") if block_only else torch.randint(50257, (batch,mc.seq_len),device="cuda")
        targets = None if block_only else torch.randint(50257,(batch,mc.seq_len),device="cuda")
        compilation_seconds = 0.0
        if compile_model:
            model = torch.compile(model)

        def step(update=False):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                out = model(x)
                loss = out.float().square().mean() if block_only else F.cross_entropy(out.float().flatten(0,1),targets.flatten())
            loss.backward()
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["train"]["grad_clip"],error_if_nonfinite=True)
                optimizer.step()

        start=time.perf_counter(); step(True);torch.cuda.synchronize()
        first_step_seconds=time.perf_counter()-start
        if compile_model: compilation_seconds=first_step_seconds
        for _ in range(warmup): step()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        durations=[]
        for _ in range(iterations):
            start=time.perf_counter();step();torch.cuda.synchronize();durations.append(time.perf_counter()-start)
        forward_backward_seconds=sum(durations)/len(durations)
        start=time.perf_counter()
        for _ in range(iterations): optimizer.step()
        torch.cuda.synchronize();optimizer_seconds=(time.perf_counter()-start)/iterations
        # Match the actual pilot accumulation; optimizer runs once per update.
        accumulation = cfg["train"]["accumulation_steps"] * cfg["train"]["microbatch_sequences"] / batch
        estimated_update_seconds=accumulation*forward_backward_seconds+optimizer_seconds
        full_update_seconds=[]
        if not block_only and accumulation.is_integer():
            for _ in range(3):
                torch.cuda.synchronize();start=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                for _ in range(int(accumulation)):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(x)
                        loss=F.cross_entropy(logits.float().flatten(0,1),targets.flatten())/accumulation
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['grad_clip'],error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize();full_update_seconds.append(time.perf_counter()-start)
            estimated_update_seconds=sum(full_update_seconds)/len(full_update_seconds)
        # An extra untimed step identifies real dispatch, independently of config labels.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            step()
        dispatched = sorted({event.key for event in trace.key_averages()
                             if 'scaled_dot_product' in event.key or event.key in ('aten::bmm','aten::_softmax')})
        results.append({**operator_metadata(mc),"dispatched_operators":dispatched,"scope":"attention_plus_ffn_block" if block_only else "full_model",
                        "batch":batch,"context":mc.seq_len,"parameters":params,"precision":"bfloat16; fp32 operator reductions",
                        "compiled":compile_model,
                        "first_step_seconds":first_step_seconds,"compilation_including_first_step_seconds":compilation_seconds,
                        "warmup_steps":warmup,"measured_steps":iterations,"step_seconds":durations,
                        "mean_forward_backward_seconds":forward_backward_seconds,"optimizer_step_seconds":optimizer_seconds,
                        "forward_backward_tokens_per_second":batch*mc.seq_len/forward_backward_seconds,
                        "estimated_config_update_seconds":estimated_update_seconds,
                        "measured_full_update_seconds":full_update_seconds,
                        "estimated_run_gpu_hours":estimated_update_seconds*cfg["train"]["updates"]/3600 if not block_only else None,
                        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                        "timing_excludes":"data I/O, evaluation, checkpoints; no diagnostic matrices; compilation separately reported"})
        del model,optimizer,x,targets
        torch.cuda.empty_cache()
    # Directly measure the numerical effect of the deliberate precision change.
    torch.manual_seed(17)
    y=torch.randn(4,6,64,64,device="cuda",dtype=torch.bfloat16);v=torch.randn_like(y)
    native=project(y,v,precision="native");safe=project(y,v,precision="float32")
    ref=project(y.float(),v.float())
    def alignment(z):
        return float(((z.float()*v.float()).sum(-1).abs()/(z.float().norm(dim=-1)*v.float().norm(dim=-1)).clamp_min(1e-12)).max())
    return {"config":cfg,"software":software(),"results":results,"scientific_result":False,
            "projection_precision_effect":{"native_max_abs_error_vs_fp32":float((native.float()-ref).abs().max()),
                                           "safe_max_abs_error_vs_fp32":float((safe.float()-ref).abs().max()),
                                           "native_max_abs_cosine":alignment(native),"safe_max_abs_cosine":alignment(safe)},
            "estimate_assumptions":"same GPU and microbatch as measured; linear accumulation extrapolation; add overhead/contingency; pilot throughput is not a billion-parameter measurement"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/pilot.json");p.add_argument("--output",required=True)
    p.add_argument("--batch",type=int,default=8);p.add_argument("--warmup",type=int,default=3);p.add_argument("--iterations",type=int,default=12)
    p.add_argument("--compile",action="store_true");p.add_argument("--block-only",action="store_true")
    a=p.parse_args();r=benchmark(load_config(a.config),a.batch,a.warmup,a.iterations,a.compile,a.block_only)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)


if __name__=="__main__":
    main()
