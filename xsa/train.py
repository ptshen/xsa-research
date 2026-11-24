"""Single-GPU training with deterministic data streams and atomic resume."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
from pathlib import Path
import platform
import random
import signal
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from .config import load_config, learning_rate, tokens_per_update
from .data import BatchStream, TokenData, checksum, validate_manifest
from .model import GPT, ModelConfig, operator_metadata
from .initialize import architecture


def software():
    packages = {}
    for name in ("torch", "numpy", "tiktoken", "matplotlib", "pytest", "huggingface-hub", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def state_hash(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_save(state, path):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    if path.exists():
        os.replace(path, path.with_name("checkpoint.previous.pt"))
    os.replace(temp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_checkpoint(path, device="cpu"):
    # Only load trusted, self-produced checkpoints (optimizer/RNG require pickle).
    return torch.load(path, map_location=device, weights_only=False)


class Trainer:
    def __init__(self, cfg, data, output, device="cpu", resume=False):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("this launcher supports one GPU; distributed training is not implemented")
        self.cfg, self.device, self.output = cfg, torch.device(device), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output / "checkpoint.pt"
        if not resume and any(self.output.glob("checkpoint*.pt")):
            raise FileExistsError("existing checkpoint: use --resume or a new output directory")
        self.manifest = validate_manifest(data)
        if self.manifest["source"]["kind"] != cfg["data_kind"]:
            raise ValueError("data source kind differs from configuration; no synthetic fallback")
        self.manifest_hash = checksum(Path(data) / "manifest.json")
        random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg["seed"])
        self.model = GPT(ModelConfig(**cfg["model"])).to(self.device)
        if cfg.get('initial_state_path') and not resume:
            initial = load_checkpoint(cfg['initial_state_path'])
            if initial['architecture'] != architecture(cfg) or initial['seed'] != cfg['seed']:
                raise ValueError('initialization architecture or seed mismatch')
            self.model.load_state_dict(initial['model'])
        self.initial_hash = state_hash(self.model)
        t = cfg["train"]
        self.optimizer = torch.optim.AdamW([
            {"params": [p for p in self.model.parameters() if p.ndim >= 2], "weight_decay": t["weight_decay"]},
            {"params": [p for p in self.model.parameters() if p.ndim < 2], "weight_decay": 0.0}],
            lr=t["learning_rate"], betas=tuple(t["betas"]), eps=1e-8)
        self.stream = BatchStream(TokenData(data, "train", cfg["model"]["seq_len"]), cfg["data_seed"])
        self.validation = TokenData(data, "val", cfg["model"]["seq_len"])
        self.eval_locations = self.validation.fixed_windows(t["eval_windows"], cfg["eval_seed"])
        self.eval_hash = hashlib.sha256(json.dumps(self.eval_locations).encode()).hexdigest()
        self.update = self.tokens_seen = 0
        self.elapsed = 0.0
        self.batch_hashes = []
        self.stop_requested = False
        self.session = uuid.uuid4().hex
        if resume:
            self.restore()
        else:
            self.save()
        run_manifest = {"config": cfg, "parameter_counts": self.model.parameter_counts(), "initial_state_sha256": self.initial_hash,
                        "data_manifest_sha256": self.manifest_hash, "evaluation_windows_sha256": self.eval_hash,
                        "tokens_per_update": tokens_per_update(cfg), "total_tokens": tokens_per_update(cfg) * t["updates"],
                        "software": software(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "scientific_result": cfg["data_kind"] != "fixture", "session": self.session}
        (self.output / f"manifest-{self.session}.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
        print(json.dumps(run_manifest), flush=True)

    def autocast(self):
        if self.cfg["train"]["precision"] == "bfloat16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def log(self, event):
        event.update(update=self.update, tokens=self.tokens_seen, elapsed_seconds=self.elapsed, session=self.session)
        event.update(operator_metadata(self.cfg['model']))
        with (self.output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n"); f.flush()

    @torch.no_grad()
    def evaluate(self):
        was_training = self.model.training
        self.model.eval()
        losses = []
        try:
            for location in self.eval_locations:
                x,y = self.validation.batch([location])
                with self.autocast():
                    logits = self.model(x.to(self.device))
                    loss = F.cross_entropy(logits.float().flatten(0, 1), y.to(self.device).flatten(), reduction="sum")
                losses.append(float(loss) / y.numel())
        finally:
            self.model.train(was_training)
        ce = float(np.mean(losses))  # windows have equal length; token-weighted
        return {"type": "eval", "cross_entropy": ce, "perplexity": math.exp(ce), "window_losses": losses,
                "evaluated_tokens": len(losses) * self.cfg["model"]["seq_len"], "evaluation_windows_sha256": self.eval_hash}

    def save(self):
        state = {"format": 1, "config": self.cfg, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "update": self.update, "tokens_seen": self.tokens_seen, "elapsed_seconds": self.elapsed,
                 "scheduler": {"next_update": self.update}, "scaler": None,
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None},
                 "data_state": self.stream.state_dict(), "batch_hashes": self.batch_hashes,
                 "initial_state_sha256": self.initial_hash, "data_manifest_sha256": self.manifest_hash,
                 "evaluation_windows_sha256": self.eval_hash, "software": software()}
        atomic_save(state, self.checkpoint_path)

    def restore(self):
        try:
            state = load_checkpoint(self.checkpoint_path)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as e:
            previous = self.output / "checkpoint.previous.pt"
            print(f"Latest checkpoint unreadable ({e}); restoring {previous}", flush=True)
            state = load_checkpoint(previous)
            # Do not rotate a corrupt latest over the only valid previous on the
            # next save. Preserve the damaged artifact for diagnosis instead.
            if self.checkpoint_path.exists():
                os.replace(self.checkpoint_path, self.output / f'checkpoint.corrupt-{time.time_ns()}.pt')
        if state["config"] != self.cfg:
            raise ValueError("resume configuration mismatch; refusing to change schedule or experiment")
        if state["data_manifest_sha256"] != self.manifest_hash or state["evaluation_windows_sha256"] != self.eval_hash:
            raise ValueError("resume data or evaluation windows mismatch")
        self.model.load_state_dict(state["model"]); self.optimizer.load_state_dict(state["optimizer"])
        self.update, self.tokens_seen, self.elapsed = state["update"], state["tokens_seen"], state["elapsed_seconds"]
        if self.tokens_seen != self.update * tokens_per_update(self.cfg):
            raise ValueError("checkpoint token accounting mismatch")
        self.batch_hashes, self.initial_hash = state["batch_hashes"], state["initial_state_sha256"]
        self.stream.load_state_dict(state["data_state"])
        random.setstate(state["rng"]["python"]); np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if self.device.type == "cuda" and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        self.log({"type": "resume", "checkpoint": str(self.checkpoint_path)})

    def run(self, stop_after=None, max_seconds=None):
        t = self.cfg["train"]
        self.model.train()
        self.synchronize(); start = time.monotonic(); initial_elapsed = self.elapsed
        initial_tokens = self.tokens_seen
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        if self.update == 0:
            self.log(self.evaluate())
        while self.update < min(t["updates"], stop_after if stop_after is not None else t["updates"]):
            if self.stop_requested or (max_seconds is not None and time.monotonic() - start >= max_seconds):
                break
            lr = learning_rate(self.update, t)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            losses, hashes = [], []
            for _ in range(t["accumulation_steps"]):
                x,y,h = self.stream.next(t["microbatch_sequences"]); hashes.append(h)
                with self.autocast():
                    logits = self.model(x.to(self.device))
                    loss = F.cross_entropy(logits.float().flatten(0, 1), y.to(self.device).flatten())
                (loss / t["accumulation_steps"]).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), t["grad_clip"], error_if_nonfinite=True)
            self.optimizer.step()
            self.update += 1; self.tokens_seen += tokens_per_update(self.cfg)
            self.batch_hashes.append(hashlib.sha256("".join(hashes).encode()).hexdigest())
            self.synchronize(); self.elapsed = initial_elapsed + time.monotonic() - start
            self.log({"type": "train", "loss": float(np.mean(losses)), "lr": lr, "grad_norm": float(norm), "batch_sha256": self.batch_hashes[-1]})
            if self.update % t["eval_interval"] == 0 or self.update == t["updates"]:
                self.log(self.evaluate())
            if self.update % t["checkpoint_interval"] == 0:
                self.save()
        self.synchronize(); self.elapsed = initial_elapsed + time.monotonic() - start
        self.save()
        result = self.evaluate()
        runtime = {'session': self.session, 'attention': self.cfg['model']['attention_type'],
                   **operator_metadata(self.cfg['model']),
                   'alignment_tau': self.cfg['model'].get('alignment_tau'),
                   'session_tokens': self.tokens_seen - initial_tokens,
                   'session_seconds_including_final_evaluation_and_saves': time.monotonic() - start,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else None}
        (self.output / f'runtime-{self.session}.json').write_text(json.dumps(runtime, indent=2) + '\n')
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--attention", choices=("sa", "xsa", "alignment_cap", "value_penalty", "output_geometry")); p.add_argument("--seed", type=int)
    p.add_argument("--device", default="cuda"); p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int); p.add_argument("--max-seconds", type=float)
    p.add_argument('--initial-state')
    args = p.parse_args(); cfg = load_config(args.config)
    if args.attention:
        cfg["model"]["attention_type"] = args.attention
    if args.seed is not None:
        cfg.update(seed=args.seed, data_seed=args.seed + 1000, eval_seed=2017)
    if args.initial_state:
        cfg['initial_state_path'] = args.initial_state
    trainer = Trainer(cfg, args.data, args.output, args.device, args.resume)
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: setattr(trainer, "stop_requested", True))
    result = trainer.run(args.stop_after, args.max_seconds)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
