"""Explicit configuration, optimizer-update accounting, and nanoGPT LR schedule."""
import json
import math
from pathlib import Path

from .model import ModelConfig


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    ModelConfig(**cfg["model"])
    t = cfg["train"]
    for key in ("updates", "microbatch_sequences", "accumulation_steps", "eval_windows", "checkpoint_interval", "eval_interval"):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= t["warmup_updates"] < t["updates"]:
        raise ValueError("warmup must finish before training does")
    if t["precision"] not in ("float32", "bfloat16"):
        raise ValueError("only float32 and bfloat16 are supported; no scaler is needed")
    return cfg


def tokens_per_update(cfg, world_size=1):
    t = cfg["train"]
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return world_size * t["microbatch_sequences"] * t["accumulation_steps"] * cfg["model"]["seq_len"]


def accumulation_for(global_tokens, microbatch, seq_len, world_size=1):
    denominator = microbatch * seq_len * world_size
    if denominator <= 0 or global_tokens % denominator:
        raise ValueError("global token batch must divide microbatch * context * world_size")
    return global_tokens // denominator


def learning_rate(update, train):
    """Zero-based update index, matching pinned nanoGPT (first update LR=0)."""
    peak, warmup, end = train["learning_rate"], train["warmup_updates"], train["updates"]
    floor = peak * train["min_lr_ratio"]
    if update < warmup:
        return peak * update / warmup
    if update >= end:
        return floor
    ratio = (update - warmup) / (end - warmup)
    return floor + 0.5 * (1 + math.cos(math.pi * ratio)) * (peak - floor)
