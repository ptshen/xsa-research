"""Count real model tensors on the meta device, without allocating large models."""
import json
from pathlib import Path

import torch

from xsa.config import load_config, tokens_per_update
from xsa.model import GPT, ModelConfig

results = []
for name in ("smoke", "pilot", "paper_0.7b", "paper_1.4b", "paper_2.7b"):
    cfg = load_config(f"configs/{name}.json")
    with torch.device("meta"):
        model = GPT(ModelConfig(**cfg["model"]))
    counts = model.parameter_counts()
    results.append({"config": name, **counts, "tokens_per_update": tokens_per_update(cfg),
                    "total_tokens": tokens_per_update(cfg) * cfg["train"]["updates"],
                    "estimated_checkpoint_bytes_fp32_adam": counts["total_unique"] * 12,
                    "estimated_two_checkpoint_bytes": counts["total_unique"] * 24,
                    "checkpoint_note": "weights plus two fp32 Adam moments; excludes RNG, batch hashes and serialization overhead"})
print(json.dumps(results, indent=2))
