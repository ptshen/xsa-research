"""Document-preserving GPT-2 packing and reproducible independent batch streams."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import tiktoken
import torch

FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"


def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_documents(documents, output, source, val_fraction=.0005, split_seed=2026, shard_tokens=10000000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    enc = tiktoken.get_encoding("gpt2")
    manifest = {"format": 1, "source": source, "dtype": "<u2", "split_seed": split_seed,
                "val_document_probability": val_fraction, "test_split": None,
                "split_rule": "sha256(seed + colon + document id) / 2**256 < val_fraction",
                "packing": "source order within split; append EOS per document; no token shuffle; shard boundaries may split documents",
                "tokenizer": {"name": "gpt2", "package": "tiktoken", "version": importlib.metadata.version("tiktoken"), "eos": enc.eot_token},
                "splits": {s: {"documents": 0, "tokens": 0, "shards": []} for s in ("train", "val")}}
    buffers = {s: [] for s in manifest["splits"]}
    order_hash = hashlib.sha256()
    split_hash = {s: hashlib.sha256() for s in buffers}

    def flush(split, count):
        tokens = buffers[split][:count]
        del buffers[split][:count]
        info = manifest["splits"][split]
        path = output / f"{split}-{len(info['shards']):05d}.bin"
        np.asarray(tokens, dtype="<u2").tofile(path)
        info["shards"].append({"path": path.name, "tokens": len(tokens), "sha256": checksum(path)})

    with (output / "documents.jsonl").open("w") as index:
        for order, doc in enumerate(documents):
            identity = str(doc["id"])
            digest = hashlib.sha256(f"{split_seed}:{identity}".encode()).digest()
            split = "val" if int.from_bytes(digest, "big") / 2**256 < val_fraction else "train"
            ids = enc.encode_ordinary(doc["text"]) + [enc.eot_token]
            record = {"order": order, "id": identity, "split": split, "tokens": len(ids),
                      "text_sha256": hashlib.sha256(doc["text"].encode()).hexdigest()}
            line = json.dumps(record, sort_keys=True) + "\n"
            index.write(line); order_hash.update(line.encode()); split_hash[split].update(line.encode())
            info = manifest["splits"][split]
            info["documents"] += 1; info["tokens"] += len(ids)
            buffers[split].extend(ids)
            while len(buffers[split]) >= shard_tokens:
                flush(split, shard_tokens)
    for split in buffers:
        if buffers[split]:
            flush(split, len(buffers[split]))
        manifest["splits"][split]["document_order_sha256"] = split_hash[split].hexdigest()
    manifest["document_order_sha256"] = order_hash.hexdigest()
    manifest["documents_sha256"] = checksum(output / "documents.jsonl")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_manifest(path):
    path = Path(path)
    m = json.loads((path / "manifest.json").read_text())
    for info in m["splits"].values():
        if sum(s["tokens"] for s in info["shards"]) != info["tokens"]:
            raise ValueError("manifest token count mismatch")
        for shard in info["shards"]:
            p = path / shard["path"]
            if p.stat().st_size != 2 * shard["tokens"] or checksum(p) != shard["sha256"]:
                raise ValueError(f"checksum mismatch: {p}")
    if checksum(path / "documents.jsonl") != m["documents_sha256"]:
        raise ValueError("document index checksum mismatch")
    return m


class TokenData:
    def __init__(self, path, split, seq_len):
        self.path, self.seq_len = Path(path), seq_len
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.shards = [np.memmap(self.path / s["path"], dtype="<u2", mode="r")
                       for s in self.manifest["splits"][split]["shards"]]
        self.valid_counts = np.array([max(0, len(s) - seq_len) for s in self.shards], dtype=np.int64)
        self.cumulative = np.cumsum(self.valid_counts)
        if not len(self.cumulative) or self.cumulative[-1] == 0:
            raise ValueError(f"no {split} windows of context {seq_len}; prepare more real documents")

    def batch(self, locations):
        a = np.stack([np.asarray(self.shards[s][p:p + self.seq_len + 1], dtype=np.int64) for s,p in locations])
        return torch.from_numpy(a[:, :-1].copy()), torch.from_numpy(a[:, 1:].copy())

    def fixed_windows(self, count, seed):
        # Nonoverlapping windows, including inputs and targets, across all shards.
        counts = np.array([len(s) // (self.seq_len + 1) for s in self.shards])
        ends = np.cumsum(counts)
        if count > ends[-1]:
            raise ValueError(f"requested {count} held-out windows; only {ends[-1]} available")
        indices = np.random.default_rng(seed).choice(int(ends[-1]), size=count, replace=False)
        result = []
        for i in indices:
            shard = int(np.searchsorted(ends, i, side="right"))
            before = 0 if shard == 0 else ends[shard-1]
            result.append((shard, int(i-before) * (self.seq_len+1)))
        return result


class BatchStream:
    def __init__(self, data, seed):
        self.data, self.rng, self.sequences_seen = data, np.random.default_rng(seed), 0

    def next(self, count):
        indices = self.rng.integers(int(self.data.cumulative[-1]), size=count)
        locations = []
        for i in indices:
            s = int(np.searchsorted(self.data.cumulative, i, side="right"))
            locations.append((s, int(i - (self.data.cumulative[s-1] if s else 0))))
        x,y = self.data.batch(locations)
        digest = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
        self.sequences_seen += count
        return x,y,digest

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "sequences_seen": self.sequences_seen}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.sequences_seen = state["sequences_seen"]


def fineweb_documents(max_documents):
    """Download only consumed pinned parquet shards; no implicit whole-corpus load."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    files = sorted(f for f in HfApi().list_repo_files("HuggingFaceFW/fineweb", repo_type="dataset", revision=FINEWEB_REVISION)
                   if f.startswith("sample/100BT/") and f.endswith(".parquet"))
    count = 0
    for filename in files:
        local = hf_hub_download("HuggingFaceFW/fineweb", filename, repo_type="dataset", revision=FINEWEB_REVISION)
        for batch in pq.ParquetFile(local).iter_batches(batch_size=1024, columns=["id", "text"]):
            for doc in batch.to_pylist():
                yield doc
                count += 1
                if count >= max_documents:
                    return


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("fixture", "fineweb", "validate"):
        a = sub.add_parser(command); a.add_argument("--output", type=Path, required=True)
        if command == "fineweb":
            a.add_argument("--max-documents", type=int, required=True)
    args = p.parse_args()
    if args.command == "validate":
        m = validate_manifest(args.output)
    elif args.command == "fixture":
        docs = ({"id": f"fixture-{i}", "text": f"Record {i}. " + "The careful researcher checks each result, preserves the evidence, and repeats the measurement. " * 16} for i in range(800))
        m = prepare_documents(docs, args.output, {"kind": "fixture", "scientific_result": False}, val_fraction=.2, shard_tokens=100000)
    else:
        if args.max_documents < 1:
            p.error("max-documents must be positive")
        source = {"kind": "fineweb", "repo": "HuggingFaceFW/fineweb", "subset": "sample-100BT", "revision": FINEWEB_REVISION,
                  "max_documents": args.max_documents, "order": "lexicographic parquet filename, row order"}
        m = prepare_documents(fineweb_documents(args.max_documents), args.output, source)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
