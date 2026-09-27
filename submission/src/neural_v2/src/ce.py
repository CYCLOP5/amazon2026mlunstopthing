'mean-pooled hard-pair cross-encoder with grouped validation'

import argparse as ap
import gc
import math
import os
import shutil as sh
import socket
import time
from pathlib import Path as path

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from transformers import AutoModel as automodel, AutoTokenizer as autotokenizer
from transformers.modeling_outputs import SequenceClassifierOutput as result

import infer
import neural

kind = "mean-pooled-cross-encoder-v1"


class model(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.encoder = automodel.from_pretrained(base, add_pooling_layer=False, attn_implementation="sdpa", local_files_only=True)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, **batch):
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return result(logits=self.head(pooled))


def load(root):
    root = path(root)
    meta = infer._json(root / "neural_metadata.json")
    if meta.get("architecture") != kind or not meta.get("complete"):
        raise ValueError("incomplete or unsupported mean-pooled cross-encoder")
    files = meta.get("files", {})
    if not {"model.safetensors", "head.safetensors", "config.json", "tokenizer_config.json"} <= set(files):
        raise ValueError("cross-encoder manifest omits required files")
    for name, digest in files.items():
        if path(name).name != name or infer._sha(root / name) != digest:
            raise ValueError("cross-encoder checkpoint changed")
    net = model(root)
    net.head.load_state_dict(load_file(root / "head.safetensors"))
    return net


def batches(lengths, batch, rank, world, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lengths))
    width = batch * world
    res = []
    for start in range(0, len(order), width * 50):
        chunk = order[start:start + width * 50]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        res.extend(chunk[i:i + width] for i in range(0, len(chunk) - width + 1, width))
    rng.shuffle(res)
    return [r[rank::world] for r in res]


def save(net, tokenizer, out, meta):
    out = path(out)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "neural_metadata.json"
    marker.unlink(missing_ok=True)
    weights = {k: v.detach().half().cpu().contiguous() if v.is_floating_point() else v.detach().cpu().contiguous()
               for k, v in net.encoder.state_dict().items()}
    net.encoder.save_pretrained(out, state_dict=weights, safe_serialization=True)
    tokenizer.save_pretrained(out)
    save_file({k: v.detach().half().cpu().contiguous() for k, v in net.head.state_dict().items()}, out / "head.safetensors")
    meta = {**meta, "files": {p.name: infer._sha(p) for p in out.iterdir() if p.is_file() and p.name != marker.name}}
    infer._write(marker, meta)


def worker(rank, world, args, pairs, texts_a, texts_b, pair_meta):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    tokenizer = autotokenizer.from_pretrained(args.base, local_files_only=True)
    net = model(args.base)
    net.encoder.get_input_embeddings().weight.requires_grad_(False)
    params = sum(p.numel() for p in net.parameters())
    if params > 8_000_000_000:
        raise ValueError("cross-encoder exceeds parameter limit")
    net.cuda(rank)
    wrapped = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    labels = pairs["y"].to_numpy().astype(np.float32)
    valid = pairs["valid"].to_numpy()
    train_ids, val_ids = np.flatnonzero(~valid), np.flatnonzero(valid)
    lengths = np.fromiter((len(texts_a[i]) + len(texts_b[i]) for i in train_ids), np.int32, len(train_ids))
    plan = batches(lengths, args.batch, rank, world, args.seed)
    if not plan:
        raise ValueError("cross-encoder fitting sample is smaller than one distributed batch")
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01, foreach=False)
    warmup = max(1, len(plan) // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s + 1) / warmup) *
                                                .5 * (1 + math.cos(math.pi * min(s, len(plan)) / len(plan))))
    scaler = torch.amp.GradScaler("cuda")
    started = time.monotonic()
    running = None
    wrapped.train()
    completed = 0
    src = infer._json(args.source)
    for step, sel in enumerate(plan):
        rows = train_ids[sel]
        batch = tokenizer([texts_a[i] for i in rows], [texts_b[i] for i in rows], padding=True,
                          truncation=True, max_length=args.length, return_tensors="pt").to(rank)
        target = torch.from_numpy(labels[rows]).cuda(rank)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = wrapped(**batch).logits[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite cross-encoder loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed = step + 1
        running = float(loss) if running is None else .98 * running + .02 * float(loss)
        elapsed = time.monotonic() - started
        if rank == 0 and completed % 100 == 0:
            print({"step": completed, "steps": len(plan), "loss": running,
                   "pairs_per_second": completed * args.batch * world / elapsed}, flush=True)
        if completed % 2000 == 0:
            dist.barrier()
            if rank == 0:
                save(net, tokenizer, args.out, {"version": 1, "architecture": kind, "complete": False,
                     "parameters": params, "problem_type": "multi_label_classification", "source": src,
                     "configuration": {"maxlen": args.length}, "training": {"steps": completed, "pair_source": pair_meta}})
            dist.barrier()
        stop = torch.tensor([elapsed >= args.max_seconds], dtype=torch.int32, device=rank)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            break
    dist.barrier()
    if rank == 0:
        net.eval()
        scores = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for start in range(0, len(val_ids), args.batch):
                rows = val_ids[start:start + args.batch]
                batch = tokenizer([texts_a[i] for i in rows], [texts_b[i] for i in rows], padding=True,
                                  truncation=True, max_length=args.length, return_tensors="pt").to(rank)
                scores.append(net(**batch).logits[:, 0].float().cpu().numpy())
        logits = np.concatenate(scores)
        y = labels[val_ids]
        loss = float(np.mean(np.logaddexp(0., logits) - y * logits))
        if str(src.get("license", "")).lower() not in ("mit", "apache-2.0"):
            raise ValueError("unsupported pretrained model license")
        metadata = {"version": 1, "architecture": kind, "complete": True, "parameters": params,
                    "problem_type": "multi_label_classification", "source": src, "text_format": neural.fmt,
                    "configuration": {"maxlen": args.length, "lr": args.lr, "seed": args.seed, "batch_per_gpu": args.batch,
                                      "world": world, "word_embeddings_frozen": True, "weight_decay": .01,
                                      "precision": "fp16", "max_seconds": args.max_seconds},
                    "training": {"pairs": int(completed * args.batch * world), "planned_pairs": int(len(train_ids)),
                                 "steps": completed, "complete_epoch": completed == len(plan),
                                 "seconds": time.monotonic() - started, "pair_source": pair_meta},
                    "validation": {"scope": "owner/reference disjoint fitting-fold entities; pair metric only",
                                   "pairs": len(val_ids), "logloss": loss, "accuracy": float(np.mean((logits >= 0) == (y == 1)))}}
        save(net, tokenizer, args.out, metadata)
        print(metadata["validation"], flush=True)
    dist.barrier()
    dist.destroy_process_group()


def train(args):
    if path(args.out).exists():
        raise ValueError("cross-encoder output must be new")
    if min(args.batch, args.length, args.threads) < 1 or not 0 < args.lr < 1 or not 0 < args.max_seconds < 86400:
        raise ValueError("invalid cross-encoder fitting limits")
    src = infer._json(args.source)
    if str(src.get("license", "")).lower() not in ("mit", "apache-2.0"):
        raise ValueError("unsupported pretrained model license")
    root = path(args.data)
    meta = infer._json(path(args.pairs).with_suffix(".json"))
    if meta.get("fit_folds") != [2] or infer._sha(args.pairs) != meta.get("sha256"):
        raise ValueError("invalid cross-encoder training-pair contract")
    texts = infer._json(root / "texts.json")
    if texts.get("source") != meta.get("source") or any(infer._sha(root / name) != digest for name, digest in texts.get("files", {}).items()):
        raise ValueError("cross-encoder text source changed")
    pairs = pl.read_parquet(args.pairs)
    if pairs["valid"].null_count() or set(pairs["valid"].unique()) != {True, False}:
        raise ValueError("missing grouped cross-encoder validation split")
    for valid in (True, False):
        if set(pairs.filter(pl.col("valid") == valid)["y"].unique()) != {0, 1}:
            raise ValueError("cross-encoder split requires both labels")
    refs = pl.read_parquet(root / "ref.parquet", columns=["text"])["text"]
    targets = pl.read_parquet(root / "target.parquet", columns=["text"])["text"]
    a, b = refs.gather(pairs["qid"]).to_list(), targets.gather(pairs["tid"]).to_list()
    world = torch.cuda.device_count()
    if world < 1:
        raise ValueError("cross-encoder fitting requires a gpu")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    del refs, targets
    gc.collect()
    mp.spawn(worker, args=(world, args, pairs, a, b, meta), nprocs=world, join=True)


def ensemble(roots, out):
    out = path(out)
    if out.exists() or not roots:
        raise ValueError("ensemble needs models and a new output")
    out.mkdir(parents=True)
    members, total, maxlen = [], 0, 0
    for i, root in enumerate(roots):
        root = path(root)
        meta = infer._json(root / "neural_metadata.json")
        if meta.get("problem_type") != "multi_label_classification" or meta.get("architecture") == "neural-ensemble-v1":
            raise ValueError("unsupported ensemble member")
        if meta.get("architecture") == kind and not meta.get("complete"):
            raise ValueError("ensemble member is incomplete")
        if str(meta.get("source", {}).get("license", "")).lower() not in ("mit", "apache-2.0"):
            raise ValueError("ensemble member license is unsupported")
        name = f"m{i}"
        dest = out / name
        dest.mkdir()
        for file in root.iterdir():
            if file.is_file() and (file.suffix in (".json", ".safetensors", ".model", ".txt") or file.name.startswith("pytorch_model")):
                sh.copyfile(file, dest / file.name)
        params = int(meta["parameters"])
        if params <= 0:
            raise ValueError("ensemble member has invalid parameter count")
        total += params
        maxlen = max(maxlen, meta["configuration"]["maxlen"])
        members.append({"name": name, "directory": name, "metadata_sha256": infer._sha(dest / "neural_metadata.json"),
                        "parameters": params, "upper_gate": .995 if params >= 400000000 else 1.})
    if total > 8_000_000_000:
        raise ValueError("ensemble exceeds parameter limit")
    meta = {"version": 1, "architecture": "neural-ensemble-v1", "problem_type": "multi_label_classification",
            "parameters": total, "source": {"model": "local calibrated ensemble", "license": "mit"},
            "configuration": {"maxlen": maxlen}, "members": members, "aggregation": "mean_logit", "text_format": neural.fmt}
    infer._write(out / "neural_metadata.json", meta)
    return meta


def check():
    import tempfile
    from transformers import BertConfig as config, BertModel as bert, BertTokenizerFast as tokenizer
    lengths = np.arange(23)
    a, b = batches(lengths, 2, 0, 2, 42), batches(lengths, 2, 1, 2, 42)
    assert len(a) == len(b) == 5
    combined = np.concatenate([np.r_[x, y] for x, y in zip(a, b)])
    assert len(np.unique(combined)) == 20 and set(np.concatenate(a)).isdisjoint(np.concatenate(b))
    assert all(np.array_equal(x, y) for x, y in zip(a, batches(lengths, 2, 0, 2, 42)))
    with tempfile.TemporaryDirectory() as tmp:
        root = path(tmp)
        base = root / "base"
        bert(config(vocab_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                    intermediate_size=32, max_position_embeddings=32)).save_pretrained(base)
        (base / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\nmain\n")
        tok = tokenizer(vocab_file=str(base / "vocab.txt"))
        net = model(base)
        net.encoder.get_input_embeddings().weight.requires_grad_(False)
        batch = tok(["alpha main", "beta"], ["alpha", "main"], padding=True, return_tensors="pt")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(**batch).logits[:, 0], torch.tensor([1., 0.]))
        loss.backward()
        assert net.head.weight.grad is not None and net.encoder.get_input_embeddings().weight.grad is None
        meta = {"architecture": kind, "complete": True, "parameters": sum(p.numel() for p in net.parameters()),
                "configuration": {"maxlen": 32}, "source": {"model": "check", "revision": "check", "license": "mit"},
                "problem_type": "multi_label_classification"}
        save(net, tok, root / "trained", meta)
        single = neural.load(root / "trained", "cpu")
        rows = pl.DataFrame({"text_a": ["alpha main", "beta"], "text_b": ["alpha", "main"]})
        one = neural.predict(single, rows)
        ensemble([root / "trained", root / "trained"], root / "ensemble")
        loaded = neural.load(root / "ensemble", "cpu")
        scores = neural.predict_members(loaded, rows)
        assert set(scores) == {"np_m0", "np_m1"} and np.allclose(neural.aggregate(scores), one)
        import match
        _, signature = match._neural(root / "ensemble")
        assert signature["parameters"] == 2 * meta["parameters"] and signature["score_columns"] == ["np_m0", "np_m1"]
        import package
        files = {}
        packaged = package.neural(root / "ensemble", files)
        assert len(packaged["sources"]) == 2 and any(p.endswith("m1/head.safetensors") for p in files)
    print("cross-encoder training, ensemble and packaging checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--data", type=path)
    p.add_argument("--pairs", type=path)
    p.add_argument("--base", type=path)
    p.add_argument("--source", type=path)
    p.add_argument("--out", type=path)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=7200)
    args = p.parse_args()
    if args.check:
        check()
    elif any(x is None for x in (args.data, args.pairs, args.base, args.source, args.out)):
        p.error("data, pairs, base, source and out are required")
    else:
        train(args)
