"""compact contrastive retrieval trained on fitting business identities only"""
import argparse as ap
import hashlib as hh
import json
import math
import time
from pathlib import Path as path

import numpy as np
import polars as pl

import infer


model_id = "intfloat/multilingual-e5-small"
revision = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
fmt = "query: {nm} | {ad}"


def training_pairs(data, n, seed):
    ref = pl.scan_parquet(data / "train/ref.parquet").filter(pl.col("fold") == 2)
    available = ref.select(pl.col("deg").sum()).collect().item()
    if n < 2 or not available:
        raise ValueError("contrastive fitting needs positive pairs")
    n = min(n, available)
    fraction = min(1., n / available * 1.1)
    refs = ref.select(pl.col("rid").cast(pl.Int32).alias("own"), "co", pl.col("nm").alias("rn"), pl.col("ad").alias("ra"))
    targets = pl.concat([pl.scan_parquet(data / "train" / f"s{i}.parquet").select("rid", "own", "nm", "ad") for i in (2, 3)])
    targets = targets.filter((pl.col("own") >= 0) & ((pl.col("rid").hash(seed) % 1048576) < math.ceil(fraction * 1048576)))
    pairs = targets.join(refs, on="own", how="inner").collect(engine="streaming")
    if len(pairs) < n:
        raise ValueError("deterministic oversample did not cover requested pair count")
    return pairs.sort("rid").sample(n=n, seed=seed, shuffle=True)


def batches(pairs, batch, seed):
    rng = np.random.default_rng(seed)
    p = pairs.with_row_index("i")
    name = pl.col("rn").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]", "").str.slice(0, 5)
    city = pl.col("ra").str.to_lowercase().str.split(",").list.slice(-2, 2).list.join(",").str.strip_chars()
    p = p.with_columns(pl.when(pl.col("i") % 2 == 0).then(pl.lit("n") + name).otherwise(pl.lit("a") + city).alias("key"),
                       pl.Series("noise", rng.random(len(p))))
    order = p.sort("co", "key", "noise")["i"].to_numpy()
    result = order[:len(order) // batch * batch].reshape(-1, batch).copy()
    rng.shuffle(result, axis=0)
    return result


def loss(a, b, owner, temperature=.05):
    import torch
    import torch.nn.functional as fn
    logits = a.float() @ b.float().T / temperature
    same = owner[:, None] == owner[None, :]
    eye = torch.eye(len(owner), device=a.device, dtype=torch.bool)
    logits = logits.masked_fill(same & ~eye, -1e4)
    labels = torch.arange(len(owner), device=a.device)
    return (fn.cross_entropy(logits, labels) + fn.cross_entropy(logits.T, labels)) / 2


def train(data, base, out, count=1_500_000, batch=64, length=96, lr=5e-5, seed=42, device="cuda", threads=4, resume=None):
    import torch
    import torch.nn.functional as fn
    from transformers import AutoModel as auto_model, AutoTokenizer as auto_tokenizer
    data, base, out = (path(x).resolve() for x in (data, base, out))
    if min(batch, length, threads) < 1 or lr <= 0 or (out.exists() and not resume):
        raise ValueError("invalid training arguments or existing output")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but unavailable")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    checkpointing = device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 16 * 1024 ** 3
    out.mkdir(parents=True, exist_ok=True)
    pair_file = out / "training_pairs.parquet"
    if resume:
        old = infer._json(path(resume) / "checkpoint.json")
        if infer._sha(pair_file) != old["signature"]["pairs_sha256"]:
            raise ValueError("resume training pair cache changed")
        pairs = pl.read_parquet(pair_file)
    else:
        pairs = training_pairs(data, count, seed)
        infer._pq(pairs, pair_file)
    groups = batches(pairs, batch, seed)
    if not len(groups):
        raise ValueError("not enough pairs for one contrastive batch")
    signature = {"data_meta_sha256": infer._sha(data / "meta.json"), "pairs": len(pairs), "requested_pairs": count, "seed": seed,
                 "batch": batch, "length": length, "lr": lr, "source_model": model_id, "source_revision": revision,
                 "gradient_checkpointing": checkpointing,
                 "base_files": {n: infer._sha(base / n) for n in ("config.json", "model.safetensors")},
                 "pairs_sha256": infer._sha(pair_file),
                 "groups_sha256": hh.sha256(groups.tobytes()).hexdigest(),
                 "pair_ids_sha256": hh.sha256(pairs.select("rid", "own").to_numpy().tobytes()).hexdigest()}
    if (base / "checkpoint.json").is_file():
        warmstart = infer._json(base / "checkpoint.json")
        previous = infer._json(base.parent / "training.json")
        if (previous.get("fit_folds") != [2] or previous.get("signature") != warmstart["signature"] or
                warmstart["signature"]["data_meta_sha256"] != signature["data_meta_sha256"]):
            raise ValueError("warmstart checkpoint uses different prepared data")
        signature["warmstart"] = {"checkpoint_sha256": infer._sha(base / "checkpoint.json"),
                                 "trained_pairs": warmstart["step"] * warmstart["signature"]["batch"], "fit_folds": [2]}
    left = [fmt.format(nm=n or "", ad=a or "") for n, a in pairs.select("rn", "ra").iter_rows()]
    right = [fmt.format(nm=n or "", ad=a or "") for n, a in pairs.select("nm", "ad").iter_rows()]
    owners = pairs["own"].to_numpy().copy()
    del pairs
    tokenizer = auto_tokenizer.from_pretrained(base, local_files_only=True)
    net = auto_model.from_pretrained(resume or base, local_files_only=True).to(device)
    if net.config.hidden_size != 384 or net.config.num_hidden_layers != 12:
        raise ValueError("base is not the expected compact encoder architecture")
    if checkpointing:
        net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.01)
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and not bf16)
    dtype = torch.bfloat16 if bf16 else torch.float16
    start = 0
    if resume:
        old = infer._json(path(resume) / "checkpoint.json")
        if old["signature"] != signature:
            raise ValueError("resume training population or configuration changed")
        state = torch.load(path(resume) / "state.pt", weights_only=True, map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        start = old["step"]
    infer._write(out / "training.json", {"signature": signature, "fit_folds": [2], "complete": False})
    net.train()
    begun = time.monotonic()
    warmup = max(1, len(groups) // 20)

    def encode(text):
        batch_inputs = tokenizer(text, padding=True, truncation=True, max_length=length, return_tensors="pt").to(device)
        output = net(**batch_inputs).last_hidden_state.float()
        mask = batch_inputs["attention_mask"].unsqueeze(-1)
        return fn.normalize((output * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)

    for step in range(start, len(groups)):
        ids = groups[step]
        rate = lr * min(1., (step + 1) / warmup) * .5 * (1 + math.cos(math.pi * step / len(groups)))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=dtype, enabled=device == "cuda"):
            a = encode([left[i] for i in ids])
            b = encode([right[i] for i in ids])
            value = loss(a, b, torch.as_tensor(owners[ids], device=device))
        if not torch.isfinite(value):
            raise ValueError("nonfinite contrastive loss; last complete checkpoint retained")
        scaler.scale(value).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        if step % 100 == 0:
            print(json.dumps({"step": step, "steps": len(groups), "loss": value.item(),
                              "pairs_per_second": (step + 1 - start) * batch / (time.monotonic() - begun)}), flush=True)
        if (step + 1) % 1000 == 0:
            checkpoint = out / f"checkpoint-{step + 1}"
            checkpoint.mkdir()
            net.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []},
                       checkpoint / "state.pt")
            infer._write(checkpoint / "checkpoint.json", {"signature": signature, "step": step + 1})
    net.save_pretrained(out)
    tokenizer.save_pretrained(out)
    files = {p.name: infer._sha(p) for p in out.iterdir() if p.is_file() and
             (p.suffix == ".safetensors" or p.name in {"config.json", "tokenizer.json", "tokenizer_config.json",
                                                      "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.txt"})}
    metadata = {"version": 1, "kind": "contrastive-business-retriever", "complete": True, "model": model_id,
                "base_revision": revision, "fit_folds": [2], "format": fmt, "length": length,
                "signature": signature, "parameters": sum(p.numel() for p in net.parameters()), "files": files,
                "trained_pairs": len(groups) * batch, "seconds": time.monotonic() - begun,
                "precision": "bf16" if bf16 else "fp16" if device == "cuda" else "fp32"}
    infer._write(out / "retriever.json", metadata)
    infer._write(out / "training.json", {"signature": signature, "fit_folds": [2], "complete": True})
    return metadata


def bundle(root):
    root = path(root).resolve()
    meta = infer._json(root / "retriever.json")
    if (meta.get("version") != 1 or meta.get("kind") != "contrastive-business-retriever" or not meta.get("complete") or
            meta.get("fit_folds") != [2] or meta.get("model") != model_id or meta.get("format") != fmt or
            not {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(meta.get("files", {}))):
        raise ValueError("invalid trained retriever metadata")
    for name, sha in meta["files"].items():
        if path(name).name != name or infer._sha(root / name) != sha:
            raise ValueError("trained retriever files changed")
    return meta, infer._sha(root / "retriever.json")


def check():
    import tempfile as tf
    import torch
    import hybrid
    a = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
    value = loss(a, a, torch.tensor([1, 1, 2]))
    assert value < 1e-5
    value.backward()
    assert torch.isfinite(a.grad).all()
    with tf.TemporaryDirectory() as tmp:
        data = infer._check_data(path(tmp))
        ref = pl.read_parquet(data / "train/ref.parquet").with_columns(
            pl.when(pl.col("rid") == 0).then(2).otherwise(0).cast(pl.UInt8).alias("fold"))
        ref.write_parquet(data / "train/ref.parquet")
        pairs = training_pairs(data, 2, 42)
        assert pairs["own"].to_list() == [0, 0]
        for sr in (2, 3):
            p = data / "train" / f"s{sr}.parquet"
            pl.read_parquet(p).reverse().write_parquet(p)
        assert training_pairs(data, 2, 42).equals(pairs)
        assert batches(pairs, 2, 42).shape == (1, 2)
        checkpoint = path(tmp) / "trained"
        checkpoint.mkdir()
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
            (checkpoint / name).write_bytes(b"check")
        metadata = {"version": 1, "kind": "contrastive-business-retriever", "complete": True, "fit_folds": [2],
                    "model": model_id, "base_revision": revision, "format": fmt, "length": 96,
                    "files": {p.name: infer._sha(p) for p in checkpoint.iterdir()}}
        infer._write(checkpoint / "retriever.json", metadata)
        old = hybrid.specs([{"model": model_id, "revision": revision, "checkpoint": str(checkpoint)}])[0]
        (checkpoint / "model.safetensors").write_bytes(b"changed")
        try:
            bundle(checkpoint)
        except ValueError:
            pass
        else:
            raise AssertionError("changed checkpoint accepted")
        metadata["files"]["model.safetensors"] = infer._sha(checkpoint / "model.safetensors")
        infer._write(checkpoint / "retriever.json", metadata)
        new = hybrid.specs([{"model": model_id, "revision": revision, "checkpoint": str(checkpoint)}])[0]
        assert hybrid.dirs(path(tmp), old, "us", "train")[0] != hybrid.dirs(path(tmp), new, "us", "train")[0]
    print("retriever checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=path, default=path("cache/data"))
    p.add_argument("--base", type=path, default=path("cache/models/e5-small-source"))
    p.add_argument("--out", type=path, default=path("artifacts/retr-small"))
    p.add_argument("--pairs", type=int, default=1_500_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--length", type=int, default=96)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", type=path)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        check()
    else:
        print(json.dumps(train(a.data, a.base, a.out, a.pairs, a.batch, a.length, device=a.device, threads=a.threads, resume=a.resume), indent=2))
