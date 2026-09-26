"""full-population source1-to-target retrieval and reciprocal-rank features"""
import argparse as ap
import math
from pathlib import Path as path

import faiss
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import hybrid
import infer
import retr


field = "rr_e5_small"


def config(root):
    root = path(root)
    meta = infer._json(root / "config.json")
    if (meta.get("version") != 1 or meta.get("kind") != "source1-target-retrieval" or
            not isinstance(meta.get("k"), int) or not 1 <= meta["k"] <= 1000 or
            meta.get("code_sha256") != infer._sha(path(__file__))):
        raise ValueError("invalid or stale reverse retrieval configuration")
    if (set(meta.get("countries", {})) != {"train", "test"} or
            any(not isinstance(xs, list) or any(not isinstance(c, str) for c in xs) for xs in meta["countries"].values())):
        raise ValueError("invalid reverse country coverage")
    return meta, infer._sha(root / "config.json")


def build(data, cache, checkpoint, root, split, country, k=40, probes=64, batch=128, threads=8, device="cuda", blank_only=False):
    data, cache, checkpoint, root = (path(p).resolve() for p in (data, cache, checkpoint, root))
    model_meta, digest = retr.bundle(checkpoint)
    if not 1 <= k <= 1000 or min(probes, batch, threads) < 1:
        raise ValueError("invalid reverse retrieval settings")
    contract = {"version": 1, "kind": "source1-target-retrieval", "model_sha256": digest, "k": k,
                "probes": probes, "blank_only": blank_only, "algorithm": "faiss-ivf-flat-inner-product",
                "data_meta_sha256": infer._sha(data / "meta.json"), "code_sha256": infer._sha(path(__file__)),
                "countries": {s: sorted(pl.read_parquet(data / s / "ref.parquet", columns=["co"])["co"].unique().to_list())
                              for s in ("train", "test")}}
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists() and infer._json(root / "config.json") != contract:
        raise ValueError("reverse root uses different settings")
    infer._write(root / "config.json", contract)
    out = root / f"{split}_{hybrid.tag(country)}"
    if out.exists():
        raise ValueError("reverse country output already exists")
    refs = pl.read_parquet(data / split / "ref.parquet").filter(pl.col("co") == country).sort("rid")
    all_targets = pl.concat([pl.scan_parquet(data / split / f"s{i}.parquet") for i in (2, 3)])
    total = all_targets.select(pl.len()).collect().item()
    targets = all_targets.filter(pl.col("co") == country)
    if blank_only:
        targets = targets.filter(pl.col("ad").fill_null("").str.strip_chars() == "")
    targets = targets.sort("rid").collect(engine="streaming")
    if not len(refs) or not len(targets):
        raise ValueError("reverse retrieval needs reference and target records")
    spec = hybrid.specs([{"model": retr.model_id, "revision": retr.revision, "checkpoint": str(checkpoint)}])[0]
    model, _, parameters, _ = hybrid.load(spec, device)
    rv = hybrid.refs(data, cache, spec, model, batch, refs, country, split, parameters)[0]
    tv = hybrid.refs(data, cache, spec, model, batch, targets, country, split + "_targets", parameters)[0]
    faiss.omp_set_num_threads(threads)
    dimension = tv.shape[1]
    clusters = min(4096, max(1, int(math.sqrt(len(tv)))), max(1, len(tv) // 100))
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, clusters, faiss.METRIC_INNER_PRODUCT)
    sample = np.random.default_rng(42).choice(len(tv), min(len(tv), 262144), replace=False)
    index.train(np.asarray(tv[sample], np.float32))
    for lo in range(0, len(tv), 100000):
        index.add(np.asarray(tv[lo:lo + 100000], np.float32))
    index.nprobe = min(probes, clusters)
    out.mkdir()
    ids = targets["rid"].to_numpy()
    parts = []
    for lo in range(0, len(rv), 1024):
        q = np.asarray(rv[lo:lo + 1024], np.float32)
        _, near = index.search(q, min(k, len(tv)))
        if (near < 0).any():
            raise ValueError("reverse search returned incomplete neighbours")
        part = out / f"part-{lo:09d}.parquet"
        frame = pl.DataFrame({"tid": ids[near].reshape(-1),
                              "qid": np.repeat(refs["rid"].to_numpy()[lo:lo + len(q)], near.shape[1]),
                              "rank": np.tile(np.arange(1, near.shape[1] + 1, dtype=np.uint16), len(q))})
        infer._pq(frame, part)
        parts.append(part)
    sorted_pairs = out / "pairs.parquet"
    infer._sink(pl.scan_parquet(parts).sort("tid", "qid"), sorted_pairs)
    size = pq.ParquetFile(sorted_pairs).metadata.num_rows
    qids = np.lib.format.open_memmap(out / "qids.npy", mode="w+", dtype=np.uint32, shape=(size,))
    ranks = np.lib.format.open_memmap(out / "ranks.npy", mode="w+", dtype=np.uint16, shape=(size,))
    counts = np.zeros(total, np.uint64)
    offset = 0
    for values in pq.ParquetFile(sorted_pairs).iter_batches():
        frame = pl.from_arrow(values)
        qids[offset:offset + len(frame)] = frame["qid"].to_numpy()
        ranks[offset:offset + len(frame)] = frame["rank"].to_numpy()
        unique, count = np.unique(frame["tid"].to_numpy(), return_counts=True)
        counts[unique] += count.astype(np.uint64)
        offset += len(frame)
    qids.flush()
    ranks.flush()
    np.save(out / "offsets.npy", np.r_[np.uint64(0), counts.cumsum()], allow_pickle=False)
    meta = {"contract_sha256": infer._sha(root / "config.json"), "split": split, "country": country,
            "references": len(refs), "targets": len(targets), "total_targets": total, "pairs": size,
            "clusters": clusters, "effective_probes": index.nprobe, "source_model": model_meta["model"],
            "files": {n: infer._sha(out / n) for n in ("qids.npy", "ranks.npy", "offsets.npy")}}
    infer._write(out / "metadata.json", meta)
    for p in parts:
        p.unlink()
    return meta


def load(root, data, split, country, spec):
    root = path(root)
    contract, digest = config(root)
    if contract["model_sha256"] != spec.get("checkpoint_sha256") or contract["data_meta_sha256"] != infer._sha(path(data) / "meta.json"):
        raise ValueError("reverse retrieval model or dataset differs")
    directory = root / f"{split}_{hybrid.tag(country)}"
    meta = infer._json(directory / "metadata.json")
    if meta["contract_sha256"] != digest or meta["split"] != split or meta["country"] != country:
        raise ValueError("reverse retrieval scope differs")
    if set(meta["files"]) != {"qids.npy", "ranks.npy", "offsets.npy"}:
        raise ValueError("reverse retrieval files are incomplete")
    if any(infer._sha(directory / n) != sha for n, sha in meta["files"].items()):
        raise ValueError("reverse retrieval cache changed")
    state = {"metadata": meta, **{n: np.load(directory / f"{n}.npy", mmap_mode="r") for n in ("qids", "ranks", "offsets")}}
    if (state["qids"].dtype != np.uint32 or state["ranks"].dtype != np.uint16 or state["offsets"].dtype != np.uint64 or
            len(state["qids"]) != meta["pairs"] or len(state["ranks"]) != meta["pairs"] or
            len(state["offsets"]) != meta["total_targets"] + 1 or state["offsets"][0] != 0 or
            state["offsets"][-1] != meta["pairs"] or (state["offsets"][1:] < state["offsets"][:-1]).any() or
            (state["ranks"] < 1).any() or (state["ranks"] > contract["k"]).any()):
        raise ValueError("invalid reverse retrieval arrays")
    return state


def pairs(state, tids, references):
    tids = np.asarray(tids, dtype=np.uint32)
    if len(tids) and tids.max() + 1 >= len(state["offsets"]):
        raise ValueError("reverse query outside indexed target pool")
    sizes = state["offsets"][tids + 1] - state["offsets"][tids]
    positions = [np.arange(state["offsets"][t], state["offsets"][t + 1], dtype=np.int64) for t in tids]
    positions = np.concatenate(positions) if positions else np.empty(0, np.int64)
    frame = pl.DataFrame({"tid": np.repeat(tids, sizes.astype(np.int64)), "qid": state["qids"][positions],
                          field: (1 / state["ranks"][positions]).astype(np.float32)})
    return frame.filter(pl.col("qid").is_in(references["rid"].implode()))


def check():
    state = {"offsets": np.array([0, 1, 3, 3], np.uint64), "qids": np.array([4, 4, 7], np.uint32),
             "ranks": np.array([2, 1, 3], np.uint16)}
    found = pairs(state, np.array([2, 1], np.uint32), pl.DataFrame({"rid": [7]}))
    assert found.select("tid", "qid").rows() == [(1, 7)] and abs(found[field][0] - 1 / 3) < 1e-6
    from unittest.mock import patch
    import block

    class model:
        def encode(self, txt, **kwargs):
            return np.tile(np.array([[1., 0.]], np.float32), (len(txt), 1))

    rev = {"offsets": np.array([0, 1], np.uint64), "qids": np.array([1], np.uint32), "ranks": np.array([1], np.uint16)}
    references = pl.DataFrame({"rid": [0, 1]}, schema_overrides={"rid": pl.UInt32})
    queries = pl.DataFrame({"rid": [0], "nm": ["name"], "ad": [""], "nn": ["name"], "an": [""], "co": ["us"], "own": [1], "sr": [2]},
                           schema_overrides={"rid": pl.UInt32, "sr": pl.UInt8})
    encoder = {"spec": {"model": retr.model_id}, "model": model(), "feature": "ds_e5_small", "device": "cpu",
               "refs": np.array([[1., 0.], [0., 1.]], np.float32), "rows": np.array([0, 1]), "index": None, "reverse": rev}
    runtime = {"country": "us", "idx": None, "eq": None, "refs": references, "batch": 1, "encoders": [encoder],
               "dense_features": ["ds_e5_small", field]}
    with patch.object(block, "search", return_value=pl.DataFrame(schema=block.schema)), patch.object(block, "rescore", side_effect=lambda q, p, i: p):
        recovered = hybrid.search(runtime, queries, k_lex=1, k_dense=1, threads=1)
    assert recovered.filter(pl.col("y") == 1)["qid"].to_list() == [1]
    assert recovered.filter(pl.col("qid") == 1)[field][0] == 1.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = path(tmp)
        data = infer._check_data(root)
        for split in ("train", "test"):
            for source, offset in ((2, 0), (3, 2)):
                filename = data / split / f"s{source}.parquet"
                frame = pl.read_parquet(filename)
                infer._pq(frame.with_columns(pl.Series("rid", np.arange(offset, offset + len(frame), dtype=np.uint32))), filename)

        def vectors(data, cache, spec, encoder, batch, frame, country, split, parameters):
            return np.tile(np.array([[1., 0.]], np.float32), (len(frame), 1)), None, None, None

        with patch.object(retr, "bundle", return_value=({"model": retr.model_id}, "check")), \
             patch.object(hybrid, "specs", return_value=[{"model": retr.model_id, "checkpoint_sha256": "check"}]), \
             patch.object(hybrid, "load", return_value=(model(), "cpu", 2, None)), patch.object(hybrid, "refs", side_effect=vectors):
            result = build(data, root / "cache", root / "model", root / "reverse", "train", "us", k=1, threads=1, device="cpu")
            for split, country in (("train", "france"), ("test", "us"), ("test", "france")):
                build(data, root / "cache", root / "model", root / "reverse", split, country, k=1, threads=1, device="cpu")
        loaded = load(root / "reverse", data, "train", "us", {"checkpoint_sha256": "check"})
        assert result["pairs"] == result["references"] and loaded["offsets"][-1] == result["pairs"]
        import package
        _, digest = config(root / "reverse")
        specs = [{"checkpoint_sha256": "check"}]
        files = {"code/business_entity_resolution/src/reverse.py": path(__file__)}
        package.reverse_assets([{"reverse_contract_sha256": digest}], specs, [root / "reverse"], files)
        assert specs[0]["reverse_contract_sha256"] == digest and any(n.endswith("test_france/ranks.npy") for n in files)
    print("reverse retrieval checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=path, default=path("cache/data"))
    p.add_argument("--cache", type=path, default=path("cache"))
    p.add_argument("--checkpoint", type=path)
    p.add_argument("--out", type=path)
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--country")
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--probes", type=int, default=64)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--blank-only", action="store_true")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        check()
    elif a.checkpoint and a.out and a.country:
        print(build(a.data, a.cache, a.checkpoint, a.out, a.split, a.country, a.k, a.probes, a.batch, a.threads, blank_only=a.blank_only))
    else:
        p.error("checkpoint, out and country are required")
