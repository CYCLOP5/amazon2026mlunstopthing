'compare compact retrieval checkpoints on complete reference pools'

import argparse as ap
import gc
import time
from pathlib import Path as path

import polars as pl

import embed
import hybrid
import infer
import retr


def summary(data, source, result, folder):
    anchors, queries, _, country, fold = embed.load_run(data, source)
    if fold != 0:
        raise ValueError("retriever comparison is restricted to development fold0")
    pairs = pl.read_parquet([folder / p for p in result["parts"]])
    ranked = pairs.sort("tid", "ds_e5_small", "qid", descending=[False, True, False])
    rows = []
    for k in (1, 3, 10, 50):
        top = ranked.group_by("tid", maintain_order=True).head(k)
        metrics = hybrid.score(anchors, top)
        for missing in (False, True):
            truth = queries.filter((pl.col("own") >= 0) &
                                   ((pl.col("ad").fill_null("").str.strip_chars() == "") == missing))
            hits = top.filter(pl.col("y") == 1).join(truth.select(pl.col("rid").alias("tid")), on="tid", how="semi")
            metrics[f"recall_{'blank' if missing else 'address'}"] = len(hits) / len(truth) if len(truth) else 1.
        rows.append({"k": k, **metrics})
    return {"country": country, "anchors": len(anchors), "queries": len(queries),
            "reference_pool": result["reference_pool"], "union_recall": result["link_recall"],
            "cosine_ranking": rows}


def run(assets, out, device="cuda", batch=512, threads=20, cache=None):
    assets, out = path(assets).resolve(), path(out).resolve()
    cache = path(cache or "cache/retriever-validation").resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("retriever validation output must be new")
    out.mkdir(parents=True, exist_ok=True)
    meta = infer._json(assets / "assets.json")
    for name, digest in meta["files"].items():
        if infer._sha(assets / name) != digest:
            raise ValueError("retriever validation asset changed")
    learned, digest = retr.bundle(assets / "learned")
    if infer._sha(assets / "data/meta.json") != learned["signature"]["data_meta_sha256"]:
        raise ValueError("retriever fitting data differs from validation data")
    report = {"scope": "fold0 queries, complete country reference pools; not a matching score",
              "format": retr.fmt, "reference_format": "query", "length": learned["length"],
              "checkpoint_sha256": digest, "variants": {}}
    started = time.monotonic()
    for name in ("frozen", "learned"):
        model_path = assets / ("base" if name == "frozen" else "learned")
        model, actual_device, params = embed.model_load(retr.model_id, retr.revision, device, learned["length"], model_path)
        spec = {"model": retr.model_id, "revision": retr.revision, "length": learned["length"],
                "encoder": model, "params": params, "device": actual_device}
        if name == "learned":
            spec["checkpoint"] = str(model_path)
        report["variants"][name] = []
        for country in meta["countries"]:
            src = assets / "queries" / country
            folder = out / name / country
            res = hybrid.probe(assets / "data", cache, src, folder, models=[spec],
                                  device=device, batch=batch, k_lex=10, k_dense=50, threads=threads)
            row = summary(assets / "data", src, res, folder)
            report["variants"][name].append(row)
            print(name, row, flush=True)
            infer._write(out / "report.json", report)
        del model, spec
        gc.collect()
        if actual_device == "cuda":
            import torch
            torch.cuda.empty_cache()
    report["seconds"] = time.monotonic() - started
    infer._write(out / "report.json", report)
    return report


def check():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = path(tmp)
        data = infer._check_data(root)
        src, folder = root / "queries", root / "result"
        src.mkdir(); folder.mkdir()
        refs = pl.read_parquet(data / "train/ref.parquet")
        infer._pq(refs.filter(pl.col("rid") == 0), src / "anchors.parquet")
        queries = pl.concat([pl.read_parquet(data / "train" / f"s{i}.parquet") for i in (2, 3)]).filter(pl.col("co") == "us")
        infer._pq(queries, src / "queries.parquet")
        pairs = pl.DataFrame({"qid": [0, 1, 0], "tid": [10, 10, 20], "own": [0, 0, 0],
                              "y": [1, 0, 1], "ds_e5_small": [.8, .9, .9]})
        infer._pq(pairs, folder / "pairs.parquet")
        res = summary(data, src, {"parts": ["pairs.parquet"], "reference_pool": 3, "link_recall": 1.}, folder)
        assert res["cosine_ranking"][0]["link_recall"] == .5
        assert res["cosine_ranking"][1]["link_recall"] == 1.
    print("retriever evaluation checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--assets", type=path)
    p.add_argument("--out", type=path)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--threads", type=int, default=20)
    p.add_argument("--cache", type=path)
    a = p.parse_args()
    if a.check:
        check()
    elif a.assets is None or a.out is None:
        p.error("assets and out are required")
    else:
        print(run(a.assets, a.out, a.device, a.batch, a.threads, a.cache))
