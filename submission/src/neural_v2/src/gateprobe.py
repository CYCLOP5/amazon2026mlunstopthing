'audit candidate survival and adaptive gate budgets on held-out owners'

import argparse as ap
import concurrent.futures as cf
from pathlib import Path as path
import shutil as sh

import lightgbm as lgb
import numpy as np
import polars as pl

import block
import embed
import full
import infer
import match
import train


def keep(rank, score, maximum, floor=None):
    return (rank < maximum) & ((rank == 0) | (score >= floor)) if floor is not None else rank < maximum


def audit(assets, gate, out, work, threads=80):
    assets, gate, out, work = map(path, (assets, gate, out, work))
    source = infer._json(assets / "assets.json")
    queries = [x for x in source["queries"] if x["fold"] == 0]
    threads = full.limits(len(queries), threads)
    if infer._sha(assets / "data/meta.json") != source["data_meta_sha256"]:
        raise ValueError("gate probe data changed")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True)
    data = assets / "data"
    rs = full.retrievers(assets, gate / "reverse", work / "retrievers.json")
    meta, names, _, signature = match._gate(gate / "gate")
    def generate(item, gpu):
        label = item["directory"]
        src = work / "queries" / label
        sh.copytree(assets / "queries" / label, src)
        _, _, _, country, fold = embed.load_run(data, src)
        infer._write(src / "metrics.json", {"country": country, "fold": fold, "score_version": block.sv})
        full.command("hybrid.py", ["--data", data, "--cache", work / "cache", "--run", src,
                                   "--out", work / "runs" / label, "--retrievers-file", rs,
                                   "--k-lex", 10, "--k-dense", 50, "--batch", 512, "--threads", threads // len(queries),
                                   "--device", "cuda"], out / (label + ".log"), gpu, threads // len(queries))
    with cf.ThreadPoolExecutor(max_workers=len(queries)) as pool:
        jobs = [pool.submit(generate, item, i) for i, item in enumerate(queries)]
        for job in jobs:
            job.result()
    model = lgb.Booster(model_file=str(gate / "gate/lgb.txt"))
    rows = []
    for item in queries:
        run = train._run(data, work / "runs" / item["directory"], 0)
        x, y, tid, qid = train._features(data, work, run, threads, names, meta["feature_backend"],
                                        gate / "gate/normalizer.json", work / "cache")
        d = pl.DataFrame({"qid": qid, "tid": tid, "y": y, "gate": train.predict(model, x, threads)})
        d = d.sort(["tid", "gate", "qid"], descending=[False, True, False]).with_columns(
            pl.int_range(pl.len(), dtype=pl.UInt32).over("tid").alias("rank"))
        d.write_parquet(out / (item["country"] + "-scores.parquet"), compression="zstd")
        run["anchors"].write_parquet(out / (item["country"] + "-anchors.parquet"), compression="zstd")
        run["queries"].write_parquet(out / (item["country"] + "-queries.parquet"), compression="zstd")
        y, qid = d["y"].to_numpy().astype(bool), d["qid"].to_numpy()
        rank, score = d["rank"].to_numpy(), d["gate"].to_numpy()
        total = int(run["anchors"]["deg"].sum())
        policies = [(k, None) for k in (1, 3, 5, 10, 20, 50)]
        policies += [(k, f) for k in (20, 50) for f in (.0001, .001, .005, .01, .02)]
        for maximum, floor in policies:
            sel = keep(rank, score, maximum, floor)
            counts = d.filter(pl.Series(sel)).group_by("tid").len()["len"].to_numpy()
            metric = train.anchor_f05(run["anchors"], qid, y, sel & y)
            row = {"country": item["country"], "maximum": maximum, "gate_floor": floor,
                   "true_link_recall": float((sel & y).sum() / total), "oracle_macro_f05": float(metric["f05"].mean()),
                   "mean_per_query": float(sel.sum() / len(run["queries"])),
                   "p95_per_query": float(np.quantile(counts, .95)), "max_per_query": int(counts.max()),
                   "pairs": int(sel.sum()), "true_links": total, "queries": len(run["queries"])}
            rows.append(row)
            print(row, flush=True)
    res = {"scope": "complete aliases of fixed held-out owners against full reference pools; candidate oracle only",
              "gate_sha256": signature["sha256"], "data_meta_sha256": infer._sha(data / "meta.json"), "policies": rows}
    infer._write(out / "result.json", res)
    return res


def check():
    rank = np.array([0, 1, 2, 0, 1])
    score = np.array([.9, .04, .0001, .0009, .0008])
    assert keep(rank, score, 50, .01).tolist() == [True, True, False, True, False]
    assert keep(rank, score, 1, .01).tolist() == [True, False, False, True, False]
    assert keep(rank, score, 3).all()
    frame = pl.DataFrame({"tid": [0, 0, 0, 1, 1], "qid": [2, 1, 0, 7, 9], "gate": score})
    assert match._top(frame, 50, .01).select("tid", "qid").rows() == [(0, 1), (0, 2), (1, 7)]
    assert match._top(frame, 1, .01).select("tid", "qid").rows() == [(0, 2), (1, 7)]
    tied = pl.DataFrame({"tid": [0, 0], "qid": [9, 3], "gate": [.001, .001]})
    assert match._top(tied, 50, .01)["qid"].to_list() == [3]
    print("adaptive gate budget checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=path)
    p.add_argument("--gate", type=path)
    p.add_argument("--out", type=path)
    p.add_argument("--work", type=path, default=path("work/gateprobe"))
    p.add_argument("--threads", type=int, default=80)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        check()
    elif a.assets and a.gate and a.out:
        audit(a.assets, a.gate, a.out, a.work, a.threads)
    else:
        p.error("--assets --gate --out required")
