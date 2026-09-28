"""reuse full scored candidates to build owner-isolated hard-pair datasets"""

import argparse as ap
from pathlib import Path as path

import numpy as np
import polars as pl

import infer
import neural
import post


def group(ids):
    return post.partition(np.asarray(ids, np.uint64), 20) == 0


def split_rows(frame, folds, nref):
    qid, tid, own = (frame[c].to_numpy() for c in ("qid", "tid", "own"))
    owner = np.maximum(own, 0)
    eligible = (folds[qid] == 2) & ((own < 0) | (folds[owner] == 2))
    valid = group(qid)
    other = group(np.where(own >= 0, own, tid.astype(np.int64) + nref))
    return frame.with_columns(pl.Series("valid", valid)).filter(pl.Series(eligible & (valid == other)))


def text_table(frame, out):
    expression = pl.concat_str(pl.lit("name: "), pl.col("nm").fill_null(""), pl.lit("\naddress: "),
                               pl.col("ad").fill_null(""), pl.lit("\ncountry: "), pl.col("co").fill_null(""))
    frame.select("rid", expression.alias("text")).sort("rid").sink_parquet(out, compression="zstd", row_group_size=262144)


def build(scores, out, seeds=(21, 41, 51, 71, 131), entities=600000, max_pairs=3000000):
    scores, out = path(scores), path(out)
    source = post.verified(scores, "train")
    data = path(source["data"])
    out.mkdir(parents=True, exist_ok=True)
    refs = pl.read_parquet(data / "train/ref.parquet")
    folds = refs["fold"].to_numpy()
    targets = pl.concat([pl.scan_parquet(data / "train" / f"s{i}.parquet") for i in (2, 3)])
    common = {"data_meta_sha256": source["data_meta_sha256"], "score_sha256": source["score_sha256"],
              "source_config_sha256": source["config_sha256"], "text_format": neural.fmt}
    if (out / "texts.json").exists():
        old = infer._json(out / "texts.json")
        if old.get("source") != common or any(infer._sha(out / name) != digest for name, digest in old["files"].items()):
            raise ValueError("cross-encoder text cache changed")
    else:
        text_table(refs.lazy(), out / "ref.parquet")
        text_table(targets, out / "target.parquet")
        infer._write(out / "texts.json", {"source": common, "files": {n: infer._sha(out / n) for n in ("ref.parquet", "target.parquet")}})
    pool = refs.filter(pl.col("fold") == 2).sort("rid")
    raw = pl.scan_parquet(scores).select("qid", "tid", "own", "y", "gate_prob", "neural_prob")
    for seed in seeds:
        file = out / f"pairs-{seed}.parquet"
        if file.exists():
            raise ValueError("cross-encoder pair output already exists")
        selected = pool.sample(n=min(entities, len(pool)), seed=seed).sort("rid")
        ids = selected["rid"]
        negatives = (raw.filter(pl.col("qid").is_in(ids.implode()) & (pl.col("y") == 0))
                     .with_columns(pl.max_horizontal("gate_prob", "neural_prob").alias("hardness"))
                     .select("qid", "tid", "own", "y", "hardness").collect(engine="streaming"))
        positives = (targets.filter(pl.col("own").is_in(ids.cast(pl.Int32).implode()))
                     .select(pl.col("own").cast(pl.UInt32).alias("qid"), pl.col("rid").alias("tid"), "own",
                             pl.lit(1, pl.UInt8).alias("y"), pl.lit(1., pl.Float32).alias("hardness"))
                     .collect(engine="streaming"))
        frame = pl.concat([positives, negatives], how="vertical_relaxed")
        frame = split_rows(frame, folds, len(refs))
        positive = frame.filter(pl.col("y") == 1)
        negative = frame.filter(pl.col("y") == 0).sort("hardness", "qid", "tid", descending=[True, False, False])
        negative = negative.head(max(0, max_pairs - len(positive)))
        frame = pl.concat([positive, negative]).select("qid", "tid", "own", "y", "valid")
        frame = frame.sample(fraction=1., shuffle=True, seed=seed)
        if len(positive) > max_pairs or frame["valid"].n_unique() != 2:
            raise ValueError("invalid hard-pair sample or validation split")
        for valid in (True, False):
            if set(frame.filter(pl.col("valid") == valid)["y"].unique()) != {0, 1}:
                raise ValueError("hard-pair split requires both labels")
        infer._pq(frame, file)
        metadata = {"version": 1, "kind": "owner-isolated-hard-cross-encoder-pairs", "fit_folds": [2], "seed": seed,
                    "reference_entities": len(selected), "pairs": len(frame), "positive": int(frame["y"].sum()),
                    "validation_pairs": frame.filter(pl.col("valid")).height,
                    "selection": "all selected-entity positives plus highest-scored cached negatives; both endpoints in fitting fold",
                    "validation": "same owner/reference holdout group; orphan groups keyed by target id",
                    "source": common, "sha256": infer._sha(file)}
        infer._write(file.with_suffix(".json"), metadata)
        print(metadata, flush=True)
    return {"texts": str(out / "texts.json"), "seeds": list(seeds)}


def check():
    refs = np.full(40, 2, np.uint8)
    held = int(np.flatnonzero(group(np.arange(40)))[0])
    fit = int(np.flatnonzero(~group(np.arange(40)))[0])
    frame = pl.DataFrame({"qid": [held, fit, fit], "tid": [0, 1, 2], "own": [held, held, fit], "y": [1, 0, 1]})
    result = split_rows(frame, refs, len(refs))
    assert result["tid"].to_list() == [0, 2] and result["valid"].to_list() == [True, False]
    print("hard-pair owner-split checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--scores", type=path)
    p.add_argument("--out", type=path)
    p.add_argument("--seeds", nargs="+", type=int, default=[21, 41, 51, 71, 131])
    p.add_argument("--entities", type=int, default=600000)
    p.add_argument("--max-pairs", type=int, default=3000000)
    a = p.parse_args()
    if a.check:
        check()
    elif a.scores is None or a.out is None:
        p.error("scores and out are required")
    else:
        print(build(a.scores, a.out, a.seeds, a.entities, a.max_pairs))
