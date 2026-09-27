'france variants from scores that were never used for france: varun v2\'s stack probability ("newest")'
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from er.stack import decode  # noqa: E402

VARIANTS = {"run6": ("friend",), "v2": ("newest",), "mix": ("newest", "friend"), "mix3": ("newest", "friend", "graph")}


def mean_available(cols):
    vals = [pl.col(c) for c in cols]
    n = pl.sum_horizontal([v.is_not_null().cast(pl.Float32) for v in vals])
    s = pl.sum_horizontal([v.fill_null(0.0) for v in vals])
    return pl.when(n > 0).then(s / n).otherwise(None)


def main():
    ap = argparse.ArgumentParser()
    for k in ("train", "test", "data", "run6", "fusion_accepted", "targeted", "out"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cols = ["qid", "tid", "newest", "graph", "co"]
    refs = pl.read_parquet(os.path.join(a.data, "train", "ref.parquet"), columns=["rid", "fold", "deg", "co"])
    tr = (pl.scan_parquet(a.train).select(cols + ["y"]).filter(pl.col("co").str.to_lowercase() != "france")
            .join(refs.lazy().select(pl.col("rid").alias("qid"), "fold"), on="qid", how="left")
            .filter(pl.col("fold").is_in([0, 1])).collect())
    val = pl.read_parquet(os.path.join(a.run6, "val_pred.parquet"), columns=["qid", "tid", "p2"]).rename({"p2": "friend"})
    tr = tr.join(val, on=["qid", "tid"], how="left")
    print("train pairs", tr.height, "friend coverage", tr["friend"].is_not_null().mean(), flush=True)
    te = pl.scan_parquet(a.test).select(cols).filter(pl.col("co").str.to_lowercase() == "france").collect()
    tp = pl.read_parquet(os.path.join(a.run6, "test_pred.parquet"), columns=["qid", "tid", "p2"]).rename({"p2": "friend"})
    fr_ids = te.select("qid").unique()
    te = te.join(tp.join(fr_ids, on="qid", how="semi"), on=["qid", "tid"], how="full", coalesce=True)
    print("france pairs", te.height, {c: int(te[c].is_not_null().sum()) for c in ("newest", "friend", "graph")}, flush=True)

    anchors = {f: refs.filter((pl.col("fold") == f) & (pl.col("co").str.to_lowercase() != "france"))
                      .select(pl.col("rid").alias("qid"), "deg") for f in (0, 1)}
    fus = pl.read_parquet(a.fusion_accepted).select("qid", "tid")
    targeted = pl.read_parquet(a.targeted).select("qid", "tid")
    tset = targeted.join(fr_ids, on="qid", how="semi")
    stats = {}
    for name, srcs in VARIANTS.items():
        d = tr.with_columns(mean_available(srcs).alias("p")).filter(pl.col("p").is_not_null())
        tune = decode.tune(d.filter(pl.col("fold") == 0).select("qid", "tid", "p", "y"), anchors[0])
        rule, thr = tune["rule"], tune["threshold"]
        acc1 = decode.apply(rule, d.filter(pl.col("fold") == 1).select("qid", "tid", "p", "y"), thr)
        audit = decode.score(acc1, anchors[1])
        f = te.with_columns(mean_available(srcs).alias("p")).filter(pl.col("p").is_not_null())
        acc = decode.apply(rule, f.select("qid", "tid", "p"), thr).select("qid", "tid")
        per = fr_ids.join(acc.group_by("qid").len(), on="qid", how="left").with_columns(pl.col("len").fill_null(0))
        both = acc.join(tset, on=["qid", "tid"], how="inner").height
        stats[name] = {"rule": rule, "threshold": thr, "tune_f05": tune["macro_f05"], "audit_usin": audit,
                       "france_pairs": acc.height, "france_per_s1": acc.height / fr_ids.height,
                       "france_zero_share": float((per["len"] == 0).mean()),
                       "overlap_with_targeted": both, "only_variant": acc.height - both, "only_targeted": tset.height - both}
        print(name, json.dumps(stats[name]), flush=True)
        full = pl.concat([fus.join(fr_ids, on="qid", how="anti"), acc])
        full.write_parquet(os.path.join(a.out, f"accepted_{name}.parquet"))
    with open(os.path.join(a.out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
