'stage eval_blocking: pair recall and oracle macro f0.5 of the train candidates (chunk by chunk)'
import glob
import os

import polars as pl

from er.blocking.filters import final_candidate_filter
from er.io import load_gt_pairs


def _report(paths):
    return os.path.join(paths.blocking("train"), "eval.txt")


def is_done(cfg, paths, split=None):
    return os.path.exists(_report(paths))


def run(cfg, paths, split=None):
    gt = load_gt_pairs(paths.raw("train", "gt")).with_columns(pl.lit(1, pl.Int8).alias("y"))
    n_s1 = pl.scan_parquet(paths.prep_file("train", 1)).select(pl.len()).collect().item()
    rules = {f"top{k}": pl.col("brank") < k for k in (1, 2, 3, 5, 10, 15, 20, 30) if k <= cfg["blocking"]["top_k"]}
    rules["final (model input)"] = None
    hits = {k: [] for k in rules}
    n_pairs = {k: 0 for k in rules}
    for f in sorted(glob.glob(os.path.join(paths.blocking("train"), "*.parquet"))):
        g = (pl.read_parquet(f).with_columns(pl.col("bscore").max().over("o_id").alias("o_bmax"))
               .join(gt, on=["s1_id", "o_id"], how="left"))
        for k, e in rules.items():
            x = g.filter(e if e is not None else final_candidate_filter(cfg["candidates"], g.columns))
            n_pairs[k] += x.height
            hits[k].append(x.filter(pl.col("y") == 1).select("s1_id"))
    lines = [f"GT pairs {gt.height:,}  Source-1 entities {n_s1:,}"]
    per_true = gt.group_by("s1_id").agg(pl.len().alias("n"))
    for k in rules:
        h = pl.concat(hits[k]).group_by("s1_id").agg(pl.len().alias("h"))
        per = per_true.join(h, on="s1_id", how="left").fill_null(0)
        r = per["h"] / per["n"]
        oracle = ((1.25 * r / (0.25 + r)).fill_nan(0).sum() + (n_s1 - per.height)) / n_s1
        lines.append(f" {k:22s} pair-recall={per['h'].sum() / gt.height:.4f}  oracle F0.5={oracle:.4f}  "
                     f"pairs={n_pairs[k]:,}")
    print("\n".join(lines))
    open(_report(paths), "w").write("\n".join(lines) + "\n")
