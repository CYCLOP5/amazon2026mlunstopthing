'Stages features_train / features_test'
import glob
import os
import time

import polars as pl

from er.blocking.filters import final_candidate_filter
from er.features import ATTRS, build_features
from er.features.context import add_context, s1_aggregates
from er.io import load_gt_pairs, s1_bucket
from er.safe import guard


def is_done(cfg, paths, split):
    if split == "test" and not cfg["features"]["cache_test"]:
        return True
    return os.path.exists(os.path.join(paths.features(split), ".done"))


def feature_chunks(cfg, paths, split, skip=lambda name: False):
    'yields (chunk_name, part_index, features dataframe) over the final candidate set'
    fc, tr = cfg["features"], cfg["training"]
    block_dir = paths.blocking(split)
    s1agg = s1_aggregates(os.path.join(block_dir, "*.parquet"))
    cols = ["entity_id", "country"] + ATTRS
    s1_lf = pl.scan_parquet(paths.prep_file(split, 1)).select(cols)
    oth_lf = pl.concat([pl.scan_parquet(paths.prep_file(split, i)).select(cols) for i in (2, 3)])


    names1 = pl.scan_parquet(paths.prep_file(split, 1)).select("entity_id", "country", "name_core").collect()
    name_freq = names1.group_by("country", "name_core").agg(pl.len().cast(pl.Int32).alias("nfreq"))
    s1_freq = names1.join(name_freq, on=["country", "name_core"]).select(
        pl.col("entity_id").alias("s1_id"), pl.col("nfreq").alias("s1_name_freq"))
    del names1
    for cf in sorted(glob.glob(os.path.join(block_dir, "*.parquet"))):
        name = os.path.basename(cf).replace(".parquet", "")
        if skip(name):
            continue
        c = add_context(pl.read_parquet(cf), s1agg)
        c = c.filter(final_candidate_filter(cfg["candidates"], c.columns))
        if split == "train":
            c = c.filter(s1_bucket(tr["hash_seed"]) < tr["sample_pct"])
        if c.height == 0:
            continue

        s1 = s1_lf.filter(pl.col("entity_id").is_in(c["s1_id"].unique().implode())).collect()
        oth = oth_lf.filter(pl.col("entity_id").is_in(c["o_id"].unique().implode())).collect()
        o_freq = (oth.select(pl.col("entity_id").alias("o_id"), "country", "name_core")
                     .join(name_freq, on=["country", "name_core"], how="left")
                     .select("o_id", pl.col("nfreq").fill_null(0).alias("o_name_s1freq")))
        c = (c.join(s1_freq, on="s1_id", how="left", maintain_order="left")
              .join(o_freq, on="o_id", how="left", maintain_order="left"))
        for k, off in enumerate(range(0, c.height, fc["part_rows"])):
            feat = build_features(c.slice(off, fc["part_rows"]), s1, oth, fc["groups"])
            guard("features")
            yield name, k, feat
        del c, s1, oth


def run(cfg, paths, split):
    t0 = time.time()
    out_dir = paths.features(split)
    os.makedirs(out_dir, exist_ok=True)
    gt = None
    if split == "train":
        gt = load_gt_pairs(paths.raw("train", "gt")).with_columns(pl.lit(1, pl.Int8).alias("label"))

    def done(name):
        return os.path.exists(os.path.join(out_dir, f"{name}.ok"))

    last, n = None, 0
    for name, k, feat in feature_chunks(cfg, paths, split, skip=done):
        if gt is not None:
            feat = (feat.join(gt, on=["s1_id", "o_id"], how="left", maintain_order="left")
                        .with_columns(pl.col("label").fill_null(0)))
        feat.write_parquet(os.path.join(out_dir, f"{name}_{k:02d}.parquet"))
        if last is not None and last != name:
            open(os.path.join(out_dir, f"{last}.ok"), "w").write("ok")
        last = name
        n += feat.height
        pos = f" (pos {int(feat['label'].sum()):,})" if gt is not None else ""
        print(f" {name} part {k}: {feat.height:,} pairs{pos}  t={time.time()-t0:.0f}s", flush=True)
    if last is not None:
        open(os.path.join(out_dir, f"{last}.ok"), "w").write("ok")
    open(os.path.join(out_dir, ".done"), "w").write("ok")
    print(f"{split} features: {n:,} pairs written this run, {time.time()-t0:.0f}s")
