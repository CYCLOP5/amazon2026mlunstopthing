'reading the prepared records and the externally scored candidate pairs'
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl

SCORE_COLS = ("prob", "gate_prob", "neural_prob")


def load_refs(data: str, split: str) -> pl.DataFrame:
    cols = ["rid", "eid", "nm", "ad", "co"] + (["fold", "deg"] if split == "train" else [])
    return pl.read_parquet(os.path.join(data, split, "ref.parquet"), columns=cols).with_columns(
        pl.col("rid").cast(pl.UInt32))


def load_targets(data: str, split: str) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(os.path.join(data, split, f"s{sr}.parquet"),
                                      columns=["rid", "eid", "nm", "ad", "co"])
                        .with_columns(pl.col("rid").cast(pl.UInt32), pl.lit(sr, pl.UInt8).alias("sr"))
                      for sr in (2, 3)])


def add_lexical(pairs: pl.DataFrame, lex_dirs: list, refs: pl.DataFrame, tg: pl.DataFrame,
                top_k: int, min_rel: float) -> tuple[pl.DataFrame, dict]:
    'union the neural candidates with our own blocking candidates (er.blocking output:'
    files = [f for d in lex_dirs for f in glob.glob(os.path.join(d, "*.parquet"))]
    if not files:
        raise SystemExit(f"no blocking candidates in {lex_dirs}")
    lx = (pl.scan_parquet(files).select("s1_id", "o_id", "bscore", "brank", "cos_name", "cos_addr")
            .with_columns(pl.col("bscore").max().over("o_id").alias("_best"))
            .filter((pl.col("brank") < top_k) & (pl.col("bscore") >= min_rel * pl.col("_best")))
            .drop("_best").collect())
    e1 = refs.select(pl.col("eid").alias("s1_id"), pl.col("rid").alias("qid"))
    e2 = tg.select(pl.col("eid").alias("o_id"), pl.col("rid").alias("tid"))
    lx = lx.join(e1, on="s1_id", how="inner").join(e2, on="o_id", how="inner").drop("s1_id", "o_id").with_columns(
        pl.col("bscore", "cos_name", "cos_addr").cast(pl.Float32), pl.col("brank").cast(pl.Int16))
    scores = [c for c in SCORE_COLS if c in pairs.columns]
    u = (pairs.with_columns(pl.lit(1, pl.Int8).alias("in_neural"))
              .join(lx, on=["qid", "tid"], how="full", coalesce=True)
              .with_columns(pl.col("in_neural").fill_null(0), *[pl.col(c).fill_null(1e-6) for c in scores]))
    info = {"lexical_pairs": lx.height, "lexical_only_pairs": int((u["in_neural"] == 0).sum()),
            "union_pairs": u.height}
    return u.sort("tid", "qid"), info


def add_external(pairs: pl.DataFrame, files: list, name: str, col: str) -> tuple[pl.DataFrame, dict]:
    "union with another model's scored pairs (parquet: qid, tid, <col>). its score becomes feature column"
    ext = (pl.scan_parquet(files).select(pl.col("qid").cast(pl.UInt32), pl.col("tid").cast(pl.UInt32),
                                        pl.col(col).cast(pl.Float32).alias(name))
             .unique(["qid", "tid"]).collect())
    scores = [c for c in SCORE_COLS if c in pairs.columns]
    u = (pairs.with_columns(pl.lit(1, pl.Int8).alias("in_base"))
              .join(ext, on=["qid", "tid"], how="full", coalesce=True)
              .with_columns(pl.col("in_base").fill_null(0), pl.col(name).fill_null(1e-4),
                            *[pl.col(c).fill_null(1e-6) for c in scores]))
    if "in_neural" in u.columns:
        u = u.with_columns(pl.col("in_neural").fill_null(0))
    info = {f"{name}_pairs": ext.height, f"{name}_only_pairs": int((u["in_base"] == 0).sum()), "union_pairs": u.height}
    return u.sort("tid", "qid"), info


def find_runs(roots: list) -> list:
    'Every scored run folder (one containing manifest.json + parts/) below the given roots'
    runs = []
    for r in roots:
        for m in glob.glob(os.path.join(r, "**", "manifest.json"), recursive=True):
            d = os.path.dirname(m)
            if os.path.isdir(os.path.join(d, "parts")):
                runs.append(d)
    return sorted(set(runs))


def load_pairs(roots: list, n_targets: int, threads: int = 8,
               require_complete: bool = True) -> tuple[pl.DataFrame, dict]:
    'All scored pairs of a split. Runs may overlap (restarted / redistributed jobs): each'
    runs = find_runs(roots)
    if not runs:
        raise SystemExit(f"no scored runs (manifest.json + parts/) below {roots}")
    parts, configs = [], {}
    for d in runs:
        m = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
        configs.setdefault(m.get("config_sha256"), []).append(d)
        for p in m["parts"]:
            parts.append((os.path.join(d, p["name"]), os.path.join(d, p["coverage"]), p["pairs"]))
    if len(configs) != 1:
        raise SystemExit(f"scored runs come from {len(configs)} different model configurations: "
                         f"{ {k: v[:2] for k, v in configs.items()} }")

    def read(item):
        pq, cov, n = item
        d = pl.read_parquet(pq)
        if d.height != n:
            raise SystemExit(f"{pq}: {d.height} pairs, manifest says {n}")
        keep = [c for c in ("qid", "tid", *SCORE_COLS, "y") if c in d.columns]
        return d.select(keep), np.load(cov, allow_pickle=False)

    covered = np.zeros(n_targets, dtype=bool)
    chunks, dup_parts, n_read = [], 0, 0
    with ThreadPoolExecutor(threads) as ex:
        for d, cov in ex.map(read, parts):
            n_read += 1
            new = ~covered[cov]
            if not new.any():
                dup_parts += 1
                continue
            if not new.all():
                d = d.filter(pl.col("tid").is_in(pl.Series(cov[new]).implode()))
            covered[cov[new]] = True
            chunks.append(d)
            if n_read % 500 == 0:
                print(f"  read {n_read:,}/{len(parts):,} parts", flush=True)
    missing = int((~covered).sum())
    if missing and require_complete:
        raise SystemExit(f"scored runs do not cover {missing:,} of {n_targets:,} target records "
                         f"(add the missing job output folders to the roots)")
    scores = [c for c in SCORE_COLS if c in chunks[0].columns]
    pairs = pl.concat(chunks, how="vertical_relaxed").with_columns(
        pl.col("qid", "tid").cast(pl.UInt32), pl.col(scores).cast(pl.Float32))
    info = {"runs": len(runs), "parts": len(parts), "duplicate_parts": dup_parts, "pairs": pairs.height,
            "targets": n_targets, "uncovered_targets": missing, "config_sha256": next(iter(configs)),
            "score_columns": [c for c in SCORE_COLS if c in pairs.columns]}
    if pairs.select("qid", "tid").n_unique() != pairs.height:
        raise SystemExit("duplicate (qid, tid) pairs after de-duplication")
    return pairs.sort("tid", "qid"), info
