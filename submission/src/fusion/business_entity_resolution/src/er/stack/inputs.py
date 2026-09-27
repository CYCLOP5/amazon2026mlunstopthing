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
        coverage = np.load(cov, allow_pickle=False)
        if coverage.ndim != 1 or coverage.dtype.kind not in "iu":
            raise ValueError(f"{cov}: coverage must be a one-dimensional integer array")
        if len(coverage) and (coverage.min() < 0 or coverage.max() >= n_targets):
            raise ValueError(f"{cov}: target ID outside [0, {n_targets})")
        if np.unique(coverage).size != coverage.size:
            raise ValueError(f"{cov}: duplicate target IDs in coverage")
        if d.filter(~pl.col("tid").is_in(pl.Series(coverage).implode())).height:
            raise ValueError(f"{pq}: pair target missing from declared coverage")
        for col in SCORE_COLS:
            if col not in d.columns or d[col].null_count() or not d[col].is_finite().all():
                raise ValueError(f"{pq}: missing/nonfinite score {col}")
            if not d[col].is_between(0, 1).all():
                raise ValueError(f"{pq}: score {col} outside [0, 1]")
        return d.select(keep), coverage

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
    if not chunks:
        raise ValueError("No candidate parts were loaded")
    scores = [c for c in SCORE_COLS if c in chunks[0].columns]
    pairs = pl.concat(chunks, how="vertical_relaxed").with_columns(
        pl.col("qid", "tid").cast(pl.UInt32), pl.col(scores).cast(pl.Float32))
    info = {"runs": len(runs), "parts": len(parts), "duplicate_parts": dup_parts, "pairs": pairs.height,
            "targets": n_targets, "uncovered_targets": missing, "config_sha256": next(iter(configs)),
            "score_columns": [c for c in SCORE_COLS if c in pairs.columns]}
    if pairs.select("qid", "tid").n_unique() != pairs.height:
        raise SystemExit("duplicate (qid, tid) pairs after de-duplication")
    return pairs.sort("tid", "qid"), info
