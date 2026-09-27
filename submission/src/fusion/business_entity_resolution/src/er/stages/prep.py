'stage prep: normalise every source file of both splits (resumable per file)'
import os
import time

import polars as pl

from er.io import read_tsv
from er.normalize import normalize_records


def is_done(cfg, paths, split=None):
    return all(os.path.exists(paths.prep_file(s, i)) for s in ("train", "test") for i in (1, 2, 3))


def run(cfg, paths, split=None):
    os.makedirs(paths.prep, exist_ok=True)
    rows = cfg["prep"]["chunk_rows"]
    for s in ("train", "test"):
        for i in (1, 2, 3):
            out = paths.prep_file(s, i)
            if os.path.exists(out):
                print(f" {s} s{i}: already done", flush=True)
                continue
            t0 = time.time()
            raw = read_tsv(paths.raw(s, i))
            parts = [normalize_records(raw.slice(o, rows)) for o in range(0, raw.height, rows)]
            pl.concat(parts).write_parquet(out + ".tmp")
            os.replace(out + ".tmp", out)
            print(f" {s} s{i}: {raw.height:,} records in {time.time() - t0:.0f}s", flush=True)
            del raw, parts
