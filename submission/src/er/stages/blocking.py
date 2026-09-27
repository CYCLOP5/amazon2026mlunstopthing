'Stages blocking_train / blocking_test: run the configured blocker (resumable per chunk)'
import os
import re
import time

import polars as pl

from er.blocking import BLOCKERS


def is_done(cfg, paths, split):
    return os.path.exists(os.path.join(paths.blocking(split), ".done"))


def run(cfg, paths, split):
    out_dir = paths.blocking(split)
    os.makedirs(out_dir, exist_ok=True)
    cols = ["entity_id", "country", "name_skel", "addr_skel"]
    s1 = pl.read_parquet(paths.prep_file(split, 1), columns=cols)
    oth = pl.concat([pl.read_parquet(paths.prep_file(split, i), columns=cols) for i in (2, 3)])

    def out_path(country, k):
        return os.path.join(out_dir, f"{re.sub(r'[^A-Za-z0-9]+', '_', country)}_{k:03d}.parquet")

    t0 = time.time()
    BLOCKERS.get(cfg["blocking"]["method"])(s1, oth, cfg["blocking"], out_path)
    open(os.path.join(out_dir, ".done"), "w").write("ok")
    print(f"{split}: blocking finished in {time.time() - t0:.0f}s")
