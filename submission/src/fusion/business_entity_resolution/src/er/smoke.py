'build a small, consistent sample of the real challenge data for a quick end-to-end check'
import os

import polars as pl

from er.io import read_tsv


def _keep(col, pct):
    return (pl.col(col).hash(seed=99) % 100) < pct


def _write(df: pl.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.write_csv(path, separator="\t", quote_style="never", null_value="")


def make_smoke_data(src_data: str, out_data: str, pct: int = 2):
    tr = os.path.join(src_data, "train")
    s1 = read_tsv(os.path.join(tr, "train_source1.tsv")).filter(_keep("entity_id", pct))
    gt = read_tsv(os.path.join(tr, "train_ground_truth.tsv")).filter(
        pl.col("source1_entity_id").is_in(s1["entity_id"].implode()))
    matched = (gt.select(pl.col("matched_entity_ids").fill_null("").str.split(","))
                 .explode("matched_entity_ids")["matched_entity_ids"])
    _write(s1, os.path.join(out_data, "train", "train_source1.tsv"))
    _write(gt, os.path.join(out_data, "train", "train_ground_truth.tsv"))
    for i in (2, 3):
        d = read_tsv(os.path.join(tr, f"train_source{i}.tsv"))
        _write(d.filter(pl.col("entity_id").is_in(matched.implode()) | _keep("entity_id", pct)),
               os.path.join(out_data, "train", f"train_source{i}.tsv"))
        del d
    for i in (1, 2, 3):
        d = read_tsv(os.path.join(src_data, "test", f"test_source{i}.tsv")).filter(_keep("entity_id", pct))
        _write(d, os.path.join(out_data, "test", f"test_source{i}.tsv"))
    print(f"smoke data written to {out_data}: train S1={s1.height:,}")
