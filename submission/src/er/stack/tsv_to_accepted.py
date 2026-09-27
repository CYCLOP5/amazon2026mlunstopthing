"convert any matching_results.tsv (e.g. a team's submission) into accepted.parquet (qid, tid) in our id space,"
import sys

import polars as pl


def main(tsv, data, out):
    m = (pl.read_csv(tsv, separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.Utf8})
           .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "rec"}))
    ref = pl.read_parquet(f"{data}/test/ref.parquet", columns=["rid", "eid"]).select(
        pl.col("rid").cast(pl.UInt32).alias("qid"), pl.col("eid").alias("s1"))
    tg = pl.concat([pl.read_parquet(f"{data}/test/s{s}.parquet", columns=["rid", "eid"]) for s in (2, 3)]).select(
        pl.col("rid").cast(pl.UInt32).alias("tid"), pl.col("eid").alias("rec"))
    a = m.join(ref, on="s1").join(tg, on="rec").select("qid", "tid").sort("qid", "tid")
    assert a.height == m.height, f"{m.height - a.height} pairs did not map to ids"
    a.write_parquet(out)
    print(f"{tsv}: {m.height:,} pairs -> {out}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
