'write matching_results.tsv from accepted (qid, tid) pairs (e.g. calibrate.py output)'
import os
import shutil
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from er.io import write_id_lists  # noqa: E402


def main(ref, ds, acc_path, out, cand):
    os.makedirs(out, exist_ok=True)
    r = pl.read_parquet(ref, columns=["rid", "eid"]).sort("rid")
    parts, off = [], 0
    for s in ("2", "3"):
        x = pl.read_csv(os.path.join(ds, "test", f"test_source{s}.tsv"), separator="\t", quote_char=None,
                        columns=["entity_id"]).with_row_index("tid")
        parts.append(x.with_columns((pl.col("tid") + off).cast(pl.UInt32)))
        off += x.height
    t = pl.concat(parts).rename({"entity_id": "o_id"})
    acc = pl.read_parquet(acc_path).select("qid", "tid")
    m = (acc.join(r.select(pl.col("rid").alias("qid"), pl.col("eid").alias("s1_id")), on="qid")
            .join(t, on="tid").select("s1_id", "o_id"))
    assert m.height == acc.height, (m.height, acc.height)
    c = (pl.read_csv(cand, separator="\t", schema_overrides={"candidate_entity_ids": pl.Utf8})
           .with_columns(pl.col("candidate_entity_ids").fill_null("").str.split(",")).explode("candidate_entity_ids")
           .rename({"source1_entity_id": "s1_id", "candidate_entity_ids": "o_id"}))
    missing = m.join(c, on=["s1_id", "o_id"], how="anti")
    write_id_lists(m, r["eid"], "matched_entity_ids", os.path.join(out, "matching_results.tsv"))
    if missing.height:
        print(f"adding {missing.height} matches missing from the candidate file", flush=True)
        write_id_lists(pl.concat([c.filter(pl.col("o_id") != ""), missing]), r["eid"], "candidate_entity_ids",
                       os.path.join(out, "candidate_pairs.tsv"))
    elif os.path.abspath(cand) != os.path.abspath(os.path.join(out, "candidate_pairs.tsv")):
        shutil.copyfile(cand, os.path.join(out, "candidate_pairs.tsv"))
    print(f"wrote {out}: {m.height:,} matches", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])
