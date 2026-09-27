'decoy words vs alias noise by position in the record name'
import os
import sys

import polars as pl

LEGAL = (r"(?:s\.?a\.?r\.?l|s\.?a\.?s\.?u|s\.?a\.?s|e\.?u\.?r\.?l|s\.?c\.?i|s\.?n\.?c|s\.?a|e\.?i|llc|l\.l\.c|inc|ltd|corp|"
         r"corporation|company|co|lp|llp|pllc|pc|plc|pvt\.?\s*ltd|private\s+limited|limited)")
FR_DECOY = ["groupe", "holding", "international", "distribution", "participations", "developpement"]
EN_WORDS = ["group", "holdings", "partners", "enterprises", "ventures", "industries", "solutions", "international",
            "services", "service", "center", "associates"]
FR_ALIAS = ["fils", "associes", "services", "cie"]


def low(c):
    return pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()


def position(w):
    after = low("rec_name").str.contains(r"\b" + LEGAL + r"\.?\W*(?:&\s*|et\s+|and\s+)?\(?\[?" + w + r"\b\W*$")
    before = low("rec_name").str.contains(r"\b" + w + r"\b\W*" + LEGAL + r"\b")
    has_legal = low("rec_name").str.contains(r"\b" + LEGAL + r"\b")
    at_end = low("rec_name").str.contains(r"\b" + w + r"\W*$")
    return (pl.when(after).then(pl.lit("after legal")).when(before).then(pl.lit("before legal"))
              .when(has_legal).then(pl.lit("inside, legal elsewhere"))
              .when(at_end).then(pl.lit("no legal, at end")).otherwise(pl.lit("no legal, inside")))


def tag(k, words):
    parts = []
    for w in words:
        x = k.filter(low("rec_name").str.contains(r"\b" + w + r"\b") & ~low("s1_name").str.contains(r"\b" + w + r"\b"))
        parts.append(x.with_columns(pl.lit(w).alias("w"), position(w).alias("pos")))
    return pl.concat(parts)


def main(keys_dir, stack_dir, data, accepted, out):
    os.makedirs(out, exist_ok=True)
    pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(60)

    y = pl.read_parquet(os.path.join(stack_dir, "val_pred.parquet"), columns=["qid", "tid", "y"])
    tr = pl.read_parquet(os.path.join(keys_dir, "keys_train.parquet"), columns=["qid", "tid", "co", "acc", "rec_name", "s1_name"])
    tr = tr.join(y, on=["qid", "tid"], how="left")
    t = tag(tr, EN_WORDS).group_by("w", "pos").agg(pl.len().alias("n"), pl.col("y").mean().alias("true"), pl.col("acc").mean().alias("acc"))
    print("US/India labels: record-only word, truth by position", flush=True)
    print(t.sort("w", "pos"), flush=True)
    t.write_parquet(os.path.join(out, "usin_position.parquet"))
    del tr, y

    te = pl.read_parquet(os.path.join(keys_dir, "keys_test.parquet"), columns=["qid", "tid", "acc", "rec_name", "s1_name"])
    f = tag(te, FR_DECOY + FR_ALIAS)
    print("France pool (T2 decisions): by word and position", flush=True)
    print(f.group_by("w", "pos").agg(pl.len().alias("pool"), pl.col("acc").sum().alias("acc_T2")).sort("w", "pos"), flush=True)
    acc = pl.read_parquet(accepted).select("qid", "tid")
    fa = f.drop("acc").join(acc, on=["qid", "tid"]).filter(pl.col("w").is_in(FR_DECOY))
    print("France accepted in the given submission, decoy words by position:", flush=True)
    print(fa.group_by("w", "pos").len().sort("w", "pos"), flush=True)

    ref = pl.read_parquet(os.path.join(data, "test", "ref.parquet"), columns=["rid", "eid"]).rename({"rid": "qid", "eid": "s1_id"})
    tg = pl.concat([pl.read_parquet(os.path.join(data, "test", f"s{s}.parquet"), columns=["rid", "eid"]) for s in (2, 3)]).rename(
        {"rid": "tid", "eid": "o_id"})
    ref = ref.with_columns(pl.col("qid").cast(pl.UInt32)); tg = tg.with_columns(pl.col("tid").cast(pl.UInt32))
    ids = fa.unique(["qid", "tid"]).join(ref, on="qid").join(tg, on="tid")
    ids.select("qid", "tid", "s1_id", "o_id", "w", "pos", "rec_name", "s1_name").write_parquet(os.path.join(out, "fr_decoy_accepted.parquet"))
    print("wrote", ids.height, "accepted French decoy-word pairs with positions", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])
