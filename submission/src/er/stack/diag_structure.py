'label-free structural test for doubtful matches (runs on azure, prepared data)'
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from er.stack import decode  # noqa: E402
from er.stack.calibrate import segments  # noqa: E402

LEGAL = ["sarl", "sas", "sa", "sasu", "eurl", "sci", "ei", "snc", "scop", "llc", "inc", "ltd", "pvt", "private",
         "limited", "co", "corp", "company", "lp", "llp", "pllc", "pc", "plc", "the", "l", "c", "s", "a", "r"]


def toks(c):
    return (pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()
              .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" ")
              .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(LEGAL))).list.unique())


def name_pattern(d):
    d = d.with_columns(toks("rec_name").alias("_a"), toks("s1_name").alias("_b"))
    d = d.with_columns(pl.col("_a").list.set_difference("_b").alias("_da"), pl.col("_b").list.set_difference("_a").alias("_db"))
    d = d.with_columns(pl.col("_da").list.len().alias("_na"), pl.col("_db").list.len().alias("_nb"),
                       pl.col("_da").list.first().alias("wa"), pl.col("_db").list.first().alias("wb"))
    pat = (pl.when((pl.col("_na") == 0) & (pl.col("_nb") == 0)).then(pl.lit("same core"))
             .when((pl.col("_na") == 1) & (pl.col("_nb") == 1)).then(pl.lit("1 word swapped"))
             .when((pl.col("_na") == 1) & (pl.col("_nb") == 0)).then(pl.lit("1 word added"))
             .when((pl.col("_na") == 0) & (pl.col("_nb") == 1)).then(pl.lit("1 word dropped")).otherwise(pl.lit("other")))
    return d.with_columns(pat.alias("name_pat")).drop("_a", "_b", "_da", "_db", "_na", "_nb")


def direction(top):
    'swap direction per (co, wa, wb) over the pool of best candidates (label-free)'
    sw = top.filter(pl.col("name_pat") == "1 word swapped")
    cnt = sw.group_by("co", "wa", "wb").len("n")
    rev = cnt.rename({"wa": "wb", "wb": "wa", "n": "n_rev"})
    return (cnt.join(rev, on=["co", "wa", "wb"], how="left").with_columns(pl.col("n_rev").fill_null(0))
               .with_columns(pl.when(pl.col("n") + pl.col("n_rev") < 20).then(pl.lit("rare"))
                               .when(pl.col("n") / (pl.col("n") + pl.col("n_rev")) >= 0.9).then(pl.lit("one-way"))
                               .when(pl.col("n") / (pl.col("n") + pl.col("n_rev")) < 0.7).then(pl.lit("two-way"))
                               .otherwise(pl.lit("mid")).alias("dir")).select("co", "wa", "wb", "dir"))


def buckets(split, pred, ref, tg, lab_col):
    'accepted pairs with bucket labels + per-s1 other-match counts'
    acc = decode.apply("expected_f", pred.select("qid", "tid", "p"), 0.5).select("qid", "tid")
    top = (decode.top1(pred.select("qid", "tid", "p")).join(ref.select("qid", "co", "s1_name"), on="qid")
             .join(tg.select("tid", "rec_name", "rec_addr", "sr"), on="tid"))
    top = name_pattern(top)
    top = top.join(direction(top), on=["co", "wa", "wb"], how="left")
    a = acc.join(top.drop("p"), on=["qid", "tid"], how="inner")
    a = segments(a, ref.select("qid", "ad"), tg.select("tid", pl.col("rec_addr").alias("ad")))
    noaddr = pl.col("rec_addr").fill_null("").str.strip_chars() == ""
    b = (pl.when(noaddr & (pl.col("name_pat") == "same core")).then(pl.lit("no address, same core"))
           .when(noaddr).then(pl.lit("no address, other name"))
           .when((pl.col("seg") == "equal") & (pl.col("name_pat") == "same core")).then(pl.lit("same house, same core"))
           .when((pl.col("seg") == "equal") & (pl.col("name_pat") == "1 word swapped") & (pl.col("dir") == "two-way")).then(pl.lit("same house, 2-way swap"))
           .when((pl.col("seg") == "equal") & (pl.col("name_pat") == "1 word swapped")).then(pl.lit("same house, 1-way/rare swap"))
           .when((pl.col("seg") == "equal") & (pl.col("name_pat") == "other")).then(pl.lit("same house, other name"))
           .when(pl.col("seg") == "equal").then(pl.lit("same house, word added/dropped"))
           .when(pl.col("seg") == "conflict").then(pl.lit("house conflict"))
           .otherwise(pl.lit("unknown house")))
    a = a.with_columns(b.alias("bucket"))
    per = acc.join(tg.select("tid", "sr"), on="tid").group_by("qid").agg(pl.len().alias("n_all"),
                                                                        (pl.col("sr") == 2).sum().alias("n2"))
    a = a.join(per, on="qid").with_columns((pl.col("n_all") - 1).alias("other"),
                                          (pl.when(pl.col("sr") == 2).then(pl.col("n2")).otherwise(pl.col("n_all") - pl.col("n2")) - 1).alias("other_src"))
    if lab_col:
        a = a.join(pred.select("qid", "tid", "y"), on=["qid", "tid"], how="left")
    return a, per


def main(data, stack_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rep = {}
    lines = []
    for split in ("train", "test"):
        cols = ["rid", "nm", "ad", "co"] + (["fold"] if split == "train" else [])
        ref = pl.read_parquet(os.path.join(data, split, "ref.parquet"), columns=cols).rename(
            {"rid": "qid", "nm": "s1_name"}).with_columns(pl.col("qid").cast(pl.UInt32), pl.col("co").str.to_lowercase())
        tg = pl.concat([pl.read_parquet(os.path.join(data, split, f"s{s}.parquet"), columns=["rid", "nm", "ad"])
                          .with_columns(pl.lit(s, pl.UInt8).alias("sr")) for s in (2, 3)]).rename(
            {"rid": "tid", "nm": "rec_name", "ad": "rec_addr"}).with_columns(pl.col("tid").cast(pl.UInt32))
        if split == "train":
            pr = pl.read_parquet(os.path.join(stack_dir, "val_pred.parquet"), columns=["qid", "tid", "y", "p2"]).rename({"p2": "p"})
            a, per = buckets(split, pr, ref, tg, True)
            a = a.join(ref.filter(pl.col("fold") == 1).select("qid"), on="qid")
            n_s1 = ref.filter(pl.col("fold") == 1).group_by("co").len()
        else:
            pr = pl.read_parquet(os.path.join(stack_dir, "test_pred.parquet"), columns=["qid", "tid", "p2"]).rename({"p2": "p"})
            a, per = buckets(split, pr, ref, tg, False)
            n_s1 = ref.group_by("co").len()

        agg = [pl.len().alias("pairs"), pl.col("other").mean().round(3).alias("other_mean"),
               pl.col("other_src").mean().round(3).alias("other_same_src")]
        base = a.group_by("co").agg(agg).with_columns(pl.lit("ALL accepted").alias("bucket"))
        if split == "train":
            t = a.group_by("co", "bucket", "y").agg(agg).sort("co", "bucket", "y")
        else:
            t = a.group_by("co", "bucket").agg(agg).sort("co", "bucket")
        lines.append(f"## {split}\n")
        with pl.Config(tbl_rows=200, tbl_width_chars=220):
            lines.append(str(base.sort("co")))
            lines.append(str(t))
        rep[split] = {"base": base.to_dicts(), "buckets": t.to_dicts()}
        print("\n".join(lines[-3:]), flush=True)
    open(os.path.join(out_dir, "diag_structure.md"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump(rep, open(os.path.join(out_dir, "diag_structure.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main(*sys.argv[1:4])
