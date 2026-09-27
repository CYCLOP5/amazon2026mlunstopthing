'Fine-grained pattern scan: where does France accept far more than the US/India labelled true rate?'
import os
import sys

import polars as pl
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from er.stack import decode  # noqa: E402

LEGAL = {"sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci", "ei": "ei", "snc": "snc",
         "llc": "llc", "inc": "inc", "corp": "corp", "corporation": "corp", "ltd": "ltd", "limited": "ltd", "lp": "lp",
         "llp": "llp", "pllc": "pllc", "pc": "pc", "plc": "plc", "pvt": "pvt", "private": "pvt", "co": "co", "company": "co"}
DROPW = set(LEGAL) | {"the", "l", "c", "s", "a", "r"}
STOP = {"de", "du", "des", "la", "le", "les", "et", "d", "l", "of", "and", "the"}
DBA = r"doing business as|formerly|f/k/a|\bfka\b|\bt/a\b|\baka\b|\bdba\b|trading as|operating as|known as"


def norm(c):
    return pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()


def toks(c):
    return (norm(c).str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" ")
              .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(list(DROPW)))).list.unique())


def legal(c):
    t = (norm(c).str.replace_all(r"(?:\b([a-z])\.\s?)", "$1").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().str.split(" "))
    return t.list.eval(pl.element().filter(pl.element().is_in(list(LEGAL))).replace(LEGAL)).list.unique().list.sort().list.join(",")


def nums(c):
    return norm(c).str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).list.eval(pl.element().filter(pl.element() != ""))


def street(c):
    return (norm(c).str.replace_all(r"[^a-z]+", " ").str.strip_chars().str.split(" ")
              .list.eval(pl.element().filter((pl.element().str.len_chars() >= 4)))).list.unique()


def base_frame(pairs, ref, tg):
    d = pairs.join(ref.select("qid", "co", "s1_name", "s1_addr"), on="qid").join(tg, on="tid")
    d = d.with_columns(toks("rec_name").alias("a"), toks("s1_name").alias("b"))
    d = d.with_columns(pl.col("a").list.set_difference("b").alias("da"), pl.col("b").list.set_difference("a").alias("db"))
    d = d.with_columns(pl.col("da").list.len().alias("na"), pl.col("db").list.len().alias("nb"),
                       pl.col("da").list.first().alias("wa"), pl.col("db").list.first().alias("wb"))
    return d


def word_classes(pool):
    'per country: class of each word from the side balance over one-word swaps'
    sw = pool.filter((pl.col("na") == 1) & (pl.col("nb") == 1))
    rec = sw.group_by("co", pl.col("wa").alias("w")).len("n_rec")
    s1s = sw.group_by("co", pl.col("wb").alias("w")).len("n_s1")
    b = rec.join(s1s, on=["co", "w"], how="full", coalesce=True).fill_null(0).with_columns(
        (pl.col("n_rec") + pl.col("n_s1")).alias("n"), (pl.col("n_rec") / (pl.col("n_rec") + pl.col("n_s1"))).alias("rs"))
    cls = (pl.when(pl.col("n") < 50).then(pl.lit("rare"))
             .when(pl.col("w").is_in(list(STOP))).then(pl.lit("stop"))
             .when(pl.col("rs") > 0.7).then(pl.lit("filler"))
             .when(pl.col("rs") < 0.3).then(pl.lit("s1side"))
             .otherwise(pl.lit("cat")))
    return b.with_columns(cls.alias("wcls")).select("co", "w", "wcls")


def s1_vocab(ref):
    return ref.select("co", toks("s1_name").alias("t")).explode("t").drop_nulls().unique().rename({"t": "w"}).with_columns(pl.lit(1).alias("in_s1"))


def keys(d, wc, vocab):
    wa = wc.rename({"w": "wa", "wcls": "ca"})
    wb = wc.rename({"w": "wb", "wcls": "cb"})
    d = d.join(wa, on=["co", "wa"], how="left").join(wb, on=["co", "wb"], how="left").with_columns(
        pl.col("ca").fill_null("rare"), pl.col("cb").fill_null("rare"))

    ex = d.select("qid", "tid", "co", "da").explode("da").drop_nulls().rename({"da": "w"}).join(vocab, on=["co", "w"], how="left")
    unseen = ex.group_by("qid", "tid").agg(pl.col("in_s1").is_null().mean().alias("unseen"))
    d = d.join(unseen, on=["qid", "tid"], how="left").with_columns(pl.col("unseen").fill_null(0.0))
    raw = pl.col("rec_name").fill_null("")
    shape = (pl.when(norm("rec_name").str.contains(DBA)).then(pl.lit("dba"))
               .when(norm("rec_name").str.contains(r"\.com|www\.|\.fr\b|\.net|\.org") | raw.str.contains(r"(?i)[a-z]com$")).then(pl.lit("domain"))
               .when(raw.str.strip_chars().str.contains(r"^[A-Z]{2,4}$")).then(pl.lit("acronym"))
               .otherwise(pl.lit("")))
    name = (pl.when(shape != "").then(shape)
              .when((pl.col("na") == 0) & (pl.col("nb") == 0)).then(pl.lit("same core"))
              .when((pl.col("na") == 1) & (pl.col("nb") == 1)).then(pl.lit("swap ") + pl.col("ca") + pl.lit("<-") + pl.col("cb"))
              .when((pl.col("na") == 1) & (pl.col("nb") == 0)).then(pl.lit("added ") + pl.col("ca"))
              .when((pl.col("na") == 0) & (pl.col("nb") == 1)).then(pl.lit("dropped ") + pl.col("cb"))
              .when(pl.col("unseen") >= 0.99).then(pl.lit("made-up name"))
              .otherwise(pl.lit("other")))
    d = d.with_columns(name.alias("name_key"), legal("rec_name").alias("lr"), legal("s1_name").alias("ls"),
                       nums("s1_addr").list.first().alias("hs"), nums("rec_addr").alias("rn"),
                       street("rec_addr").alias("sr_"), street("s1_addr").alias("ss_"))
    d = d.with_columns(pl.col("sr_").list.set_intersection("ss_").list.len().alias("sov"), pl.col("sr_").list.len().alias("srl"))
    typo = pl.struct("hs", "rn").map_elements(
        lambda s: bool(s["hs"]) and bool(s["rn"]) and any(s["hs"] in r or r in s["hs"] or Levenshtein.distance(s["hs"], r) <= 1
                                                          for r in s["rn"]), return_dtype=pl.Boolean)
    same_street = pl.col("sov") >= (pl.col("srl") / 2).ceil().clip(1, None)
    addr = (pl.when(pl.col("rec_addr").fill_null("").str.strip_chars() == "").then(pl.lit("no address"))
              .when(pl.col("hs").is_null() | (pl.col("rn").list.len() == 0)).then(pl.lit("no number"))
              .when(pl.col("rn").list.contains(pl.col("hs")) & same_street).then(pl.lit("same no+street"))
              .when(pl.col("rn").list.contains(pl.col("hs"))).then(pl.lit("same no, other street"))
              .when(typo).then(pl.lit("typo no"))
              .otherwise(pl.lit("unrelated no")))
    leg = (pl.when((pl.col("lr") == "") & (pl.col("ls") == "")).then(pl.lit("none")).when((pl.col("lr") == "") | (pl.col("ls") == ""))
             .then(pl.lit("one")).when(pl.col("lr") == pl.col("ls")).then(pl.lit("eq")).otherwise(pl.lit("CONFLICT")))
    return d.with_columns(addr.alias("addr_key"), leg.alias("legal_key")).with_columns(
        (pl.col("name_key") + " | " + pl.col("addr_key") + " | legal " + pl.col("legal_key")).alias("key"))


def main(data, stack, accepted_test, out):
    os.makedirs(out, exist_ok=True)
    res = {}
    for split in ("train", "test"):
        cols = ["rid", "nm", "ad", "co"] + (["fold"] if split == "train" else [])
        ref = pl.read_parquet(os.path.join(data, split, "ref.parquet"), columns=cols).rename(
            {"rid": "qid", "nm": "s1_name", "ad": "s1_addr"}).with_columns(pl.col("qid").cast(pl.UInt32), pl.col("co").str.to_lowercase())
        tg = pl.concat([pl.read_parquet(os.path.join(data, split, f"s{s}.parquet"), columns=["rid", "nm", "ad"]) for s in (2, 3)]).rename(
            {"rid": "tid", "nm": "rec_name", "ad": "rec_addr"}).with_columns(pl.col("tid").cast(pl.UInt32))
        if split == "train":
            pr = pl.read_parquet(os.path.join(stack, "val_pred.parquet"), columns=["qid", "tid", "y", "p2"]).rename({"p2": "p"})
            acc = decode.apply("expected_f", pr.select("qid", "tid", "p"), 0.5).select("qid", "tid", pl.lit(True).alias("acc"))
        else:
            pr = pl.read_parquet(os.path.join(stack, "test_pred.parquet"), columns=["qid", "tid", "p2"]).rename({"p2": "p"})
            acc = pl.read_parquet(accepted_test).select("qid", "tid", pl.lit(True).alias("acc"))
        top = decode.top1(pr.select("qid", "tid", "p"))
        if "y" in pr.columns:
            top = top.join(pr.select("qid", "tid", "y"), on=["qid", "tid"], how="left")
        del pr
        pool = base_frame(top, ref, tg)
        wc = word_classes(pool)
        if split == "train":
            pool = pool.join(ref.filter(pl.col("fold") == 1).select("qid"), on="qid")
        k = keys(pool, wc, s1_vocab(ref)).join(acc, on=["qid", "tid"], how="left").with_columns(pl.col("acc").fill_null(False))
        agg = [pl.len().alias("n"), pl.col("acc").mean().alias("acc_rate"), pl.col("acc").sum().alias("accepted")]
        if split == "train":
            agg.insert(1, pl.col("y").mean().alias("true_rate"))
        res[split] = k.group_by("co", "key").agg(agg)
        k.filter(pl.col("co") == "france" if split == "test" else pl.col("co") != "").select(
            "qid", "tid", "co", "key", "acc", "rec_name", "rec_addr", "s1_name", "s1_addr").write_parquet(os.path.join(out, f"keys_{split}.parquet"))
        print(f"{split}: {k.height:,} best-candidate pairs, {res[split]['key'].n_unique():,} keys", flush=True)
    v = res["train"]
    us = v.filter(pl.col("co") == "us").select("key", pl.col("n").alias("n_us"), pl.col("true_rate").alias("true_us"), pl.col("acc_rate").alias("acc_us"))
    ind = v.filter(pl.col("co") == "india").select("key", pl.col("n").alias("n_in"), pl.col("true_rate").alias("true_in"), pl.col("acc_rate").alias("acc_in"))
    fr = res["test"].filter(pl.col("co") == "france").select("key", pl.col("n").alias("n_fr"), pl.col("acc_rate").alias("acc_fr"), pl.col("accepted").alias("acc_n_fr"))
    t = (fr.join(us, on="key", how="left").join(ind, on="key", how="left")
           .with_columns(pl.max_horizontal(pl.col("true_us").fill_null(-1), pl.col("true_in").fill_null(-1)).alias("true_max"))
           .with_columns((pl.col("acc_fr") - pl.col("true_max")).alias("excess"))
           .with_columns((pl.col("excess").clip(0, None) * pl.col("n_fr")).alias("excess_pairs"))
           .sort("excess_pairs", descending=True))
    t.write_parquet(os.path.join(out, "scan.parquet"))
    with pl.Config(tbl_rows=60, tbl_width_chars=260, fmt_str_lengths=70, float_precision=3):
        print(t.filter((pl.col("n_fr") >= 300) & (pl.col("true_max") >= 0) & ((pl.col("n_us").fill_null(0) + pl.col("n_in").fill_null(0)) >= 100)).head(40))
        print("France keys with no US/India counterpart (top by accepted):")
        print(t.filter(pl.col("true_max") < 0).sort("acc_n_fr", descending=True).head(15))


if __name__ == "__main__":
    main(*sys.argv[1:5])
