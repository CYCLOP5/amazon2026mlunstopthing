'french category-word-swap removal: 0.984 file -> 0.986 file (targeted), optionally -> t2'
import argparse
import json
import os
import shutil
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from er.io import write_id_lists  # noqa: E402
from er.stack import decode  # noqa: E402

LEGAL = ["sarl", "sas", "sa", "sasu", "eurl", "sci", "ei", "snc", "scop", "llc", "inc", "ltd", "pvt", "private",
         "limited", "co", "corp", "company", "lp", "llp", "pllc", "pc", "plc", "the", "l", "c", "s", "a", "r"]
STOP = ["de", "du", "des", "la", "le", "les", "et", "d", "l", "france"]


def toks(c):
    return (pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()
              .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" ")
              .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(LEGAL))).list.unique())


def hn(c):
    return pl.col(c).fill_null("").str.extract(r"(\d+)", 1)


def patterns(d):
    d = d.with_columns(toks("rec_name").alias("a"), toks("s1_name").alias("b"))
    d = d.with_columns(pl.col("a").list.set_difference("b").list.len().alias("na"),
                       pl.col("b").list.set_difference("a").list.len().alias("nb"))
    name = (pl.when((pl.col("na") == 0) & (pl.col("nb") == 0)).then(pl.lit("same core"))
              .when((pl.col("na") == 1) & (pl.col("nb") == 1)).then(pl.lit("1 word swapped"))
              .when((pl.col("na") == 1) & (pl.col("nb") == 0)).then(pl.lit("1 word added"))
              .when((pl.col("na") == 0) & (pl.col("nb") == 1)).then(pl.lit("1 word dropped"))
              .otherwise(pl.lit("other")))
    addr = (pl.when(pl.col("rec_addr").fill_null("").str.strip_chars() == "").then(pl.lit("no address"))
              .when(hn("rec_addr") == hn("s1_addr")).then(pl.lit("same house no"))
              .otherwise(pl.lit("diff house no")))
    return d.with_columns(name.alias("name_pat"), addr.alias("addr_pat"),
                          pl.col("a").list.set_difference("b").list.first().alias("wa"),
                          pl.col("b").list.set_difference("a").list.first().alias("wb"))


def swap_direction(pool):
    s = pool.filter(pl.col("name_pat") == "1 word swapped")
    cnt = s.group_by("co", "wa", "wb").len("n")
    rev = cnt.rename({"wa": "wb", "wb": "wa", "n": "n_rev"})
    s = (s.join(cnt, on=["co", "wa", "wb"]).join(rev, on=["co", "wa", "wb"], how="left")
          .with_columns(pl.col("n_rev").fill_null(0)).with_columns((pl.col("n") / (pl.col("n") + pl.col("n_rev"))).alias("one_way")))
    return s.with_columns(pl.when(pl.col("n") + pl.col("n_rev") < 20).then(pl.lit("rare pair"))
                            .when(pl.col("one_way") >= 0.9).then(pl.lit("one-way >=0.9"))
                            .when(pl.col("one_way") >= 0.7).then(pl.lit("0.7-0.9"))
                            .otherwise(pl.lit("two-way <0.7")).alias("direction"))


def targets(dataset):
    parts, off = [], 0
    for s in ("2", "3"):
        x = pl.read_csv(os.path.join(dataset, "test", f"test_source{s}.tsv"), separator="\t", quote_char=None,
                        columns=["entity_id", "business_name", "business_address"]).with_row_index("tid")
        parts.append(x.with_columns((pl.col("tid") + off).cast(pl.UInt32)))
        off += x.height
    return pl.concat(parts).rename({"entity_id": "o_id", "business_name": "rec_name", "business_address": "rec_addr"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accepted", required=True, help="accepted.parquet (qid, tid) of the 0.984 file (final_blend.py)")
    ap.add_argument("--pool-pred", required=True, help="test_pred.parquet (qid, tid, p2) for the direction statistic")
    ap.add_argument("--dataset", required=True, help="raw challenge data dir (test/test_source{1,2,3}.tsv)")
    ap.add_argument("--test-ref", required=True, help="test Source-1 parquet: rid, eid, nm, ad, co")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=int, default=1, choices=(1, 2, 3),
                    help="1 = TARGETED (0.986), 2 = T2, 3 = T2G (T2 minus records adding the decoy word 'Groupe')")
    ap.add_argument("--groupe", action="store_true", help="also apply the level-3 decoy-word removal at level 1 or 2")
    ap.add_argument("--decoy-words", default="groupe", help="comma list for the decoy-word removal (T2G/T1G: groupe; "
                    "T1GD: groupe,developpement)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ref = pl.read_parquet(a.test_ref, columns=["rid", "eid", "nm", "ad", "co"]).rename(
        {"rid": "qid", "nm": "s1_name", "ad": "s1_addr"}).with_columns(pl.col("qid").cast(pl.UInt32))
    fr = ref.filter(pl.col("co").str.to_lowercase() == "france").select("qid", "co", "s1_name", "s1_addr")
    fq = fr.select("qid")
    tg = targets(a.dataset)
    tgn = tg.select("tid", "rec_name", "rec_addr")


    pp = pl.read_parquet(a.pool_pred, columns=["qid", "tid", "p2"]).rename({"p2": "p"}).join(fq, on="qid")
    pool = patterns(decode.top1(pp).join(fr, on="qid").join(tgn, on="tid"))
    del pp
    sw = swap_direction(pool)
    st = sw.select("co", "wa", "wb", "direction").unique(["co", "wa", "wb"])

    acc = pl.read_parquet(a.accepted).select("qid", "tid")
    fr_acc = acc.join(fq, on="qid")
    d = patterns(fr_acc.join(fr, on="qid").join(tgn, on="tid")).join(st, on=["co", "wa", "wb"], how="left")
    drop1 = d.filter((pl.col("name_pat") == "1 word swapped") & (pl.col("addr_pat") == "same house no")
                     & (pl.col("direction") == "two-way <0.7")).select("qid", "tid")
    keep = acc.join(drop1, on=["qid", "tid"], how="anti")
    info = {"level": a.level, "accepted_in": acc.height, "france_in": fr_acc.height, "removed_two_way": drop1.height}

    if a.level >= 2:
        rec = sw.group_by(pl.col("wa").alias("w")).len("n_rec")
        s1s = sw.group_by(pl.col("wb").alias("w")).len("n_s1")
        bal = rec.join(s1s, on="w", how="full", coalesce=True).fill_null(0).with_columns(
            (pl.col("n_rec") + pl.col("n_s1")).alias("n"), (pl.col("n_rec") / (pl.col("n_rec") + pl.col("n_s1"))).alias("rs"))
        C = sorted(bal.filter((pl.col("n") >= 50) & (pl.col("rs") >= 0.3) & (pl.col("rs") <= 0.7) & ~pl.col("w").is_in(STOP)
                              & (pl.col("w").str.len_chars() >= 3) & ~pl.col("w").str.contains(r"\d"))["w"].to_list())
        k = patterns(keep.join(fq, on="qid").join(fr, on="qid").join(tgn, on="tid")).with_columns(
            pl.col("a").list.set_difference("b").list.set_intersection(pl.lit(C)).list.len().alias("ca"),
            pl.col("b").list.set_difference("a").list.set_intersection(pl.lit(C)).list.len().alias("cb"))
        drop2 = k.filter((pl.col("ca") > 0) & (pl.col("cb") > 0)).select("qid", "tid")
        keep = keep.join(drop2, on=["qid", "tid"], how="anti")
        info.update({"category_words": len(C), "removed_category_swaps": drop2.height})

    if a.level == 3 or a.groupe:
        g = keep.join(fq, on="qid").join(fr, on="qid").join(tgn, on="tid").with_columns(
            toks("rec_name").alias("rt"), toks("s1_name").alias("st"))
        for w in a.decoy_words.split(","):
            drop3 = g.filter(pl.col("rt").list.contains(w) & ~pl.col("st").list.contains(w)).select("qid", "tid")
            keep = keep.join(drop3, on=["qid", "tid"], how="anti")
            g = g.join(drop3, on=["qid", "tid"], how="anti")
            info.update({f"removed_{w}": drop3.height})

    keep.write_parquet(os.path.join(a.out, "accepted.parquet"))

    s1_ids = ref.sort("qid")["eid"]
    m = keep.join(ref.select("qid", pl.col("eid").alias("s1_id")), on="qid").join(tg.select("tid", "o_id"), on="tid").select("s1_id", "o_id")
    assert m.height == keep.height
    write_id_lists(m, s1_ids, "matched_entity_ids", os.path.join(a.out, "matching_results.tsv"))
    c = (pl.read_csv(a.candidates, separator="\t", schema_overrides={"candidate_entity_ids": pl.Utf8})
           .with_columns(pl.col("candidate_entity_ids").fill_null("").str.split(",")).explode("candidate_entity_ids")
           .rename({"source1_entity_id": "s1_id", "candidate_entity_ids": "o_id"}))
    missing = m.join(c, on=["s1_id", "o_id"], how="anti")
    if missing.height:
        write_id_lists(pl.concat([c.filter(pl.col("o_id") != ""), missing]), s1_ids, "candidate_entity_ids",
                       os.path.join(a.out, "candidate_pairs.tsv"))
    else:
        shutil.copyfile(a.candidates, os.path.join(a.out, "candidate_pairs.tsv"))
    info.update({"matches_out": keep.height, "candidates_added": missing.height})
    json.dump(info, open(os.path.join(a.out, "france_swaps.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
