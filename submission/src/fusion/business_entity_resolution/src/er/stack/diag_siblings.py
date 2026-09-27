'diagnostic: would record-to-record (sibling) evidence recover the missed links?'
import json
import os
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from er.stack import decode  # noqa: E402
from er.stack.features import normalize  # noqa: E402


def cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def main(data, stack_dir, out_dir, fold=1):
    os.makedirs(out_dir, exist_ok=True)
    rep = json.load(open(os.path.join(stack_dir, "report.json")))
    sel = rep["methods"][rep["selected"]]
    col, rule, thr = sel["score"], sel["tune"]["rule"], sel["tune"]["threshold"]
    vp = pl.read_parquet(os.path.join(stack_dir, "val_pred.parquet"))
    refs = pl.read_parquet(os.path.join(data, "train", "ref.parquet"), columns=["rid", "nm", "ad", "co", "fold", "deg"])
    tg = pl.concat([pl.read_parquet(os.path.join(data, "train", f"s{sr}.parquet"), columns=["rid", "nm", "ad", "co", "own"])
                    for sr in (2, 3)]).with_columns(pl.col("rid").cast(pl.UInt32))
    anchors = refs.filter(pl.col("fold") == fold).select(pl.col("rid").cast(pl.UInt32).alias("qid"))
    d = vp.select("qid", "tid", pl.col(col).alias("p"), "y")
    acc = decode.apply(rule, d, thr).select("qid", "tid")
    best = decode.top1(d).select("tid", pl.col("qid").alias("best_q"), pl.col("p").alias("best_p"))
    conf = best.filter(pl.col("best_p") >= 0.5).select(pl.col("best_q").alias("qid"), pl.col("tid").alias("sid"))

    links = (tg.filter(pl.col("own") >= 0).select(pl.col("rid").alias("tid"), pl.col("own").cast(pl.UInt32).alias("qid"))
               .join(anchors, on="qid"))
    missed = (links.join(acc, on=["qid", "tid"], how="anti").join(d.select("qid", "tid", "p"), on=["qid", "tid"], how="left")
                   .join(best, on="tid", how="left")
                   .with_columns(pl.when(pl.col("p").is_null()).then(pl.lit("not a candidate"))
                                   .when(pl.col("best_q") == pl.col("qid")).then(pl.lit("best candidate, rejected"))
                                   .otherwise(pl.lit("another S1 ranked first")).alias("cause")))
    own = tg.select(pl.col("rid").alias("tid"), "own")
    wrong = (acc.join(anchors, on="qid").join(own, on="tid", how="left")
                .filter(pl.col("own") != pl.col("qid").cast(pl.Int32))
                .with_columns(pl.when(pl.col("own") < 0).then(pl.lit("wrong: decoy")).otherwise(pl.lit("wrong: other S1")).alias("cause")))

    decoy_near = (d.join(anchors, on="qid").join(own, on="tid", how="left").filter(pl.col("own") < 0)
                   .join(acc, on=["qid", "tid"], how="anti").unique("tid").select("qid", "tid")
                   .with_columns(pl.lit("control: rejected decoy").alias("cause")))
    decoy_near = decoy_near.sample(min(200_000, decoy_near.height), seed=1)
    probe = pl.concat([missed.select("qid", "tid", "cause"), wrong.select("qid", "tid", "cause"), decoy_near])
    print("probe rows", probe.group_by("cause").len().to_dicts(), flush=True)


    s = probe.join(conf, on="qid", how="left").filter(pl.col("sid").is_null() | (pl.col("sid") != pl.col("tid")))
    need = pl.concat([s["tid"], s["sid"].drop_nulls()]).unique()
    attr = normalize(tg.filter(pl.col("rid").is_in(need.implode())).select("rid", "nm", "ad", "co")).select(
        "rid", "name_core", "addr_can", pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("hn"))
    a_t = attr.rename({"rid": "tid", "name_core": "tn", "addr_can": "ta", "hn": "th"})
    a_s = attr.rename({"rid": "sid", "name_core": "sn", "addr_can": "sa", "hn": "sh"})
    s = s.join(a_t, on="tid", how="left").join(a_s, on="sid", how="left")
    has = s.filter(pl.col("sid").is_not_null())
    has = has.with_columns(
        pl.Series("n", cp(has["tn"].fill_null("").to_list(), has["sn"].fill_null("").to_list(), fuzz.token_set_ratio)),
        pl.Series("a", cp(has["ta"].fill_null("").to_list(), has["sa"].fill_null("").to_list(), fuzz.token_set_ratio)),
    ).with_columns(((pl.col("th") == pl.col("sh")) & (pl.col("th") != "")).alias("hn_eq"))
    agg = has.group_by("qid", "tid").agg(pl.len().alias("sibs"), pl.col("n").max().alias("best_n"), pl.col("a").max().alias("best_a"),
                                         ((pl.col("n") + pl.col("a")) / 2).max().alias("best_na"), pl.col("hn_eq").any().alias("any_hn"))
    r = probe.join(agg, on=["qid", "tid"], how="left").with_columns(pl.col("sibs").fill_null(0))
    levels = [("name>=95 & addr>=90", (pl.col("best_n") >= 95) & (pl.col("best_a") >= 90)),
              ("name>=90 & addr>=80", (pl.col("best_n") >= 90) & (pl.col("best_a") >= 80)),
              ("name>=95 (any address)", pl.col("best_n") >= 95),
              ("addr>=95 & same house no.", (pl.col("best_a") >= 95) & pl.col("any_hn")),
              ("name+addr mean>=85", pl.col("best_na") >= 85)]
    table = []
    for cause in sorted(r["cause"].unique().to_list()):
        g = r.filter(pl.col("cause") == cause)
        row = {"cause": cause, "rows": g.height, "with_siblings": int((g["sibs"] > 0).sum())}
        for name, e in levels:
            row[name] = int(g.filter(e.fill_null(False)).height)
        table.append(row)
    lines = ["# Sibling diagnostic (audit fold)", "",
             "Rows: missed true links by cause, wrong accepted pairs, and a sample of correctly rejected decoys (control).",
             "Columns: how many of them are that similar to at least one record the TRUE / accepted S1 already owns.", "",
             "| cause | rows | has siblings | " + " | ".join(n for n, _ in levels) + " |",
             "|---|---|---|" + "---|" * len(levels)]
    for t in table:
        lines.append(f"| {t['cause']} | {t['rows']:,} | {t['with_siblings']:,} | " +
                     " | ".join(f"{t[n]:,} ({t[n] / max(t['rows'], 1):.0%})" for n, _ in levels) + " |")
    lines += ["", "Read: a level is useful when it covers many missed links but few decoys / wrong pairs."]
    open(os.path.join(out_dir, "diag_siblings.md"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump(table, open(os.path.join(out_dir, "diag_siblings.json"), "w"), indent=1)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    fold = int(a[a.index("--fold") + 1]) if "--fold" in a else 1
    main(a[0], a[1], a[2], fold)
