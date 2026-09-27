'where the remaining points are lost (audit fold) and how test differs from validation'
import json
import os

import polars as pl

from er.stack.inputs import load_refs

DRIFT = ["t_n", "s1_nclaim", "s1_ntop1", "s1_ntop1_p50", "s1_sum_p", "prob", "neural_prob", "gate_prob",
         "sib_n", "sib_n_src", "s1_name_freq", "addr_empty2", "nonascii2"]


def _targets_own(data, split):
    return pl.concat([pl.read_parquet(os.path.join(data, split, f"s{sr}.parquet"),
                                      columns=["rid", "eid", "nm", "ad", "co", "own"])
                        .with_columns(pl.col("rid").cast(pl.UInt32), pl.lit(sr, pl.UInt8).alias("sr"))
                      for sr in (2, 3)])


def run(tr: pl.DataFrame, te: pl.DataFrame, acc_tr: pl.DataFrame, acc_te: pl.DataFrame, col: str,
        sc: dict, out_dir: str, tgt_attr_tr: pl.DataFrame, n_examples: int = 25):
    fold = sc["audit_fold"]
    refs = load_refs(sc["data"], "train")
    tg = _targets_own(sc["data"], "train")
    anchors = refs.filter(pl.col("fold") == fold).select(pl.col("rid").alias("qid"), "deg", "co", "nm", "ad")

    links = (tg.filter(pl.col("own") >= 0).select(pl.col("rid").alias("tid"), pl.col("own").cast(pl.UInt32).alias("qid"), "sr")
               .join(anchors.select("qid"), on="qid", how="inner"))
    cand = tr.select("qid", "tid", pl.col(col).alias("p"))
    best = cand.sort(["tid", "p", "qid"], descending=[False, True, False]).unique("tid", keep="first").select(
        "tid", pl.col("qid").alias("best_q"), pl.col("p").alias("best_p"))
    acc = acc_tr.select("qid", "tid").with_columns(pl.lit(1, pl.Int8).alias("accepted"))
    L = (links.join(cand, on=["qid", "tid"], how="left").join(best, on="tid", how="left")
              .join(acc, on=["qid", "tid"], how="left")
              .with_columns(pl.when(pl.col("accepted") == 1).then(pl.lit("found"))
                              .when(pl.col("p").is_null()).then(pl.lit("missed: true S1 not a candidate"))
                              .when(pl.col("best_q") == pl.col("qid")).then(pl.lit("missed: best candidate, rejected"))
                              .otherwise(pl.lit("missed: another S1 ranked first")).alias("cause")))

    feat_cols = [c for c in ("addr_empty2", "nonascii2", "s1_name_freq") if c in tr.columns]
    tfeat = tr.select("tid", *feat_cols).unique("tid")
    L = (L.join(tfeat, on="tid", how="left")
          .join(anchors.select("qid", "deg", "co"), on="qid", how="left")
          .with_columns(pl.when(pl.col("sr") == 3).then(pl.lit("S3")).otherwise(pl.lit("S2")).alias("source"),
                        pl.when(pl.col("deg") <= 2).then(pl.lit("1-2")).when(pl.col("deg") <= 4).then(pl.lit("3-4"))
                          .otherwise(pl.lit("5+")).alias("entity_size")))
    rep = {"links": L.height, "by_cause": _share(L, ["cause"])}
    for dim in ("co", "source", "entity_size", "addr_empty2", "nonascii2"):
        if dim in L.columns:
            rep[f"missed_share_by_{dim}"] = (L.group_by(dim).agg(pl.len().alias("links"),
                                                                 (pl.col("cause") != "found").mean().alias("missed_share"))
                                              .sort(dim).to_dicts())
    if "s1_name_freq" in L.columns:
        rep["missed_share_by_chain"] = (L.with_columns((pl.col("s1_name_freq").fill_null(1) > 1).alias("chain_name"))
                                         .group_by("chain_name").agg(pl.len().alias("links"),
                                                                     (pl.col("cause") != "found").mean().alias("missed_share"))
                                         .to_dicts())

    own = tg.select(pl.col("rid").alias("tid"), "own")
    W = (acc_tr.select("qid", "tid").join(anchors.select("qid"), on="qid", how="inner").join(own, on="tid", how="left")
               .filter(pl.col("own") != pl.col("qid").cast(pl.Int32))
               .with_columns(pl.when(pl.col("own") < 0).then(pl.lit("wrong: decoy record (no owner)"))
                               .otherwise(pl.lit("wrong: record of another S1")).alias("cause")))
    rep["wrong_by_cause"] = W.group_by("cause").len().to_dicts()

    nc = tgt_attr_tr.select(pl.col("rid").alias("tid"), "name_core")
    conf = (best.filter(pl.col("best_p") >= sc["confident_p"]).select(pl.col("best_q").alias("qid"), pl.col("tid").alias("sid"))
                .join(nc.rename({"tid": "sid", "name_core": "sname"}), on="sid", how="left"))
    nc_links = L.filter(pl.col("cause") == "missed: true S1 not a candidate").join(nc, on="tid", how="left")
    hit = nc_links.join(conf, on="qid", how="inner").filter(pl.col("name_core") == pl.col("sname")).select("qid", "tid").unique()
    rep["not_candidate_links"] = nc_links.height
    rep["not_candidate_covered_by_same_name_sibling"] = hit.height
    rep["not_candidate_name_missing"] = int(nc_links["name_core"].is_null().sum())


    tt = tg.select(pl.col("rid").alias("tid"), pl.col("nm").alias("rec_name"), pl.col("ad").alias("rec_addr"))
    rr = refs.select(pl.col("rid").alias("qid"), pl.col("nm").alias("s1_name"), pl.col("ad").alias("s1_addr"))
    lines = ["# Error analysis (audit fold)", "", f"true links: {L.height:,}", ""]
    for r in rep["by_cause"]:
        lines.append(f"- {r['cause']}: {r['n']:,} ({r['share']:.2%})")
    lines += ["", f"wrong matches: {W.height:,}"] + [f"- {r['cause']}: {r['len']:,}" for r in rep["wrong_by_cause"]]
    lines += ["", f"not-a-candidate links: {nc_links.height:,}; covered by a confident same-name sibling of the true S1: "
                  f"{hit.height:,}", ""]
    for cause in sorted(set(L["cause"].to_list()) - {"found"}):
        x = L.filter(pl.col("cause") == cause)
        x = x.sample(min(n_examples, x.height), seed=1).join(tt, on="tid", how="left").join(rr, on="qid", how="left")
        x = x.join(rr.rename({"qid": "best_q", "s1_name": "top_name", "s1_addr": "top_addr"}), on="best_q", how="left")
        lines += [f"## {cause}", ""]
        for r in x.iter_rows(named=True):
            p = "-" if r["p"] is None else f"{r['p']:.3f}"
            lines.append(f"- p={p} | REC `{r['rec_name']}` | `{r['rec_addr']}`  \n  TRUE `{r['s1_name']}` | `{r['s1_addr']}`")
            if r["best_q"] is not None and r["best_q"] != r["qid"]:
                lines.append(f"  TOP  `{r['top_name']}` | `{r['top_addr']}` (p={r['best_p']:.3f})")
        lines.append("")
    for cause in sorted(set(W["cause"].to_list())):
        x = W.filter(pl.col("cause") == cause)
        x = x.sample(min(n_examples, x.height), seed=1).join(tt, on="tid", how="left").join(rr, on="qid", how="left")
        lines += [f"## {cause}", ""]
        for r in x.iter_rows(named=True):
            lines.append(f"- REC `{r['rec_name']}` | `{r['rec_addr']}`  \n  ACCEPTED S1 `{r['s1_name']}` | `{r['s1_addr']}`")
        lines.append("")


    cols = [c for c in DRIFT if c in tr.columns and c in te.columns]
    co_tr = tr.join(refs.select(pl.col("rid").alias("qid"), "co"), on="qid", how="left")
    co_te = te.join(load_refs(sc["data"], "test").select(pl.col("rid").alias("qid"), "co"), on="qid", how="left")
    drift = []
    for name, d in (("validation", co_tr), ("test", co_te)):
        drift += d.group_by("co").agg(pl.len().alias("pairs"), *[pl.col(c).cast(pl.Float64).mean().alias(c) for c in cols]
                                      ).with_columns(pl.lit(name).alias("split")).to_dicts()
    rep["drift_feature_means"] = drift
    ref_te = load_refs(sc["data"], "test").select(pl.col("rid").alias("qid"), "co")
    rate = []
    for name, a, r in (("validation", acc_tr.join(anchors.select("qid"), on="qid"), anchors.select("qid", "co")),
                       ("test", acc_te, ref_te)):
        m = a.select("qid").join(r, on="qid").group_by("co").len().rename({"len": "matches"})
        rate += (r.group_by("co").len().rename({"len": "s1"}).join(m, on="co", how="left")
                  .with_columns((pl.col("matches") / pl.col("s1")).alias("matches_per_s1"), pl.lit(name).alias("split"))
                  .to_dicts())
    rep["acceptance"] = rate
    lines += ["## Drift: matches per Source-1", ""] + [f"- {r['split']} {r['co']}: {r['matches_per_s1']:.3f} (S1 {r['s1']:,})"
                                                       for r in rate]
    open(os.path.join(out_dir, "analysis.md"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump(rep, open(os.path.join(out_dir, "analysis.json"), "w"), indent=1, default=str)
    return rep


def _share(d, by):
    return (d.group_by(by).agg(pl.len().alias("n")).with_columns((pl.col("n") / pl.col("n").sum()).alias("share"))
              .sort("n", descending=True).to_dicts())
