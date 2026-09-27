'where is macro f0.5 lost on validation? (labelled fold, run-6 decisions)'
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from er.stack import decode  # noqa: E402


def f05(tp, npred, ntrue):
    tp, npred, ntrue = (np.asarray(x, dtype=float) for x in (tp, npred, ntrue))
    p = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
    r = np.where(ntrue > 0, tp / np.maximum(ntrue, 1), 0.0)
    f = np.where(tp > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)
    return np.where((npred == 0) & (ntrue == 0), 1.0, f)


def main(data, stack, out, fold=1):
    os.makedirs(out, exist_ok=True)
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
    ref = pl.read_parquet(os.path.join(data, "train", "ref.parquet"), columns=["rid", "nm", "ad", "co", "fold", "deg"]).rename({"rid": "qid"})
    ref = ref.with_columns(pl.col("qid").cast(pl.UInt32))
    key = pl.col("nm").fill_null("").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", "")
    ref = ref.with_columns(key.alias("nk")).with_columns(pl.len().over("co", "nk").alias("n_same_name"))
    a = ref.filter(pl.col("fold") == fold)
    vp = pl.read_parquet(os.path.join(stack, "val_pred.parquet"), columns=["qid", "tid", "y", "p2"]).join(a.select("qid"), on="qid")
    acc = decode.apply("expected_f", vp.select("qid", "tid", pl.col("p2").alias("p")), 0.5).select("qid", "tid").with_columns(pl.lit(True).alias("acc"))
    d = vp.join(acc, on=["qid", "tid"], how="left").with_columns(pl.col("acc").fill_null(False))
    per = d.group_by("qid").agg((pl.col("acc") & (pl.col("y") == 1)).sum().alias("tp"), pl.col("acc").sum().alias("npred"),
                                (pl.col("y") == 1).sum().alias("cand_true"), (pl.col("acc") & (pl.col("y") == 0)).sum().alias("fp"))
    e = a.join(per, on="qid", how="left").with_columns(pl.col(c).fill_null(0) for c in ("tp", "npred", "cand_true", "fp"))
    e = e.with_columns((pl.col("deg") - pl.col("cand_true")).clip(0, None).alias("missing"),
                       (pl.col("cand_true") - pl.col("tp")).alias("rejected"))
    tp, npred, deg = e["tp"].to_numpy(), e["npred"].to_numpy(), e["deg"].to_numpy()
    miss, rej, fp = e["missing"].to_numpy(), e["rejected"].to_numpy(), e["fp"].to_numpy()
    base = f05(tp, npred, deg)
    fixes = {"missing": f05(tp + miss, npred + miss, deg), "rejected": f05(tp + rej, npred + rej, deg),
             "false": f05(tp, npred - fp, deg), "all": np.ones_like(base)}
    n = len(base)
    print(f"fold {fold}: {n:,} S1, macro F0.5 = {base.mean():.5f}; candidate oracle = {f05(e['cand_true'].to_numpy(), e['cand_true'].to_numpy(), deg).mean():.5f}", flush=True)
    for k, v in fixes.items():
        print(f"  repair {k:9s}: +{(v - base).mean():.5f}", flush=True)
    typ = (pl.when(pl.col("deg") == 0).then(pl.lit("singleton (no true match)"))
             .when(pl.col("ad").fill_null("").str.strip_chars() == "").then(pl.lit("S1 blank address"))
             .when(pl.col("n_same_name") > 1).then(pl.lit("S1 name twins"))
             .otherwise(pl.lit("normal")))
    e = e.with_columns(typ.alias("type"), pl.Series("loss", 1 - base),
                       *[pl.Series(f"gain_{k}", fixes[k] - base) for k in ("missing", "rejected", "false")])
    t = e.group_by("co", "type").agg(pl.len().alias("s1"), (pl.col("loss").sum() / n).alias("loss_share"),
                                     *[(pl.col(f"gain_{k}").sum() / n).alias(f"fix_{k}") for k in ("missing", "rejected", "false")],
                                     pl.col("missing").sum().alias("missing_pairs"), pl.col("rejected").sum().alias("rejected_pairs"),
                                     pl.col("fp").sum().alias("false_pairs")).sort("loss_share", descending=True)
    print(t, flush=True)
    t.write_parquet(os.path.join(out, "bottleneck.parquet"))
    e.write_parquet(os.path.join(out, "per_s1.parquet"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
