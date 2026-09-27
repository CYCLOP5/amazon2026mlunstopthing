'US/India: which accepted pairs do the *other* models reject, and how often are those pairs wrong?'
import argparse
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from er.stack import decode  # noqa: E402

K = ["qid", "tid"]
VOTERS = ["newest", "neural", "gate", "friend", "graph", "hybrid"]


def sets(df):
    below = {v: pl.col(v).is_not_null() & (pl.col(v) < 0.5) for v in VOTERS}
    n_dis = pl.sum_horizontal([below[v].cast(pl.Int8) for v in VOTERS])
    n_av = pl.sum_horizontal([pl.col(v).is_not_null().cast(pl.Int8) for v in VOTERS])
    out = {f"{v}<0.5": below[v] for v in VOTERS}
    out.update({"any2": n_dis >= 2, "any3": n_dis >= 3, "half": (n_dis * 2 >= n_av) & (n_av > 0),
                "newest&friend": below["newest"] & below["friend"],
                "min<0.2": pl.min_horizontal([pl.col(v) for v in VOTERS]) < 0.2,
                "mean<0.6": pl.mean_horizontal([pl.col(v) for v in VOTERS]) < 0.6})
    return out


def keep_one(base, rem):
    'do not empty any source-1 entity: spare removals where nothing else of that entity stays'
    left = base.join(rem.select(K), on=K, how="anti").select("qid").unique()
    return rem.join(left, on="qid", how="semi")


def main():
    ap = argparse.ArgumentParser()
    for k in ("train", "test", "data", "old_val", "new_val", "old_test", "new_test", "out"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref = pl.read_parquet(os.path.join(a.data, "train", "ref.parquet"), columns=["rid", "fold", "deg", "co"])
    f1 = ref.filter((pl.col("fold") == 1) & (pl.col("co").str.to_lowercase() != "france")).select(pl.col("rid").alias("qid"), "deg")
    old = decode.top1(pl.scan_parquet(a.old_val).filter(pl.col("p") >= 0.7842838168144226).collect()).join(f1, on="qid", how="semi")
    nw = pl.scan_parquet(a.new_val).filter(pl.col("p") >= 0.8).collect().join(f1, on="qid", how="semi")
    new = decode.expected_f(nw.select("qid", "tid", "p", "y"), 0.8).select(K)
    base = old.join(new, on=K, how="semi").select(K + ["y"])
    sc = (pl.scan_parquet(a.train).select(K + VOTERS).join(base.lazy().select(K), on=K, how="semi").collect())
    base = base.join(sc, on=K, how="left")
    f0 = decode.score(base.select("qid", "tid", "y"), f1)["macro_f05"]
    print(f"val base (DROP) pairs {base.height:,} F {f0:.5f}; voter coverage", {v: round(base[v].is_not_null().mean(), 3) for v in VOTERS}, flush=True)
    report = {}
    for name, expr in sets(base).items():
        for safe in (False, True):
            rem = base.filter(expr)
            if safe:
                rem = keep_one(base, rem)
            kept = base.join(rem.select(K), on=K, how="anti")
            s = decode.score(kept.select("qid", "tid", "y"), f1)["macro_f05"]
            key = name + (" keep1" if safe else "")
            report[key] = {"pairs": rem.height, "false_rate": (1 - rem["y"].mean()) if rem.height else None, "val_delta": s - f0}
            print(f"{key:22s} pairs {rem.height:7,} false {report[key]['false_rate'] if rem.height else 0:.3f} val {s - f0:+.5f}", flush=True)
    with open(os.path.join(a.out, "val_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    rt = pl.read_parquet(os.path.join(a.data, "test", "ref.parquet"), columns=["rid", "eid", "co"])
    usin = rt.filter(pl.col("co").str.to_lowercase() != "france").select(pl.col("rid").alias("qid"))
    tb = (pl.read_parquet(a.old_test).select(K).join(pl.read_parquet(a.new_test).select(K), on=K, how="semi")
            .join(usin, on="qid", how="semi"))
    ts = pl.scan_parquet(a.test).select(K + VOTERS).join(tb.lazy(), on=K, how="semi").collect()
    tb = tb.join(ts, on=K, how="left")
    tg = pl.concat([pl.read_parquet(os.path.join(a.data, "test", f"s{i}.parquet"), columns=["rid", "eid"]) for i in (2, 3)])
    print(f"test base (DROP US/India) pairs {tb.height:,}", flush=True)
    for name, expr in sets(tb).items():
        for safe in (False, True):
            rem = tb.filter(expr)
            if safe:
                rem = keep_one(tb, rem)
            rem = (rem.select(K).join(rt.select(pl.col("rid").alias("qid"), pl.col("eid").alias("s1")), on="qid")
                      .join(tg.select(pl.col("rid").alias("tid"), pl.col("eid").alias("rec")), on="tid"))
            fn = (name + ("_keep1" if safe else "")).replace("<", "lt").replace("&", "and").replace(".", "")
            rem.write_parquet(os.path.join(a.out, f"test_remove_{fn}.parquet"))
            print(f"test {fn:24s} {rem.height:7,}", flush=True)


if __name__ == "__main__":
    main()
