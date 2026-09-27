"short blend search over the final team's france score pool"
import argparse as ap
import gc
import json
from pathlib import Path as path

import numpy as np
import polars as pl

import decode
from sprintcal import best_cut, export, metric

legal = ["sarl", "sas", "sa", "sasu", "eurl", "sci", "ei", "snc", "scop", "llc", "inc", "ltd", "pvt", "private", "limited", "co", "corp", "company", "lp", "llp", "pllc", "pc", "plc", "the", "l", "c", "s", "a", "r"]
mixes = [(1, 1, 1, 0), (2, 1, 1, 0), (1, 2, 1, 0), (1, 1, 1, 1), (2, 1, 1, 2), (1, 1, 0, 1)]


def tokens(c):
    return (pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()
            .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" ")
            .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(legal))).list.unique())


def patterns(d):
    d = d.with_columns(tokens("rec_name").alias("a"), tokens("s1_name").alias("b"))
    d = d.with_columns(pl.col("a").list.set_difference("b").alias("extra"), pl.col("b").list.set_difference("a").alias("missing"))
    return d.with_columns((pl.col("extra").list.len().eq(1) & pl.col("missing").list.len().eq(1)).alias("swap"),
                          pl.col("extra").list.first().alias("wa"), pl.col("missing").list.first().alias("wb"),
                          (pl.col("rec_addr").fill_null("").str.extract(r"(\d+)", 1) == pl.col("s1_addr").fill_null("").str.extract(r"(\d+)", 1)).fill_null(False).alias("same_house"))


def blend(frame, weights):
    numerator = np.zeros(len(frame), np.float64)
    denominator = np.zeros(len(frame), np.float64)
    for column, weight in zip(("newest", "friend", "graph", "collective"), weights):
        if not weight:
            continue
        v = frame[column].to_numpy()
        valid = np.isfinite(v)
        numerator[valid] += v[valid] * weight
        denominator[valid] += weight
    return np.divide(numerator, denominator, out=np.zeros(len(frame)), where=denominator > 0).astype(np.float32)


def load(prepared, collective, train=False):
    columns = ["qid", "tid", "co", "newest", "friend", "graph"] + (["y"] if train else [])
    f = pl.scan_parquet(prepared).select(columns)
    if not train:
        f = f.filter(pl.col("co") == "france")
    f = f.collect(engine="streaming")
    p = pl.scan_parquet(collective).select("qid", "tid", pl.col("p").alias("collective"))
    if not train:
        p = p.join(f.select("qid").unique().lazy(), on="qid", how="semi")
    p = p.collect(engine="streaming")
    if len(p) == len(f) and p.select("qid", "tid").equals(f.select("qid", "tid")):
        return f.with_columns(p["collective"])
    return f.join(p, on=["qid", "tid"], how="left", validate="1:1", maintain_order="left")


def run(data, train, test, scores, pool, baseline, out, threads=12, pairs_only=False):
    out.mkdir(parents=True, exist_ok=True)
    refs = pl.read_parquet(data / "train/ref.parquet", columns=["rid", "co", "deg", "fold"]).sort("rid")
    wanted = refs["fold"].to_numpy() == 0
    degrees = refs["deg"].to_numpy()
    frame = load(train / "prepared-train.parquet", scores / "validation_predictions.parquet", True)
    q, t, y = (frame[c].to_numpy() for c in ("qid", "tid", "y"))
    trials = []
    for weights in mixes:
        p = blend(frame, weights)
        top = decode.winners(q, t, p)
        cut = best_cut(p[top], q[top], y[top], degrees, wanted)
        res = metric(q[top], y[top], p[top] >= cut, degrees, wanted)
        trials.append({"weights": weights, "cut": cut, "metric": res})
        print(json.dumps(trials[-1]), flush=True)
    best = max(trials, key=lambda z: z["metric"]["macro_f05"])
    (out / "selection.json").write_text(json.dumps({"trials": trials, "selected": best}, indent=2))
    del frame, refs, q, t, y, p, top
    gc.collect()
    frame = load(test / "prepared-test.parquet", scores / "test_predictions.parquet")
    p = blend(frame, best["weights"])
    top = decode.winners(frame["qid"].to_numpy(), frame["tid"].to_numpy(), p)
    accepted = frame[top[p[top] >= best["cut"]]].select("qid", "tid")
    del frame, p, top
    refs = pl.scan_parquet(data / "test/ref.parquet").filter(pl.col("co") == "france").select(pl.col("rid").alias("qid"), pl.col("nm").alias("s1_name"), pl.col("ad").alias("s1_addr")).collect()
    targets = pl.concat([pl.scan_parquet(data / "test" / f"s{i}.parquet").select(pl.col("rid").alias("tid"), pl.col("nm").alias("rec_name"), pl.col("ad").alias("rec_addr")) for i in (2, 3)])
    oldpool = (pl.scan_parquet(pool / "test_pred.parquet").select("qid", "tid", "p2").join(refs.select("qid").lazy(), on="qid", how="semi").collect(engine="streaming"))
    ids = decode.winners(oldpool["qid"].to_numpy(), oldpool["tid"].to_numpy(), oldpool["p2"].to_numpy())
    raw = oldpool[ids].select("qid", "tid").lazy().join(refs.lazy(), on="qid").join(targets, on="tid").collect(engine="streaming")
    pats = patterns(raw).filter(pl.col("swap"))
    counts = pats.group_by("wa", "wb").len("n")
    reverse = counts.rename({"wa": "wb", "wb": "wa", "n": "rev"})
    direction = counts.join(reverse, on=["wa", "wb"], how="left").with_columns(pl.col("rev").fill_null(0))
    bad = direction.filter((pl.col("n") + pl.col("rev") >= 20) & (pl.col("n") / (pl.col("n") + pl.col("rev")) < .7)).select("wa", "wb")
    del raw, pats, oldpool, ids
    rows = accepted.lazy().join(refs.lazy(), on="qid").join(targets, on="tid").collect(engine="streaming")
    dropped = patterns(rows).filter(pl.col("swap") & pl.col("same_house")).join(bad, on=["wa", "wb"], how="semi").select("qid", "tid")
    accepted = accepted.join(dropped, on=["qid", "tid"], how="anti")
    accepted.write_parquet(out / "accepted_france.parquet")
    if not pairs_only:
        export(accepted, data, baseline, out / "output", replace=("france",))
    (out / "result.json").write_text(json.dumps({"selected": best, "france_matches": len(accepted), "removed_two_way": len(dropped)}, indent=2))
    print("france exported", len(accepted), flush=True)


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    for name in ("data", "train", "test", "scores", "pool", "baseline", "out"):
        p.add_argument("--" + name, type=path, required=name != "baseline")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--weights", type=float, nargs=4)
    p.add_argument("--pairs-only", action="store_true")
    a = p.parse_args()
    if not a.pairs_only and a.baseline is None:
        p.error("--baseline is required for tsv export")
    if a.weights is not None:
        if not np.isfinite(a.weights).all() or min(a.weights) < 0 or sum(a.weights) <= 0:
            p.error("weights must be finite, nonnegative and nonzero")
        mixes = [tuple(a.weights)]
    run(a.data, a.train, a.test, a.scores, a.pool, a.baseline, a.out, a.threads, a.pairs_only)
