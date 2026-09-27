'decision rules on scored pairs (qid, tid, p) and the challenge metric on entity folds'
import numpy as np
import polars as pl

from er.decision import MAX_GROUP, best_k


def top1(d: pl.DataFrame, col: str = "p") -> pl.DataFrame:
    return d.sort(["tid", col, "qid"], descending=[False, True, False]).unique("tid", keep="first", maintain_order=True)


def curve(pred: pl.DataFrame, anchors: pl.DataFrame) -> dict:
    'Best threshold for macro F0.5 over `anchors` (qid, deg), given pred (qid, p, y) already'
    n = anchors.height
    base = int((anchors["deg"] == 0).sum())
    truth = int(anchors["deg"].sum())
    d = (pred.join(anchors, on="qid", how="inner").sort(["qid", "p"], descending=[False, True])
             .with_columns(pl.int_range(1, pl.len() + 1).over("qid").alias("pn"),
                           pl.col("y").cast(pl.Int64).cum_sum().over("qid").alias("tn")))
    after = pl.when(pl.col("deg") > 0).then(1.25 * pl.col("tn") / (pl.col("pn") + 0.25 * pl.col("deg"))).otherwise(0.0)
    d = d.with_columns(after.alias("after")).with_columns(
        pl.col("after").shift(1).over("qid").fill_null(pl.when(pl.col("deg") == 0).then(1.0).otherwise(0.0)).alias("before"))
    g = (d.group_by("p").agg((pl.col("after") - pl.col("before")).sum().alias("delta"), pl.len().alias("np"),
                             pl.col("y").cast(pl.Int64).sum().alias("tp"))
          .sort("p", descending=True)
          .with_columns(((base + pl.col("delta").cum_sum()) / n).alias("f"), pl.col("np").cum_sum().alias("cnp"),
                        pl.col("tp").cum_sum().alias("ctp")))
    if g.height == 0:
        return {"threshold": 1.0, "macro_f05": base / n, "pair_precision": 1.0, "pair_recall": 0.0, "pairs": 0}
    i = int(np.argmax(g["f"].to_numpy()))
    r = g.row(i, named=True)
    if base / n >= r["f"]:
        return {"threshold": 1.0, "macro_f05": base / n, "pair_precision": 1.0, "pair_recall": 0.0, "pairs": 0}
    return {"threshold": float(r["p"]), "macro_f05": float(r["f"]), "pair_precision": r["ctp"] / r["cnp"],
            "pair_recall": r["ctp"] / truth if truth else 1.0, "pairs": int(r["cnp"])}


def score(pred: pl.DataFrame, anchors: pl.DataFrame) -> dict:
    'macro f0.5 of accepted pairs pred (qid, y) over anchors (qid, deg)'
    z = pred.join(anchors.select("qid"), on="qid", how="inner").group_by("qid").agg(
        pl.len().alias("np"), pl.col("y").cast(pl.Int64).sum().alias("tp"))
    z = anchors.join(z, on="qid", how="left").with_columns(pl.col("np", "tp").fill_null(0))
    npred, tp, deg = (z[c].to_numpy().astype(float) for c in ("np", "tp", "deg"))
    f = np.where(deg == 0, (npred == 0).astype(float), 1.25 * tp / np.maximum(npred + 0.25 * deg, 1e-9))
    return {"macro_f05": float(f.mean()), "pair_precision": float(tp.sum() / max(npred.sum(), 1)),
            "pair_recall": float(tp.sum() / max(deg.sum(), 1)), "pairs": int(npred.sum())}


def expected_f(pairs: pl.DataFrame, floor: float) -> pl.DataFrame:
    'pairs: qid, tid, p (+ y). returns accepted rows'
    best = top1(pairs).filter(pl.col("p") >= floor).sort(["qid", "p"], descending=[False, True])
    q = best["qid"].to_numpy()
    p = best["p"].to_numpy().astype(np.float64)
    keep = np.zeros(len(q), dtype=bool)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]]) if len(q) else np.array([], int)
    ends = np.r_[starts[1:], len(q)]
    for s, e in zip(starts, ends):
        if e - s == 1:
            keep[s] = p[s] > 0.5
            continue
        k = best_k(p[s:min(e, s + MAX_GROUP)])
        keep[s:s + k] = True
    return best.filter(pl.Series(keep))


def apply(rule: str, pairs: pl.DataFrame, thr: float) -> pl.DataFrame:
    if rule == "top1_threshold":
        return top1(pairs).filter(pl.col("p") >= thr)
    if rule == "plain_threshold":
        return pairs.filter(pl.col("p") >= thr)
    if rule == "expected_f":
        return expected_f(pairs, thr)
    raise ValueError(rule)


def tune(pairs: pl.DataFrame, anchors: pl.DataFrame, rules=("top1_threshold", "plain_threshold", "expected_f"),
         floors=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5)) -> dict:
    'Best rule + threshold on `anchors` (the tuning fold). pairs: all scored pairs that can'
    out = {}
    if "top1_threshold" in rules:
        out["top1_threshold"] = curve(top1(pairs).select("qid", "p", "y"), anchors)
    if "plain_threshold" in rules:
        out["plain_threshold"] = curve(pairs.select("qid", "p", "y"), anchors)
    if "expected_f" in rules:
        best = None
        for fl in floors:
            r = {"threshold": fl, **score(expected_f(pairs, fl), anchors)}
            if best is None or r["macro_f05"] > best["macro_f05"]:
                best = r
        out["expected_f"] = best
    rule = max(out, key=lambda k: out[k]["macro_f05"])
    return {"rules": out, "rule": rule, "threshold": out[rule]["threshold"], "macro_f05": out[rule]["macro_f05"]}
