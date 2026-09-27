'label-shift calibration of stacker scores for the test pool (no test labels)'
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from er.stack import decode  # noqa: E402

MIN_P, BINS, TOP_P = 0.02, 25, 0.99


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


EDGES = np.linspace(_logit(np.array([MIN_P]))[0], _logit(np.array([0.999]))[0], BINS)
CENTRES = np.r_[EDGES[0] - 1.0, (EDGES[:-1] + EDGES[1:]) / 2, EDGES[-1] + 1.0]
TOP = np.flatnonzero(np.r_[-np.inf, EDGES] >= _logit(np.array([TOP_P]))[0])


def _hist(p, y, n_ent):
    idx = np.digitize(_logit(p), EDGES)
    if y is None:
        return np.bincount(idx, minlength=BINS + 1) / n_ent
    return np.bincount(idx, weights=y, minlength=BINS + 1) / n_ent


def segments(pairs: pl.DataFrame, s1_addr: pl.DataFrame, t_addr: pl.DataFrame) -> pl.DataFrame:
    "house relation: 'equal' = source-1 house number among the target's address numbers"
    nums = lambda c: pl.col(c).fill_null("").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
    s = s1_addr.select("qid", pl.col("ad").fill_null("").str.extract(r"(\d+)", 1).str.strip_chars_start("0").fill_null("").alias("hs"))
    t = t_addr.select("tid", nums("ad").alias("nums"))
    d = pairs.join(s, on="qid", how="left").join(t, on="tid", how="left")
    rel = (pl.when((pl.col("hs") != "") & pl.col("nums").list.contains(pl.col("hs"))).then(pl.lit("equal"))
             .when((pl.col("hs") != "") & (pl.col("nums").list.len() > 0)).then(pl.lit("conflict"))
             .otherwise(pl.lit("unknown")))
    return d.with_columns(rel.alias("seg")).drop("hs", "nums")


def fit(hold: pl.DataFrame, test: pl.DataFrame, n_hold: dict, n_test: dict) -> dict:
    'hold: qid, p, y, co, seg (validation pairs of the audit anchors); test: qid, p, co, seg'
    fits = {}
    for co in sorted(test["co"].unique().to_list()):
        h = hold.filter(pl.col("co") == co)
        nh = n_hold.get(co)
        if not h.height:
            h, nh = hold, n_hold["*"]
        t = test.filter(pl.col("co") == co)
        nt = n_test[co]
        n1_all = _hist(h["p"].to_numpy(), h["y"].to_numpy().astype(float), nh)
        m_all = _hist(t["p"].to_numpy(), None, nt)
        a = float(m_all[TOP].sum() / max(n1_all[TOP].sum(), 1e-9))
        for seg in ("equal", "conflict", "unknown"):
            hs, ts = h.filter(pl.col("seg") == seg), t.filter(pl.col("seg") == seg)
            n1 = _hist(hs["p"].to_numpy(), hs["y"].to_numpy().astype(float), nh)
            m = _hist(ts["p"].to_numpy(), None, nt)
            post = np.clip(a * n1 / np.maximum(m, 1e-9), 0, 1)
            post[m < 1e-6] = np.nan
            post = pl.Series(post).fill_nan(None).fill_null(strategy="forward").fill_null(0.0).to_numpy().copy()
            post[0] = min(post[0], post[1])
            post = np.maximum.accumulate(post)
            fits[f"{co}|{seg}"] = {"a": a, "posterior": post.round(4).tolist()}
    return fits


def apply(test: pl.DataFrame, fits: dict) -> pl.DataFrame:
    parts = []
    for (cell,), g in test.with_columns((pl.col("co") + "|" + pl.col("seg")).alias("_cell")).group_by("_cell"):
        f = fits.get(cell)
        p = g["p"].to_numpy()
        if f is None:
            parts.append(g.drop("_cell"))
            continue
        adj = np.interp(_logit(p), CENTRES, np.asarray(f["posterior"]))
        adj = np.where(p >= MIN_P, adj, np.minimum(p, adj))
        parts.append(g.drop("_cell").with_columns(pl.Series("p", adj.astype(np.float32))))
    return pl.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_train")
    ap.add_argument("ref_test")
    ap.add_argument("tsv_dir")
    ap.add_argument("stack_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--col", default="p2")
    ap.add_argument("--audit", type=int, default=1)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--sanity", action="store_true", help="also calibrate fold 0 against fold 1 (no shift expected)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    def taddr(split):
        parts, off = [], 0
        for s in ("2", "3"):
            x = pl.read_csv(os.path.join(a.tsv_dir, split, f"{split}_source{s}.tsv"), separator="\t", quote_char=None,
                            columns=["business_address"]).with_row_index("tid")
            parts.append(x.with_columns((pl.col("tid") + off).cast(pl.UInt32)).rename({"business_address": "ad"}))
            off += x.height
        return pl.concat(parts)

    rtr = pl.read_parquet(a.ref_train, columns=["rid", "fold", "co", "ad", "deg"]).rename({"rid": "qid"}).with_columns(
        pl.col("co").str.to_lowercase())
    rte = pl.read_parquet(a.ref_test, columns=["rid", "co", "ad"]).rename({"rid": "qid"}).with_columns(pl.col("co").str.to_lowercase())

    vp = pl.read_parquet(os.path.join(a.stack_dir, "val_pred.parquet"), columns=["qid", "tid", "y", a.col]).rename({a.col: "p"})
    hold_all = segments(vp.filter(pl.col("p") >= 0.001), rtr.select("qid", "ad"), taddr("train")).join(
        rtr.select("qid", "fold", "co"), on="qid")
    report = {}

    def run(hold, test_pairs, anchors_test, n_hold, n_test, tag, labelled_test=None):
        fits = fit(hold, test_pairs, n_hold, n_test)
        adj = apply(test_pairs, fits)
        acc = decode.apply("expected_f", adj.select("qid", "tid", "p"), a.floor)
        out = {"fits": {k: {"a": round(v["a"], 3), "post": v["posterior"][1:]} for k, v in fits.items()}}
        if labelled_test is not None:
            out["score_calibrated"] = decode.score(acc.join(labelled_test, on=["qid", "tid"], how="left")
                                                   .with_columns(pl.col("y").fill_null(0)).select("qid", "y"), anchors_test)
        report[tag] = out
        return acc

    def counts(r, folds=None):
        r = r if folds is None else r.filter(pl.col("fold").is_in(folds))
        d = dict(r.group_by("co").len().rows())
        d["*"] = r.height
        return d

    hold = hold_all.filter(pl.col("fold") == a.audit)
    n_hold = counts(rtr, [a.audit])
    if a.sanity:
        other = 1 - a.audit
        anchors = rtr.filter(pl.col("fold") == other).select("qid", "deg")
        pairs_all = hold_all.filter(pl.col("fold") == other).select("qid", "tid", "p", "co", "seg")
        lab = vp.select("qid", "tid", "y")
        for fl in (0.5, a.floor):
            base = decode.apply("expected_f", pairs_all.select("qid", "tid", "p"), fl).join(lab, on=["qid", "tid"])
            report["sanity_uncalibrated_floor%s" % fl] = decode.score(base.select("qid", "y"), anchors)
        run(hold, pairs_all, anchors, n_hold, counts(rtr, [other]), "sanity_fold%d" % other,
            labelled_test=vp.select("qid", "tid", "y"))
        for k, v in report.items():
            print(k, v.get("score_calibrated", v) if isinstance(v, dict) else v, flush=True)
        del pairs_all, lab
    del vp, hold_all

    tp = pl.read_parquet(os.path.join(a.stack_dir, "test_pred.parquet"), columns=["qid", "tid", a.col]).rename({a.col: "p"})
    live = segments(tp.filter(pl.col("p") >= 0.001), rte.select("qid", "ad"), taddr("test")).join(rte.select("qid", "co"), on="qid")
    acc = run(hold, live.select("qid", "tid", "p", "co", "seg"), None, n_hold, counts(rte), "test")
    acc.write_parquet(os.path.join(a.out_dir, "test_accepted.parquet"))
    per = acc.join(rte.select("qid", "co"), on="qid").group_by("co").len().join(rte.group_by("co").len().rename({"len": "s1"}), on="co")
    report["test"]["matches_per_s1"] = {r[0]: round(r[1] / r[2], 3) for r in per.rows()}
    report["test"]["matches"] = acc.height
    json.dump(report, open(os.path.join(a.out_dir, "calibration.json"), "w"), indent=1, default=str)
    print("test:", report["test"]["matches"], report["test"]["matches_per_s1"], flush=True)
    for k, v in report["test"]["fits"].items():
        print(k, "a=%.3f" % v["a"], v["post"], flush=True)


if __name__ == "__main__":
    main()
