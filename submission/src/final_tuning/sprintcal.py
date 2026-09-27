'fast decision-layer optimization of the supplied collective-graph scores'
import argparse as ap
import json
from pathlib import Path as path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

import decode


def metric(q, y, keep, deg, wanted):
    count = np.bincount(q[keep], minlength=len(deg))
    tp = np.bincount(q[keep], weights=y[keep], minlength=len(deg))
    f = np.ones(len(deg))
    den = count + .25 * deg
    np.divide(1.25 * tp, den, out=f, where=den > 0)
    return {"macro_f05": float(f[wanted].mean()), "pairs": int(count[wanted].sum()),
            "precision": float(tp[wanted].sum() / max(1, count[wanted].sum())),
            "recall": float(tp[wanted].sum() / max(1, deg[wanted].sum()))}


def best_cut(p, q, y, deg, wanted):
    m = wanted[q]
    p, q, y = p[m], q[m], y[m]
    order = np.lexsort((-p, q))
    p, q, y = p[order], q[order], y[order]
    start = np.r_[True, q[1:] != q[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(len(q)), 0))
    c = np.cumsum(y)
    tp = c - np.where(first > 0, c[np.maximum(first - 1, 0)], 0)
    n = np.arange(len(q)) - first + 1
    after = 1.25 * tp / (n + .25 * deg[q])
    before = np.r_[0., after[:-1]]
    before[start] = (deg[q[start]] == 0)
    order = np.argsort(-p, kind="stable")
    p, gain = p[order], (after - before)[order]
    total = np.cumsum(gain)
    end = np.r_[p[:-1] != p[1:], True]
    ix = np.flatnonzero(end)
    return float(p[ix[np.argmax(total[ix])]])


def blank(data, split):
    d = pl.concat([pl.scan_parquet(data / split / f"s{i}.parquet").select("rid", (pl.col("ad").fill_null("").str.strip_chars() == "").alias("blank")) for i in (2, 3)]).collect(engine="streaming")
    out = np.zeros(int(d["rid"].max()) + 1, np.int8)
    out[d["rid"].to_numpy()] = d["blank"].to_numpy()
    return out


def fit_cal(p, y, segment, fitting):
    res = {}
    edges = np.linspace(-15, 15, 301)
    x = np.log(np.clip(p, 1e-7, 1-1e-7) / np.clip(1-p, 1e-7, 1))
    bins = np.clip(np.searchsorted(edges, x) - 1, 0, 299)
    centres = (edges[:-1] + edges[1:]) / 2
    for s in np.unique(segment):
        m = fitting & (segment == s)
        if m.sum() < 2000:
            m = fitting & (segment // 4 == s // 4)
        count = np.bincount(bins[m], minlength=300)
        pos = np.bincount(bins[m], weights=y[m], minlength=300)
        active = count > 0
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
        iso.fit(centres[active], (pos[active] + .5) / (count[active] + 1), sample_weight=count[active])
        res[str(int(s))] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    return res


def adjust(p, segment, curves):
    x = np.log(np.clip(p, 1e-7, 1-1e-7) / np.clip(1-p, 1e-7, 1))
    out = p.copy().astype(np.float64)
    for s in np.unique(segment):
        m = segment == s
        c = curves[str(int(s))]
        out[m] = np.interp(x[m], c["x"], c["y"])
    return out


def export(pred, data, baseline, out, replace=("india", "us")):
    refs = pl.read_parquet(data / "test/ref.parquet", columns=["rid", "eid", "co"])
    targets = pl.concat([pl.scan_parquet(data / "test" / f"s{i}.parquet").select(pl.col("rid").alias("tid"), pl.col("eid").alias("target")) for i in (2, 3)])
    joined = pred.lazy().join(targets, on="tid", how="inner").collect(engine="streaming")
    if len(joined) != len(pred) or pred["tid"].n_unique() != len(pred):
        raise ValueError("output lost ids or assigned multiple owners")
    lists = joined.group_by("qid").agg(pl.col("target").sort().str.join(",").alias("replacement"))
    old = pl.read_csv(baseline / "matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).with_columns(pl.col("matched_entity_ids").fill_null(""))
    new = (old.join(refs.select(pl.col("eid").alias("source1_entity_id"), pl.col("rid").alias("qid"), "co"), on="source1_entity_id", maintain_order="left", validate="1:1")
           .join(lists, on="qid", how="left", maintain_order="left", validate="1:1")
           .with_columns(pl.when(pl.col("co").is_in(replace)).then(pl.col("replacement").fill_null("")).otherwise(pl.col("matched_entity_ids")).alias("matched_entity_ids")))
    if len(new) != len(refs):
        raise ValueError("output source1 coverage changed")
    out.mkdir(parents=True, exist_ok=False)
    new.select("source1_entity_id", "matched_entity_ids").write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    return len(pred)


def run(data, scores, baseline, out, threads=12):
    out.mkdir(parents=True, exist_ok=True)
    refs = pl.read_parquet(data / "train/ref.parquet", columns=["rid", "co", "deg", "fold"]).sort("rid")
    degree = refs["deg"].to_numpy()
    country = np.where(refs["co"].to_numpy() == "india", 0, 1).astype(np.int8)
    partition = ((refs["rid"].to_numpy().astype(np.uint64) * 2654435761) % 3)
    cal = (refs["fold"].to_numpy() == 0) & (partition == 1)
    tune = (refs["fold"].to_numpy() == 0) & (partition == 2)
    frame = pl.read_parquet(scores / "validation_predictions.parquet")
    ids = decode.winners(frame["qid"].to_numpy(), frame["tid"].to_numpy(), frame["p"].to_numpy())
    top = frame[ids]
    del frame, ids
    q, t, p, y = (top[k].to_numpy() for k in ("qid", "tid", "p", "y"))
    co = country[q]
    missing = blank(data, "train")[t]
    s2 = pl.scan_parquet(data / "train/s2.parquet").select(pl.len()).collect().item()
    ref = np.where(co == 0, .9429982900619507, .9515875577926636)
    report = {"baseline": metric(q, y, p >= ref, degree, tune), "trials": []}
    options = [{"kind": "raw_threshold", "cuts": {"0": .9429982900619507, "1": .9515875577926636}, "metric": report["baseline"]}]
    cuts = {str(c): best_cut(p, q, y, degree, tune & (country == c)) for c in (0, 1)}
    keep = p >= np.where(co == 0, cuts["0"], cuts["1"])
    options.append({"kind": "raw_threshold", "cuts": cuts, "metric": metric(q, y, keep, degree, tune)})
    for kind in ("country", "country_source_blank"):
        segment = co * 4 + ((t >= s2).astype(np.int8) * 2 + missing if kind != "country" else 0)
        curves = fit_cal(p, y, segment, cal[q])
        calibrated = adjust(p, segment, curves)
        use = tune[q]
        for tilt in (0., .12):
            tilted = 1 / (1 + np.exp(-(np.log(calibrated / (1-calibrated)) + tilt)))
            chosen, _ = decode.choose(q[use], t[use], tilted[use], p[use], floor=.02, exact=64, threads=threads)
            keep = np.zeros(len(q), bool);keep[use] = chosen
            options.append({"kind": kind, "curves": curves, "tilt": tilt, "metric": metric(q, y, keep, degree, tune)})
    best = max(options, key=lambda z: z["metric"]["macro_f05"])
    report["trials"] = [{k:v for k,v in z.items() if k != "curves"} for z in options]
    report["selected"] = best
    (out / "selection.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"baseline": report["baseline"], "selected": {k:v for k,v in best.items() if k != "curves"}}), flush=True)
    del top, q, t, p, y, co, missing, country, refs, calibrated, tilted, segment, keep
    refs = pl.read_parquet(data / "test/ref.parquet", columns=["rid", "co"]).sort("rid")
    countries = np.where(refs["co"].to_numpy() == "india", 0, np.where(refs["co"].to_numpy() == "us", 1, 2)).astype(np.int8)
    frame = pl.read_parquet(scores / "test_predictions.parquet")
    ids = decode.winners(frame["qid"].to_numpy(), frame["tid"].to_numpy(), frame["p"].to_numpy())
    top = frame[ids].filter(pl.col("qid").is_in(refs.filter(pl.col("co") != "france")["rid"].implode()))
    del frame, ids
    q, t, p = (top[k].to_numpy() for k in ("qid", "tid", "p"));co = countries[q]
    if best["kind"] == "raw_threshold":
        keep = p >= np.where(co == 0, best["cuts"]["0"], best["cuts"]["1"])
    else:
        s2 = pl.scan_parquet(data / "test/s2.parquet").select(pl.len()).collect().item()
        segment = co * 4 + ((t >= s2).astype(np.int8) * 2 + blank(data, "test")[t] if best["kind"] != "country" else 0)
        pp = adjust(p, segment, best["curves"])
        pp = 1 / (1 + np.exp(-(np.log(pp / (1-pp)) + best["tilt"])))
        keep, _ = decode.choose(q, t, pp, p, floor=.02, exact=64, threads=threads)
    report["known_country_matches"] = export(top.filter(pl.Series(keep)).select("qid", "tid"), data, baseline, out / "output")
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print("exported", out / "output/matching_results.tsv", flush=True)


def check():
    q, y, p = np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([.9, .7, .8])
    deg, wanted = np.array([1, 0]), np.array([True, True])
    assert abs(best_cut(p, q, y, deg, wanted) - .9) < 1e-8
    assert metric(q, y, p >= .9, deg, wanted)["macro_f05"] == 1


if __name__ == "__main__":
    parser = ap.ArgumentParser(description=__doc__)
    for field in ("data", "scores", "baseline", "out"):
        parser.add_argument("--" + field, type=path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    run(args.data, args.scores, args.baseline, args.out, args.threads)
