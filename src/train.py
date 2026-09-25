import argparse as ap
import hashlib as hh
import json
import os
import tempfile as tf
from pathlib import Path as path

import joblib as jl
import numpy as np
import polars as pl

try:
    from feat import ff, make, prep
except ImportError:
    from src.feat import ff, make, prep


ver = 1


def _need(d, cs):
    z = set(d.columns)
    if not z.issuperset(cs):
        raise ValueError(f"missing columns {sorted(set(cs) - z)}")


def _sha(p):
    h = hh.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"invalid json {p}") from e


def _save_json(p, z):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(z, indent=2) + "\n", encoding="utf-8")
    t.replace(p)


def _part_paths(run, met):
    ps = met.get("parts")
    if not isinstance(ps, list) or not ps:
        raise ValueError(f"run has no listed parts {run}")
    out = []
    for s in ps:
        if not isinstance(s, str):
            raise ValueError(f"invalid part name {run}")
        p = (run / s).resolve()
        if run.resolve() not in p.parents or p.suffix != ".parquet" or not p.is_file():
            raise ValueError(f"invalid listed part {s}")
        out.append(p)
    if len(set(out)) != len(out):
        raise ValueError(f"repeated listed part {run}")
    return out


def _target_rows(data, q):
    fs = []
    for sr in (2, 3):
        ids = q.filter(pl.col("sr") == sr)["rid"].to_list()
        if ids:
            d = pl.scan_parquet(data / "train" / f"s{sr}.parquet").filter(
                pl.col("rid").is_in(ids)).select("rid", "sr", "own").collect(engine="streaming")
            fs.append(d)
    return pl.concat(fs) if fs else pl.DataFrame({"rid": [], "sr": [], "own": []})


def _run(data, run, fold):
    run = path(run).resolve()
    met = _json(run / "metrics.json")
    if met.get("fold") != fold:
        raise ValueError(f"{run} must be fold{fold}")
    co = met.get("country")
    if not isinstance(co, str) or not co:
        raise ValueError(f"invalid country {run}")
    ps = _part_paths(run, met)
    a = pl.read_parquet(run / "anchors.parquet")
    q = pl.read_parquet(run / "queries.parquet")
    _need(a, {"rid", "co", "fold", "deg"})
    _need(q, {"rid", "co", "sr", "own", "eid", "nm", "ad", "nn", "an"})
    if a.is_empty() or q.is_empty() or a["rid"].n_unique() != len(a) or q["rid"].n_unique() != len(q):
        raise ValueError(f"invalid anchor or query ids {run}")
    if a.filter((pl.col("co") != co) | (pl.col("fold") != fold)).height:
        raise ValueError(f"invalid selected anchors {run}")
    if q.filter((pl.col("co") != co) | ~pl.col("sr").is_in([2, 3])).height:
        raise ValueError(f"invalid selected queries {run}")
    full = pl.read_parquet(data / "train/ref.parquet").filter(pl.col("co") == co)
    pool = full.filter(pl.col("fold") == 2) if fold == 2 else full
    if pool.is_empty() or a.filter(~pl.col("rid").is_in(pool["rid"].implode())).height:
        raise ValueError(f"anchors not in permitted reference pool {run}")
    chk = a.select("rid", "co", "fold", "deg").join(full.select("rid", "co", "fold", "deg"), on="rid", how="left", suffix="_data")
    if chk.filter((pl.col("co") != pl.col("co_data")) | (pl.col("fold") != pl.col("fold_data")) |
                  (pl.col("deg") != pl.col("deg_data"))).height:
        raise ValueError(f"stale selected anchors {run}")
    auth = _target_rows(data, q)
    chk = q.select("rid", "sr", "own").join(auth, on="rid", how="left", suffix="_data")
    if len(auth) != len(q) or chk.filter(
            pl.col("sr") != pl.col("sr_data")) .height or chk.filter(pl.col("own") != pl.col("own_data")).height:
        raise ValueError(f"query ownership disagrees with data {run}")
    deg = q.filter(pl.col("own").is_in(a["rid"].implode())).group_by("own").len().rename({"own": "rid", "len": "n"})
    if q.filter((pl.col("own") >= 0) & ~pl.col("own").is_in(a["rid"].implode())).height:
        raise ValueError(f"queries include an unselected anchor alias {run}")
    chk = a.select("rid", "deg").join(deg, on="rid", how="left").with_columns(pl.col("n").fill_null(0))
    if chk.filter(pl.col("deg") != pl.col("n")).height:
        raise ValueError(f"queries do not contain every selected anchor alias {run}")
    return {"dir": run, "met": met, "co": co, "parts": ps, "anchors": a, "queries": q, "pool": pool}


def _pair(run, p, seen):
    d = pl.read_parquet(p)
    _need(d, {"qid", "tid", "ns", "ads", "en", "ea", "sr", "own", "y"})
    if d.select("qid", "tid").n_unique() != len(d):
        raise ValueError(f"duplicate pair {p}")
    q = run["queries"].select("rid", "sr", "own").rename({"rid": "tid"})
    d = d.join(q, on="tid", how="left", suffix="_query", validate="m:1")
    if d["own_query"].null_count() or d.filter(
            (pl.col("sr") != pl.col("sr_query")) |
            (pl.col("own") != pl.col("own_query")) |
            (pl.col("y") != (pl.col("qid").cast(pl.Int64) == pl.col("own_query")).cast(pl.UInt8))).height:
        raise ValueError(f"pair labels disagree with query ownership {p}")
    tids = set(d["tid"].to_list())
    if tids & seen:
        raise ValueError(f"query split across parts {p}")
    seen.update(tids)
    return d.drop("sr_query", "own_query")


def _key(data, run, p):
    z = {"v": ver, "features": ff, "data": _sha(data / "meta.json"), "metrics": _sha(run["dir"] / "metrics.json"),
         "anchors": _sha(run["dir"] / "anchors.parquet"), "queries": _sha(run["dir"] / "queries.parquet"), "part": _sha(p),
         "pool": run["pool"]["rid"].to_list()}
    return hh.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()


def _features(data, out, run, threads):
    st = prep(run["pool"])
    seen, xs, ys, ts, qs = set(), [], [], [], []
    for p in run["parts"]:
        k = _key(data, run, p)
        cp = out / "features" / f"{k}.npz"
        d = _pair(run, p, seen)
        if cp.exists():
            z = np.load(cp)
            x, y, tid, qid = z["x"], z["y"], z["tid"], z["qid"]
        else:
            x, names = make(st, run["queries"], d, threads)
            if names != ff:
                raise ValueError("unexpected feature names")
            y = d["y"].to_numpy().astype(np.uint8, copy=False)
            tid = d["tid"].to_numpy().astype(np.uint32, copy=False)
            qid = d["qid"].to_numpy().astype(np.uint32, copy=False)
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cp, x=x, y=y, tid=tid, qid=qid)
        if x.ndim != 2 or x.shape != (len(y), len(ff)) or not np.isfinite(x).all():
            raise ValueError(f"invalid cached features {cp}")
        xs.append(x.astype(np.float32, copy=False))
        ys.append(y.astype(np.uint8, copy=False))
        ts.append(tid.astype(np.uint32, copy=False))
        qs.append(qid.astype(np.uint32, copy=False))
    if not xs:
        raise ValueError(f"no pairs {run['dir']}")
    return tuple(np.concatenate(z) for z in (xs, ys, ts, qs))


def anchor_f05(anchors, qid, y, keep):
    n, tp = {}, {}
    for i, ok, hit in zip(qid, keep, y):
        if ok:
            i = int(i)
            n[i] = n.get(i, 0) + 1
            if hit:
                tp[i] = tp.get(i, 0) + 1
    rows = []
    for i, d in anchors.select("rid", "deg").iter_rows():
        p, h = n.get(i, 0), tp.get(i, 0)
        f = 1.0 if d == 0 and p == 0 else (1.25 * h / (p + 0.25 * d) if d else 0.0)
        rows.append((i, d, p, h, f))
    return pl.DataFrame(rows, schema=["rid", "deg", "predicted", "true_positive", "f05"], orient="row")


def f05(anchors, qid, y, keep):
    z = anchor_f05(anchors, qid, y, keep)
    return float(z["f05"].mean()) if len(z) else 0.0


def _scores(anchors, qid, tid, y, prob, keep):
    pred = int(keep.sum())
    hit = int(y[keep].sum())
    truth = int(anchors["deg"].sum())
    return {"macro_f05": f05(anchors, qid, y, keep), "pair_precision": hit / pred if pred else 1.0,
            "pair_recall": hit / truth if truth else 1.0, "pairs": pred, "true_pairs": truth}


def _tops(tid, qid, prob):
    o = np.lexsort((qid, -prob, tid))
    return o[np.r_[True, tid[o][1:] != tid[o][:-1]]]


def _thresholds(prob):
    u = np.unique(prob)
    if len(u) > 2001:
        u = np.unique(np.quantile(u, np.linspace(0, 1, 2001)))
    return np.r_[u, np.nextafter(1.0, 2.0)]


def evaluate(anchors, qid, tid, y, prob):
    out = {}
    for name, ix in (("plain_threshold", np.arange(len(prob))), ("target_top1_then_threshold", _tops(tid, qid, prob))):
        best = None
        for th in _thresholds(prob):
            keep = np.zeros(len(prob), dtype=bool)
            keep[ix] = prob[ix] >= th
            z = _scores(anchors, qid, tid, y, prob, keep)
            z["threshold"] = float(th)
            if best is None or (z["macro_f05"], z["pair_precision"], z["threshold"]) > (
                    best["macro_f05"], best["pair_precision"], best["threshold"]):
                best = z
        out[name] = best
    return out


def _fit_lgb(x, y, xv, yv, trees, threads):
    import lightgbm as lgb

    m = lgb.LGBMClassifier(n_estimators=trees, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                            subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=threads, verbosity=-1)
    m.fit(x, y, eval_X=xv, eval_y=yv, eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def _fit_cat(x, y, xv, yv, trees, threads):
    from catboost import CatBoostClassifier

    m = CatBoostClassifier(iterations=trees, learning_rate=0.05, depth=6, loss_function="Logloss",
                            eval_metric="Logloss", random_seed=42, thread_count=threads,
                            allow_writing_files=False, verbose=False)
    m.fit(x, y, eval_set=(xv, yv), early_stopping_rounds=50, verbose=False)
    return m


def predict(m, x):
    return np.asarray(m.predict_proba(x)[:, 1], dtype=np.float64)


def load_models(out):
    out = path(out)
    met = _json(out / "metadata.json")
    if met.get("feature_names") != ff:
        raise ValueError("model feature order mismatch")
    z = {}
    for name in met["models"]:
        p = out / met["model_files"][name]
        if name == "lgb":
            z[name] = jl.load(p)
        elif name == "cat":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(p)
            z[name] = m
        else:
            raise ValueError(f"unknown model {name}")
    return z, met


def train(data, trs, vals, out, model="lgb", threads=8, trees=800):
    if threads < 1 or trees < 1:
        raise ValueError("positive threads and trees required")
    data, out = path(data).resolve(), path(out).resolve()
    if not (data / "meta.json").is_file():
        raise ValueError(f"missing prepared data {data}")
    if not trs or not vals:
        raise ValueError("training and validation runs required")
    tr = [_run(data, p, 2) for p in trs]
    va = [_run(data, p, 0) for p in vals]
    svs = {r["met"].get("score_version", 1) for r in tr + va}
    if len(svs) != 1:
        raise ValueError("mixed candidate score versions")
    sv = svs.pop()
    if len({r["dir"] for r in tr + va}) != len(tr) + len(va):
        raise ValueError("run cannot be both training and validation")
    xa = [_features(data, out, r, threads) for r in tr]
    xv = [_features(data, out, r, threads) for r in va]
    x, y = np.concatenate([z[0] for z in xa]), np.concatenate([z[1] for z in xa])
    vx, vy = np.concatenate([z[0] for z in xv]), np.concatenate([z[1] for z in xv])
    tid, qid = np.concatenate([z[2] for z in xv]), np.concatenate([z[3] for z in xv])
    anchors = pl.concat([r["anchors"].select("rid", "deg") for r in va])
    vtid = np.concatenate([r["queries"]["rid"].to_numpy() for r in va])
    if anchors["rid"].n_unique() != len(anchors) or len(np.unique(vtid)) != len(vtid):
        raise ValueError("validation runs overlap anchors or queries")
    if not (y.min() == 0 and y.max() == 1):
        raise ValueError("training pairs need both labels")
    if not (vy.min() == 0 and vy.max() == 1):
        raise ValueError("validation pairs need both labels")
    fitters = {"lgb": _fit_lgb, "cat": _fit_cat}
    names = ["lgb", "cat"] if model == "both" else [model]
    out.mkdir(parents=True, exist_ok=True)
    res, files, raw, per = {}, {}, [], []
    for name in names:
        m = fitters[name](x, y, vx, vy, trees, threads)
        prob = predict(m, vx)
        met = evaluate(anchors, qid, tid, vy, prob)
        res[name] = met
        raw.append(pl.DataFrame({"model": [name] * len(prob), "tid": tid, "qid": qid, "y": vy, "prob": prob,
                                 "plain_selected": prob >= met["plain_threshold"]["threshold"],
                                 "top1_selected": np.isin(np.arange(len(prob)), _tops(tid, qid, prob)) &
                                 (prob >= met["target_top1_then_threshold"]["threshold"])}))
        for dec, key in (("plain_threshold", "plain_selected"), ("target_top1_then_threshold", "top1_selected")):
            keep = raw[-1][key].to_numpy()
            per.append(anchor_f05(anchors, qid, vy, keep).with_columns(pl.lit(name).alias("model"), pl.lit(dec).alias("decoder")))
        if name == "lgb":
            p = "lgb.joblib"
            jl.dump(m, out / p)
        else:
            p = "cat.cbm"
            m.save_model(out / p)
        files[name] = p
    pl.concat(raw).write_parquet(out / "validation_predictions.parquet", compression="zstd")
    pl.concat(per).write_parquet(out / "validation_anchors.parquet", compression="zstd")
    met = {"version": ver, "seed": 42, "score_version": sv, "feature_names": ff, "models": names, "model_files": files,
           "configuration": {"data": str(data), "data_meta_sha256": _sha(data / "meta.json"), "threads": threads, "trees": trees, "class_weights": None,
                             "augmentation": False, "train_runs": [str(r["dir"]) for r in tr],
                             "validation_runs": [str(r["dir"]) for r in va]},
           "validation": {"label": "sampled-query estimate, not official/full-pool validation",
                          "warning": "validation queries contain selected anchor aliases and sampled orphans, not every target in the full corpus",
                          "anchors": len(anchors), "queries": sum(len(r["queries"]) for r in va), "models": res}}
    _save_json(out / "metadata.json", met)
    print(json.dumps(met["validation"], indent=2), flush=True)
    return met


def _frame(rows, schema):
    return pl.DataFrame(rows, schema=schema, strict=False, orient="row")


def check():
    with tf.TemporaryDirectory() as tmp:
        root, data, runs, out = path(tmp), path(tmp) / "data", path(tmp) / "runs", path(tmp) / "model"
        (data / "train").mkdir(parents=True)
        _save_json(data / "meta.json", {"check": True})
        ref = _frame([
            [10, "S1-10", "alpha", "1 main", "alpha", "1 main", "us", 1, 1, 0, 0, 2],
            [11, "S1-11", "beta", "2 main", "beta", "2 main", "us", 1, 1, 0, 0, 2],
            [20, "S1-20", "gamma", "3 main", "gamma", "3 main", "us", 1, 1, 0, 0, 0],
        ], ["rid", "eid", "nm", "ad", "nn", "an", "co", "sr", "deg", "uni", "blank", "fold"])
        ref = ref.with_columns(pl.col("rid").cast(pl.UInt32), pl.col("sr").cast(pl.UInt8), pl.col("deg").cast(pl.UInt32),
                               pl.col("fold").cast(pl.UInt8))
        ref.write_parquet(data / "train/ref.parquet")
        s2 = _frame([
            [100, "S2-100", "alpha", "1 main", "alpha", "1 main", "us", 2, 10],
            [101, "S2-101", "beta", "9 wrong", "beta", "9 wrong", "us", 2, 11],
            [200, "S2-200", "gamma", "3 main", "gamma", "3 main", "us", 2, 20],
            [201, "S2-201", "other", "7 lane", "other", "7 lane", "us", 2, -1],
        ], ["rid", "eid", "nm", "ad", "nn", "an", "co", "sr", "own"])
        s2 = s2.with_columns(pl.col("rid").cast(pl.UInt32), pl.col("sr").cast(pl.UInt8), pl.col("own").cast(pl.Int32))
        s2.write_parquet(data / "train/s2.parquet")
        s2.head(0).write_parquet(data / "train/s3.parquet")
        for name, fold, ids in (("tr", 2, [100, 101]), ("va", 0, [200, 201])):
            d = runs / name
            d.mkdir(parents=True)
            q = s2.filter(pl.col("rid").is_in(ids))
            a = ref.filter(pl.col("fold") == fold)
            a.write_parquet(d / "anchors.parquet")
            q.write_parquet(d / "queries.parquet")
            rows = []
            for tid, own in q.select("rid", "own").iter_rows():
                for rid in ([10, 11] if fold == 2 else [20, 10]):
                    rows.append([rid, tid, 0.9 if rid == own else 0.1, 0.9 if rid == own else 0.1, int(rid == own), 0, 2, own, int(rid == own)])
            p = _frame(rows, ["qid", "tid", "ns", "ads", "en", "ea", "sr", "own", "y"]).with_columns(
                pl.col("qid").cast(pl.UInt32), pl.col("tid").cast(pl.UInt32), pl.col("sr").cast(pl.UInt8),
                pl.col("own").cast(pl.Int32), pl.col("y").cast(pl.UInt8))
            p.write_parquet(d / "pairs_00000.parquet")
            _save_json(d / "metrics.json", {"country": "us", "fold": fold, "parts": ["pairs_00000.parquet"]})
        met = train(data, [runs / "tr"], [runs / "va"], out, model="both", trees=10, threads=1)
        ms, got = load_models(out)
        assert got["feature_names"] == ff and set(ms) == {"lgb", "cat"}
        assert predict(ms["lgb"], np.zeros((1, len(ff)), np.float32)).shape == (1,)
        assert predict(ms["cat"], np.zeros((1, len(ff)), np.float32)).shape == (1,)
        z = evaluate(_frame([[20, 1]], ["rid", "deg"]), np.array([20, 10]), np.array([200, 201]),
                     np.array([1, 0], np.uint8), np.array([0.9, 0.1]))
        assert z["plain_threshold"]["macro_f05"] == 1.0
        assert z["target_top1_then_threshold"]["macro_f05"] == 1.0
        bad = _json(runs / "va" / "metrics.json")
        bad["fold"] = 1
        _save_json(runs / "va" / "metrics.json", bad)
        try:
            train(data, [runs / "tr"], [runs / "va"], out, trees=10, threads=1)
        except ValueError:
            pass
        else:
            raise AssertionError("locked fold accepted")
        assert f05(_frame([[1, 1]], ["rid", "deg"]), np.array([1]), np.array([1]), np.array([True])) == 1.0
    print("checks passed")


def main():
    root = path(__file__).resolve().parents[1]
    pa = ap.ArgumentParser()
    pa.add_argument("--data", type=path, default=root / "cache/data")
    pa.add_argument("--train", type=path, nargs="+")
    pa.add_argument("--val", type=path, nargs="+")
    pa.add_argument("--out", type=path, default=root / "cache/models/base")
    pa.add_argument("--model", choices=["lgb", "cat", "both"], default="lgb")
    pa.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    pa.add_argument("--trees", type=int, default=800)
    pa.add_argument("--check", action="store_true")
    a = pa.parse_args()
    if a.check:
        check()
    else:
        train(a.data, a.train, a.val, a.out, a.model, a.threads, a.trees)


if __name__ == "__main__":
    main()
