import argparse as ap
import hashlib as hh
import json
import os
import shutil as sh
import tempfile as tf
from pathlib import Path as path

import joblib as jl
import numpy as np
import polars as pl

try:
    from feat import df, ff, make, prep
except ImportError:
    from src.feat import df, ff, make, prep


ver = 2


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
    ds = met.get("dense_features", [])
    if not isinstance(ds, list) or len(set(ds)) != len(ds) or any(not isinstance(x, str) or x not in df for x in ds):
        raise ValueError(f"invalid dense features {run}")
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
    src = {"metrics": _sha(run / "metrics.json"), "anchors": _sha(run / "anchors.parquet"),
           "queries": _sha(run / "queries.parquet"), "parts": {str(p.relative_to(run)): _sha(p) for p in ps}}
    return {"dir": run, "met": met, "co": co, "dense_features": ds, "parts": ps, "anchors": a,
            "queries": q, "pool": pool, "source_hashes": src}


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


def _key(data, run, p, fs):
    z = {"v": ver, "feature_contract": fs, "data": _sha(data / "meta.json"), "source_hashes": run["source_hashes"],
         "part": str(p.relative_to(run["dir"])), "pool": run["pool"]["rid"].to_list()}
    return hh.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()


def _features(data, out, run, threads, fs, backend="native", normalizer=None, cache=None):
    if backend in ("hybrid-v2", "hybrid-v3"):
        if backend == "hybrid-v3":
            import gfeat as rich
        else:
            import rfeat as rich
        st = rich.prep(run["pool"], data, "train", normalizer, cache or data.parent)
        make_features = rich.make
        run = {**run, "source_hashes": {**run["source_hashes"], "normalizer": _sha(normalizer),
                                      **{p: _sha(path(__file__).parent / p) for p in
                                         ("gfeat.py", "rfeat.py", "norm2.py", "tfeat.py", "tm_prep.py", "tm_rules.py")}}}
    else:
        st, make_features = prep(run["pool"]), make
    seen, xs, ys, ts, qs = set(), [], [], [], []
    for p in run["parts"]:
        k = _key(data, run, p, fs)
        cp = out / "features" / f"{k}.npz"
        d = _pair(run, p, seen)
        if cp.exists():
            z = np.load(cp, allow_pickle=False)
            x, y, tid, qid = z["x"], z["y"], z["tid"], z["qid"]
        else:
            x, got = make_features(st, run["queries"], d, threads, run["dense_features"])
            if got != fs:
                raise ValueError("unexpected feature names")
            y = d["y"].to_numpy().astype(np.uint8, copy=False)
            tid = d["tid"].to_numpy().astype(np.uint32, copy=False)
            qid = d["qid"].to_numpy().astype(np.uint32, copy=False)
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cp, x=x, y=y, tid=tid, qid=qid)
        if (x.ndim != 2 or x.shape != (len(y), len(fs)) or len(tid) != len(y) or len(qid) != len(y) or
                not np.isfinite(x).all()):
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
    if not len(tid):
        return np.empty(0, dtype=np.intp)
    o = np.lexsort((qid, -prob, tid))
    return o[np.r_[True, tid[o][1:] != tid[o][:-1]]]


def _thresholds(prob):
    u = np.unique(prob)
    if len(u) > 2001:
        u = np.unique(np.quantile(u, np.linspace(0, 1, 2001)))
    return np.r_[u, np.nextafter(1.0, 2.0)]


def evaluate(anchors, qid, tid, y, prob):
    if not len(anchors):
        raise ValueError("evaluation needs reference anchors")
    refs = anchors.sort("rid")
    rid, degree = refs["rid"].to_numpy(), refs["deg"].to_numpy()
    base = float((degree == 0).sum())
    out = {}
    for name, ix in (("plain_threshold", np.arange(len(prob))), ("target_top1_then_threshold", _tops(tid, qid, prob))):
        th = float(np.nextafter(1., 2.))
        if len(ix):
            order = ix[np.lexsort((-prob[ix], qid[ix]))]
            q, p, labels = qid[order], prob[order], y[order]
            starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
            sizes = np.diff(np.r_[starts, len(order)])
            where = np.searchsorted(rid, q)
            inside = where < len(rid)
            where = np.minimum(where, len(rid) - 1)
            inside &= rid[where] == q
            d = degree[where]
            count = np.arange(len(order)) - np.repeat(starts, sizes) + 1
            total = np.cumsum(labels, dtype=np.float64)
            true = total - np.repeat(np.r_[0., total[:-1]][starts], sizes)
            after = np.where(inside & (d > 0), 1.25 * true / (count + .25 * d), 0.)
            before = np.r_[0., after[:-1]]
            before[starts] = (inside[starts] & (d[starts] == 0)).astype(float)
            ranked = np.argsort(-p, kind="stable")
            ends = np.flatnonzero(np.r_[p[ranked][1:] != p[ranked][:-1], True])
            macro = np.r_[base / len(anchors), (base + np.cumsum((after - before)[ranked])[ends]) / len(anchors)]
            precision = np.r_[1., np.cumsum(labels[ranked])[ends] / (ends + 1)]
            thresholds = np.r_[th, p[ranked][ends]]
            th = float(thresholds[np.lexsort((thresholds, precision, macro.round(12)))[-1]])
        keep = np.zeros(len(prob), dtype=bool)
        keep[ix] = prob[ix] >= th
        out[name] = {**_scores(anchors, qid, tid, y, prob, keep), "threshold": th}
    return out


def _blocking_one(q, qid, tid, prob):
    ids = q["rid"].to_numpy().astype(np.uint32, copy=False)
    own = q["own"].to_numpy().astype(np.int64, copy=False)
    oi = np.argsort(ids)
    si = ids[oi]
    ci = np.searchsorted(si, tid)
    if len(qid) != len(tid) or len(tid) != len(prob) or (ci >= len(si)).any() or not np.array_equal(si[ci], tid):
        raise ValueError("invalid blocking candidates")
    counts = np.zeros(len(q), dtype=np.int64)
    ui, ct = np.unique(tid, return_counts=True)
    counts[oi[np.searchsorted(si, ui)]] = ct
    o = np.lexsort((qid, -prob, tid))
    ts = tid[o]
    st = np.r_[0, np.flatnonzero(ts[1:] != ts[:-1]) + 1]
    rk = np.arange(len(o), dtype=np.int64) - np.repeat(st, np.diff(np.r_[st, len(o)])) + 1
    hit = qid[o].astype(np.int64, copy=False) == own[oi[ci[o]]]
    hr = np.zeros(len(q), dtype=np.int64)
    hr[oi[ci[o][hit]]] = rk[hit]
    pos = own >= 0
    den = int(pos.sum())
    ks = (1, 3, 5, 10, 20, 50)
    return {"positive_query_owners": den,
            "candidate_counts": {"queries": len(q), "pairs": len(tid), "mean_per_query": float(counts.mean()) if len(q) else 0.0},
            "mean_retained": {f"top{k}": float(np.minimum(counts, k).mean()) if len(q) else 0.0 for k in ks},
            "actual_true_link_recall": {f"top{k}": 1.0 if not den else float(((hr > 0) & (hr <= k) & pos).sum() / den) for k in ks}}


def blocking(q, qid, tid, prob):
    _need(q, {"rid", "own", "co"})
    if q["rid"].n_unique() != len(q) or not np.isfinite(prob).all():
        raise ValueError("invalid blocking queries")
    z = _blocking_one(q, qid, tid, prob)
    z["scope"] = "candidate-filter recall on selected queries, not official full-pool macro_f05"
    z["per_country"] = {}
    for co in sorted(q["co"].unique().to_list()):
        d = q.filter(pl.col("co") == co)
        keep = np.isin(tid, d["rid"].to_numpy())
        z["per_country"][co] = _blocking_one(d, qid[keep], tid[keep], prob[keep])
    return z


def _fit_lgb(x, y, xv, yv, trees, threads, names=None):
    import lightgbm as lgb

    m = lgb.LGBMClassifier(n_estimators=trees, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=42, n_jobs=threads, verbosity=-1)
    m.fit(x, y, eval_set=[(xv, yv)], eval_metric="binary_logloss", feature_name=names or "auto",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def _fit_cat(x, y, xv, yv, trees, threads):
    from catboost import CatBoostClassifier

    m = CatBoostClassifier(iterations=trees, learning_rate=0.05, depth=6, loss_function="Logloss",
                            eval_metric="Logloss", random_seed=42, thread_count=threads,
                            allow_writing_files=False, verbose=False)
    m.fit(x, y, eval_set=(xv, yv), early_stopping_rounds=50, verbose=False)
    return m


def predict(m, x, threads=None):
    if not hasattr(m, "predict_proba"):
        import lightgbm as lgb
        if not isinstance(m, lgb.Booster) or (threads is not None and threads < 1):
            raise ValueError("invalid lightgbm booster prediction")
        return np.asarray(m.predict(x, num_threads=threads or 1), dtype=np.float64)
    kw = {}
    if threads is not None:
        if threads < 1:
            raise ValueError("threads must be positive")
        if hasattr(m, "get_cat_feature_indices"):
            kw["thread_count"] = threads
        elif hasattr(m, "booster_"):
            kw["num_threads"] = threads
    return np.asarray(m.predict_proba(x, **kw)[:, 1], dtype=np.float64)


def _contract(met):
    if met.get("feature_backend") == "hybrid-v3":
        import gfeat
        return gfeat.contract(met)
    if met.get("feature_backend") == "hybrid-v2":
        import rfeat
        return rfeat.contract(met)
    if met.get("feature_backend") == "teammate-v1-nos1":
        try:
            import tfeat
        except ImportError:
            from src import tfeat
        return tfeat.contract(met)
    if met.get("feature_backend") not in (None, "native"):
        raise ValueError("unknown feature backend")
    names = met.get("feature_names")
    ds = met.get("dense_features")
    if ds is None:
        ds = []
    if (not isinstance(names, list) or not isinstance(ds, list) or names[:len(ff)] != ff or names[len(ff):] != ds or
            len(set(ds)) != len(ds) or any(not isinstance(x, str) or x not in df for x in ds)):
        raise ValueError("model feature order mismatch")
    return names, ds


def _width(m):
    if callable(getattr(m, "num_feature", None)):
        return int(m.num_feature())
    v = getattr(m, "feature_names_", None)
    if isinstance(v, (list, tuple)) and v:
        return len(v)
    for n in ("n_features_in_", "feature_count_"):
        v = getattr(m, n, None)
        if isinstance(v, (int, np.integer)):
            return int(v)
    raise ValueError("model feature width unavailable")


def load_models(out):
    out = path(out)
    met = _json(out / "metadata.json")
    names, _ = _contract(met)
    z = {}
    for name in met["models"]:
        p = out / met["model_files"][name]
        if name == "lgb":
            if met.get("feature_backend") in ("teammate-v1-nos1", "hybrid-v2", "hybrid-v3"):
                import lightgbm as lgb
                z[name] = lgb.Booster(model_file=str(p))
                if z[name].feature_name() != names:
                    raise ValueError("lightgbm feature names mismatch")
            else:
                z[name] = jl.load(p)
        elif name == "cat":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(p)
            z[name] = m
        else:
            raise ValueError(f"unknown model {name}")
        if _width(z[name]) != len(names):
            raise ValueError(f"model feature width mismatch {name}")
    return z, met


def train(data, trs, vals, out, model="lgb", threads=8, trees=800, backend="native", normalizer=None, cache=None):
    if threads < 1 or trees < 1:
        raise ValueError("positive threads and trees required")
    data, out = path(data).resolve(), path(out).resolve()
    if backend not in ("native", "hybrid-v2", "hybrid-v3"):
        raise ValueError("unknown training feature backend")
    if backend in ("hybrid-v2", "hybrid-v3"):
        if backend == "hybrid-v3":
            import gfeat as rich
        else:
            import rfeat as rich
        if normalizer is None or model != "lgb" or (out / "metadata.json").exists():
            raise ValueError("rich training needs a normalization model, lightgbm and a new output")
        normalizer = path(normalizer).resolve()
        out.mkdir(parents=True, exist_ok=True)
        if normalizer != out / "normalizer.json":
            sh.copyfile(normalizer, out / "normalizer.json")
        normalizer = out / "normalizer.json"
    if not (data / "meta.json").is_file():
        raise ValueError(f"missing prepared data {data}")
    if not trs or not vals:
        raise ValueError("training and validation runs required")
    tr = [_run(data, p, 2) for p in trs]
    va = [_run(data, p, 0) for p in vals]
    ds = tr[0]["dense_features"]
    if any(r["dense_features"] != ds for r in tr + va):
        raise ValueError("mixed dense feature schemas")
    fs = rich.names(ds) if backend in ("hybrid-v2", "hybrid-v3") else list(ff) + ds
    svs = {r["met"].get("score_version", 1) for r in tr + va}
    if len(svs) != 1:
        raise ValueError("mixed candidate score versions")
    sv = svs.pop()
    if len({r["dir"] for r in tr + va}) != len(tr) + len(va):
        raise ValueError("run cannot be both training and validation")
    xa = [_features(data, out, r, threads, fs, backend, normalizer, cache) for r in tr]
    xv = [_features(data, out, r, threads, fs, backend, normalizer, cache) for r in va]
    x, y = np.concatenate([z[0] for z in xa]), np.concatenate([z[1] for z in xa])
    vx, vy = np.concatenate([z[0] for z in xv]), np.concatenate([z[1] for z in xv])
    tid, qid = np.concatenate([z[2] for z in xv]), np.concatenate([z[3] for z in xv])
    anchors = pl.concat([r["anchors"].select("rid", "deg") for r in va])
    vtid = np.concatenate([r["queries"]["rid"].to_numpy() for r in va])
    vq = pl.concat([r["queries"].select("rid", "own", "co") for r in va])
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
        m = _fit_lgb(x, y, vx, vy, trees, threads, fs) if name == "lgb" else fitters[name](x, y, vx, vy, trees, threads)
        prob = predict(m, vx)
        met = evaluate(anchors, qid, tid, vy, prob)
        met["blocking_filter"] = blocking(vq, qid, tid, prob)
        res[name] = met
        raw.append(pl.DataFrame({"model": [name] * len(prob), "tid": tid, "qid": qid, "y": vy, "prob": prob,
                                 "plain_selected": prob >= met["plain_threshold"]["threshold"],
                                 "top1_selected": np.isin(np.arange(len(prob)), _tops(tid, qid, prob)) &
                                 (prob >= met["target_top1_then_threshold"]["threshold"])}))
        for dec, key in (("plain_threshold", "plain_selected"), ("target_top1_then_threshold", "top1_selected")):
            keep = raw[-1][key].to_numpy()
            per.append(anchor_f05(anchors, qid, vy, keep).with_columns(pl.lit(name).alias("model"), pl.lit(dec).alias("decoder")))
        if name == "lgb" and backend in ("hybrid-v2", "hybrid-v3"):
            p = "lgb.txt"
            m.booster_.save_model(out / p)
        elif name == "lgb":
            p = "lgb.joblib"
            jl.dump(m, out / p)
        else:
            p = "cat.cbm"
            m.save_model(out / p)
        files[name] = p
    pl.concat(raw).write_parquet(out / "validation_predictions.parquet", compression="zstd")
    pl.concat(per).write_parquet(out / "validation_anchors.parquet", compression="zstd")
    met = {"version": ver, "seed": 42, "score_version": sv, "feature_names": fs, "dense_features": ds, "models": names, "model_files": files,
           "configuration": {"data": str(data), "data_meta_sha256": _sha(data / "meta.json"), "threads": threads, "trees": trees, "class_weights": None,
                             "augmentation": False, "train_runs": [str(r["dir"]) for r in tr],
                             "validation_runs": [str(r["dir"]) for r in va]},
           "validation": {"label": "sampled-query estimate, not official/full-pool validation",
                          "warning": "validation queries contain selected anchor aliases and sampled orphans, not every target in the full corpus",
                          "anchors": len(anchors), "queries": sum(len(r["queries"]) for r in va), "models": res}}
    if backend in ("hybrid-v2", "hybrid-v3"):
        met["feature_backend"] = backend
        met["normalizer"] = {"file": "normalizer.json", "sha256": _sha(normalizer)}
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
        assert got["feature_names"] == ff and got["dense_features"] == [] and set(ms) == {"lgb", "cat"}
        assert predict(ms["lgb"], np.zeros((1, len(ff)), np.float32)).shape == (1,)
        assert predict(ms["cat"], np.zeros((1, len(ff)), np.float32)).shape == (1,)
        for m in ms.values():
            x = np.zeros((4, len(ff)), np.float32)
            assert np.allclose(predict(m, x, 1), predict(m, x, 2), rtol=0, atol=1e-12)
        old = dict(got)
        old.pop("dense_features")
        _save_json(out / "metadata.json", old)
        assert set(load_models(out)[0]) == {"lgb", "cat"}
        _save_json(out / "metadata.json", got)
        bad = dict(got)
        bad["feature_names"] = ff + ["qid"]
        bad["dense_features"] = ["qid"]
        _save_json(out / "metadata.json", bad)
        try:
            load_models(out)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe model feature accepted")
        _save_json(out / "metadata.json", got)
        rr = _run(data, runs / "tr", 2)
        assert _key(data, rr, rr["parts"][0], ff) != _key(data, rr, rr["parts"][0], ff + ["ds_e5"])
        z = evaluate(_frame([[20, 1]], ["rid", "deg"]), np.array([20, 10]), np.array([200, 201]),
                      np.array([1, 0], np.uint8), np.array([0.9, 0.1]))
        assert z["plain_threshold"]["macro_f05"] == 1.0
        assert z["target_top1_then_threshold"]["macro_f05"] == 1.0
        bq = _frame([[1, 1, "us"], [2, -1, "us"], [3, 9, "france"], [4, 4, "france"]], ["rid", "own", "co"])
        bz = blocking(bq, np.array([2, 1, 3, 8]), np.array([1, 1, 3, 4]), np.array([.8, .8, .9, .1]))
        assert bz["positive_query_owners"] == 3 and bz["actual_true_link_recall"]["top1"] == 1 / 3
        assert bz["mean_retained"]["top1"] == .75 and bz["per_country"]["us"]["actual_true_link_recall"]["top1"] == 1.0
        z0 = blocking(bq.filter(pl.col("own") < 0), np.empty(0, np.uint32), np.empty(0, np.uint32), np.empty(0))
        assert z0["positive_query_owners"] == 0 and z0["actual_true_link_recall"]["top50"] == 1.0
        good = _json(runs / "va" / "metrics.json")
        bad = dict(good)
        bad["dense_features"] = ["ds_e5"]
        _save_json(runs / "va" / "metrics.json", bad)
        try:
            train(data, [runs / "tr"], [runs / "va"], out, trees=10, threads=1)
        except ValueError:
            pass
        else:
            raise AssertionError("mixed dense schemas accepted")
        _save_json(runs / "va" / "metrics.json", good)
        bad = dict(good)
        bad["fold"] = 1
        _save_json(runs / "va" / "metrics.json", bad)
        try:
            train(data, [runs / "tr"], [runs / "va"], out, trees=10, threads=1)
        except ValueError:
            pass
        else:
            raise AssertionError("locked fold accepted")
        assert f05(_frame([[1, 1]], ["rid", "deg"]), np.array([1]), np.array([1]), np.array([True])) == 1.0
    a = pl.DataFrame({"rid": [0, 1, 2], "deg": [3, 2, 0]})
    q, t = np.repeat(np.arange(4), 6), np.tile(np.arange(6), 4)
    y = (q == np.array([0, 1, 0, -1, 0, 1])[t]).astype(np.uint8)
    rng = np.random.default_rng(17)
    for _ in range(8):
        p = rng.choice([.1, .3, .5, .8, .95], len(q))
        fast = evaluate(a, q, t, y, p)
        for name, ix in (("plain_threshold", np.arange(len(p))), ("target_top1_then_threshold", _tops(t, q, p))):
            candidates = []
            for threshold in _thresholds(p):
                keep = np.zeros(len(p), bool)
                keep[ix] = p[ix] >= threshold
                candidates.append({**_scores(a, q, t, y, p, keep), "threshold": float(threshold)})
            best = max(candidates, key=lambda z: (round(z["macro_f05"], 12), z["pair_precision"], z["threshold"]))
            assert fast[name] == best
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
    pa.add_argument("--backend", choices=("native", "hybrid-v2", "hybrid-v3"), default="native")
    pa.add_argument("--normalizer", type=path)
    pa.add_argument("--cache", type=path, default=root / "cache")
    pa.add_argument("--check", action="store_true")
    a = pa.parse_args()
    if a.check:
        check()
    else:
        train(a.data, a.train, a.val, a.out, a.model, a.threads, a.trees, a.backend, a.normalizer, a.cache)


if __name__ == "__main__":
    main()
