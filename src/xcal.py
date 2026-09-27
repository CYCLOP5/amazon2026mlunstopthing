"""cached country-transfer calibration diagnostics; no france labels"""
import argparse as ap
import concurrent.futures as cf
from pathlib import Path as path

import lightgbm as lgb
import numpy as np
import polars as pl

import decode
import infer
import opt
import post
import post_eval
import stack2


def evaluate(e, p, curves, anchors, threads=1):
    calibrated = post.adjust(p, e["country"], e["segment"], {"curves": curves})
    top = decode.winners(e["qid"], e["tid"], calibrated, p)
    top = top[e["local"][top] >= 0]
    top = top[anchors[e["local"][top]]]
    keep, details = decode.choose(e["local"][top], e["tid"][top], calibrated[top], p[top], threads=threads)
    return {**post_eval.metric(e["local"][top], e["y"][top], keep, e["degree"], anchors), **details}


def run(cache, model, data, out, threads=8):
    meta, _, e = opt.load(cache)
    model_meta, _ = stack2.bundle(model)
    if threads < 1 or any(model_meta[k] != meta["fit"][k] for k in ("source_config_sha256", "source_scores_sha256", "data_meta_sha256")):
        raise ValueError("diagnostic model/cache mismatch or invalid threads")
    refs = pl.read_parquet(path(data) / "train/ref.parquet", columns=["rid", "fold", "co", "deg"])
    refs = refs.filter((pl.col("fold") == 0) & pl.Series(post.partition(refs["rid"].to_numpy(), 3) == 1))
    if not np.array_equal(refs["deg"].to_numpy(), e["degree"]):
        raise ValueError("reference alignment changed")
    if meta["fit"]["data_meta_sha256"] != infer._sha(path(data) / "meta.json"):
        raise ValueError("diagnostic dataset changed")
    names, active = meta["fit"]["features"], e["active"]
    booster = lgb.Booster(model_file=str(path(model) / "lgb.txt"))
    if booster.feature_name() != names:
        raise ValueError("diagnostic model feature order changed")
    probabilities = {}
    for name, column in (("gate", "gate_logit"), ("neural", "neural_logit")):
        p = e["raw"].copy()
        p[active] = 1 / (1 + np.exp(-e["x"][:, names.index(column)]))
        probabilities[name] = p
    p = e["raw"].copy()
    p[active] = booster.predict(e["x"], num_threads=threads)
    probabilities["stack"] = p
    for weight in (.25, .5, .75):
        logits = weight * post.logit(p) + (1 - weight) * post.logit(probabilities["gate"])
        probabilities[f"stack{weight}"] = (1 / (1 + np.exp(-logits))).astype(np.float32)
    if all(f"logit_np_m{i}" in names for i in range(6)):
        p = e["raw"].copy()
        columns = [names.index(f"logit_np_m{i}") for i in range(6)]
        p[active] = 1 / (1 + np.exp(-e["x"][:, columns].mean(axis=1)))
        probabilities["first6_mean"] = p
    selected = e["local"] >= 0
    countries = meta["countries"]
    edges = np.linspace(post.logit([.02])[0], post.logit([.999])[0], 25)

    def probe(item):
        name, p = item
        bins = np.searchsorted(edges, post.logit(p), side="right")
        hist = {}
        for country in countries:
            for segment in range(3):
                mask = selected & (e["country"] == country) & (e["segment"] == segment)
                hist[f"{country}|{segment}"] = [np.bincount(bins[mask], minlength=26).astype(float),
                                                np.bincount(bins[mask], weights=e["y"][mask], minlength=26)]
        results = []
        for target in countries:
            sources = {c: n for c, n in countries.items() if c != target}
            training = {k: v for k, v in hist.items() if k.rsplit("|", 1)[0] in sources}
            anchors = refs["co"].to_numpy() == target
            for mode in ("density", "unscaled_density", "source_empirical"):
                if mode == "source_empirical":
                    curves = post.pooled(training, sources, hist, edges)
                else:
                    curves = post.curves(training, hist, sources, countries, edges, transfer=mode == "density")
                result = {"policy": name, "target": target, "calibration": mode,
                          "fit_countries": list(sources), "references": int(anchors.sum()),
                          **evaluate(e, p, curves, anchors, max(1, threads // 2))}
                print(result, flush=True)
                results.append(result)
        return results

    with cf.ThreadPoolExecutor(max_workers=min(2, threads)) as pool:
        results = [row for rows in pool.map(probe, probabilities.items()) for row in rows]
    result = {"scope": "partition1 country-transfer calibration proxy with complete target competition",
              "caveat": "matcher was trained on both labelled countries; this is not unseen-country training or france accuracy",
              "cache_sha256": infer._sha(path(cache) / "cache.json"),
              "model_sha256": infer._sha(path(model) / "metadata.json"), "results": results}
    infer._write(out, result)
    return result


def compare(cache, development, models, out, threads=8):
    fit_meta, _, fitting = opt.load(cache, search_only=True)
    dev_meta, _, e = opt.load(development)
    if dev_meta.get("purpose") != "development" or dev_meta.get("reference_partition") != {"modulus": 3, "remainder": 2}:
        raise ValueError("finalist comparison requires the reserved development cache")
    keys = ("source_config_sha256", "source_scores_sha256", "data_meta_sha256", "features")
    if any(fit_meta["fit"][k] != dev_meta["fit"][k] for k in keys):
        raise ValueError("calibration and development caches differ")
    selected = fitting["local"] >= 0
    edges = np.linspace(post.logit([.02])[0], post.logit([.999])[0], 25)
    rows = []
    for root in models:
        meta, digest = stack2.bundle(root)
        if any(meta[k] != fit_meta["fit"][k] for k in keys) or meta["files"]["normalizer.json"] != fit_meta["files"]["fit/normalizer.json"]:
            raise ValueError("finalist model does not match the feature caches")
        model = lgb.Booster(model_file=str(path(root) / "lgb.txt"))
        if model.feature_name() != meta["features"]:
            raise ValueError("finalist feature order changed")
        calibration = fitting["raw"].copy()
        calibration[fitting["active"]] = model.predict(fitting["x"], num_threads=threads)
        bins = np.searchsorted(edges, post.logit(calibration), side="right")
        hist = {}
        for co in fit_meta["countries"]:
            for seg in range(3):
                mask = selected & (fitting["country"] == co) & (fitting["segment"] == seg)
                hist[f"{co}|{seg}"] = [np.bincount(bins[mask], minlength=26).astype(float),
                                       np.bincount(bins[mask], weights=fitting["y"][mask], minlength=26)]
        curves = post.curves(hist, hist, fit_meta["countries"], fit_meta["countries"], edges, transfer=False)
        p = e["raw"].copy()
        p[e["active"]] = model.predict(e["x"], num_threads=threads)
        row = {"model": str(root), "model_sha256": digest,
               **evaluate(e, p, curves, np.ones(len(e["degree"]), bool), threads)}
        rows.append(row)
        print(row, flush=True)
    result = {"scope": "partition2 full-incidence development; calibration fitted on partition1 only",
              "development_cache_sha256": infer._sha(path(development) / "cache.json"),
              "calibration_cache_sha256": infer._sha(path(cache) / "cache.json"),
              "references": len(e["degree"]), "results": rows}
    infer._write(out, result)
    return result


def check():
    e = {"qid": np.array([0, 1, 2]), "tid": np.array([0, 1, 1]), "local": np.array([0, 1, -1]),
         "y": np.array([1, 1, 0]), "country": np.array(["india", "us", "us"]),
         "segment": np.zeros(3, np.uint8), "degree": np.array([1, 2])}
    curves = {f"{co}|0": {"centres": [-20., 20.], "posterior": [1., 1.]} for co in ("india", "us")}
    anchors = np.array([False, True])
    assert evaluate(e, np.array([.9, .9, .99]), curves, anchors)["macro_f05"] == 0.
    assert evaluate(e, np.array([.9, .9, .1]), curves, anchors)["macro_f05"] == 1.25 / 1.5
    edges = np.array([-2., 0., 2.])
    hist = {f"india|{i}": [np.array([10., 20., 30., 40.]), np.array([0., 5., 20., 40.])] for i in range(3)}
    target = {f"france|{i}": [np.array([100., 20., 50., 1.]), None] for i in range(3)}
    source = post.curves(hist, hist, {"india": 10}, {"india": 10}, edges, transfer=False)
    a = post.pooled(hist, {"india": 10}, target, edges)
    b = post.pooled(hist, {"india": 10}, {k: [v[0] * 7, None] for k, v in target.items()}, edges)
    for i in range(3):
        key = f"france|{i}"
        assert a[key]["posterior"] == b[key]["posterior"] == source[f"india|{i}"]["posterior"]
        assert a[key]["positive_scale"] == 1. and b[key]["test_pairs"] == a[key]["test_pairs"] * 7
    print("country-transfer full-competition checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "compare", "check"))
    p.add_argument("--cache", type=path, default=path("artifacts/full-result/tuning-cache"))
    p.add_argument("--model", type=path, default=path("artifacts/full-result/stack"))
    p.add_argument("--data", type=path, default=path("cache/data"))
    p.add_argument("--out", type=path)
    p.add_argument("--development", type=path, default=path("artifacts/retune-r2/development-cache"))
    p.add_argument("--models", type=path, nargs="+")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    if a.command == "check":
        check()
    elif a.command == "compare":
        compare(a.cache, a.development, a.models or [a.model], a.out or path("reports/learned-finalists.json"), a.threads)
    else:
        run(a.cache, a.model, a.data, a.out or path("reports/learned-country-transfer.json"), a.threads)
