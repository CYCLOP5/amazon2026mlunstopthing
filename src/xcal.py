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
    p.add_argument("command", choices=("run", "check"))
    p.add_argument("--cache", type=path, default=path("artifacts/full-result/tuning-cache"))
    p.add_argument("--model", type=path, default=path("artifacts/full-result/stack"))
    p.add_argument("--data", type=path, default=path("cache/data"))
    p.add_argument("--out", type=path, default=path("reports/learned-country-transfer.json"))
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    if a.command == "check":
        check()
    else:
        run(a.cache, a.model, a.data, a.out, a.threads)
