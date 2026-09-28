"""out-of-sample pairwise stack with raw-text and full-corpus evidence"""
import argparse as ap
import hashlib as hh
import json
import shutil as sh
import time
from pathlib import Path as path

import gfeat
import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import infer
import post
import rfeat
import tfeat


features = list(tfeat.ff[13:]) + rfeat.extra + ["gate_logit", "neural_logit"]


def active():
    return pl.max_horizontal("gate_prob", "neural_prob") >= .01


def sample_targets(ids):
    x = np.asarray(ids, dtype=np.uint64) + np.uint64(19)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return (x ^ (x >> np.uint64(31))) % 20 == 0


def raw_targets(data, split):
    d = pl.concat([pl.read_parquet(data / split / f"s{i}.parquet", columns=["rid", "nm", "ad", "co"]) for i in (2, 3)])
    if not np.array_equal(d["rid"].to_numpy(), np.arange(len(d), dtype=np.uint32)):
        raise ValueError("target ids are not contiguous")
    return d


def feature_names(name_change=False, neural_columns=()):
    import re
    if type(name_change) is not bool or len(set(neural_columns)) != len(neural_columns) or any(
            not isinstance(c, str) or not re.fullmatch(r"np_[a-z][a-z0-9_]*", c) for c in neural_columns):
        raise ValueError("invalid stack feature configuration")
    return features[:-2] + (gfeat.ff if name_change else []) + features[-2:] + ["logit_" + c for c in neural_columns]


def matrix(frame, state, targets, threads, name_change=False, neural_columns=()):
    fs = feature_names(name_change, neural_columns)
    maker = gfeat if name_change else rfeat
    parts = []
    for d in frame.iter_slices(25000):
        ids = d["tid"].unique().cast(pl.UInt32)
        queries = targets.select(pl.all().gather(ids))
        pairs = d.select("qid", "tid").with_columns(pl.lit(0., pl.Float32).alias("ns"), pl.lit(0., pl.Float32).alias("ads"))
        x, _ = maker.make(state, queries, pairs, threads)
        # cached scores lack retrieval features; discard every placeholder-derived column
        parts.append(np.column_stack((x[:, 13:], post.logit(d["gate_prob"].to_numpy()),
                                      post.logit(d["neural_prob"].to_numpy()),
                                      *[post.logit(d[c].to_numpy()) for c in neural_columns])).astype(np.float32))
    return np.concatenate(parts) if parts else np.empty((0, len(fs)), np.float32)


def fit_params(threads, file=None):
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": .05, "num_leaves": 31,
              "max_depth": 6, "min_data_in_leaf": 100, "lambda_l2": 10., "bagging_fraction": .8,
              "bagging_freq": 1, "feature_fraction": .9, "seed": 19, "num_threads": threads, "verbosity": -1}
    if file is not None:
        selected = infer._json(file)
        if "best" in selected:
            selected = selected["best"]["details"]["training_params"]
        else:
            selected = selected.get("params", selected)
        allowed = set(params) | {"deterministic", "force_col_wise"}
        if not isinstance(selected, dict) or not selected or set(selected) - allowed:
            raise ValueError("invalid or unsupported stack parameters")
        if selected.get("objective", "binary") != "binary":
            raise ValueError("stack parameters require a binary objective")
        params.update(selected)
        params["num_threads"] = threads
    return params


def fit(prepared, normalizer, out, cache, threads=8, trees=400, cache_only=False, parameters=None):
    params = fit_params(threads, parameters)
    prepared, out, cache = path(prepared), path(out), path(cache)
    source = post.verified(prepared, "train")
    config = source.get("config") or infer._json(path(source["runs"][0]["path"]) / "manifest.json")["config"]
    name_change = config.get("gate", {}).get("feature_backend") == "hybrid-v3"
    neural_columns = config.get("neural", {}).get("score_columns", [])
    fs = feature_names(name_change, neural_columns)
    maker = gfeat if name_change else rfeat
    if source.get("stack_model") or out.exists():
        raise ValueError("stack fitting needs base scores and a new output")
    data = path(source["data"])
    refs = pl.read_parquet(data / "train/ref.parquet")
    folds = refs["fold"].to_numpy()
    fit_refs = refs.filter((pl.col("fold") == 0) & pl.Series(post.partition(refs["rid"].to_numpy(), 3) == 0)).select(
        pl.col("rid").alias("qid"))
    frame = pl.scan_parquet(prepared).join(fit_refs.lazy(), on="qid", how="inner").filter(active()).collect(engine="streaming")
    own, tid = frame["own"].to_numpy(), frame["tid"].to_numpy()
    groups = np.where(own >= 0, own, tid.astype(np.int64) + len(refs))
    owner_fold = folds[np.maximum(own, 0)]
    eligible = ((own >= 0) & (owner_fold == 2)) | ((post.partition(groups, 3) == 0) & ((own < 0) | (owner_fold == 0)))
    hard = np.maximum(frame["gate_prob"].to_numpy(), frame["neural_prob"].to_numpy()) >= .1
    label = frame["y"].to_numpy()
    sample = sample_targets(tid)
    take = eligible & ((label == 1) | hard | sample)
    frame = frame.filter(pl.Series(take)).sort("qid", "tid")
    # align weights after the deterministic pair sort
    frame = frame.with_columns(pl.when((pl.col("y") == 0) & (pl.max_horizontal("gate_prob", "neural_prob") < .1))
                               .then(20.).otherwise(1.).alias("weight"))
    if frame["y"].n_unique() != 2:
        raise ValueError("stack needs positive and negative fitting pairs")
    out.mkdir(parents=True)
    sh.copyfile(normalizer, out / "normalizer.json")
    state = maker.prep(refs, data, "train", out / "normalizer.json", cache)
    targets = raw_targets(data, "train")
    x = matrix(frame, state, targets, threads, name_change, neural_columns)
    del state, targets
    y, weight = frame["y"].to_numpy(), frame["weight"].to_numpy()
    own = frame["own"].to_numpy()
    groups = np.where(own >= 0, own, frame["tid"].to_numpy().astype(np.int64) + len(refs))
    inner = post.partition(frame["qid"].to_numpy(), 5) == 0
    same_group = inner == (post.partition(groups, 5) == 0)
    train_mask, val_mask = ~inner & same_group, inner & same_group
    if set(np.unique(y[train_mask])) != {0, 1} or set(np.unique(y[val_mask])) != {0, 1}:
        raise ValueError("inner stack split has insufficient labels")
    np.savez_compressed(out / "fit.npz", x=x, y=y, weight=weight, train=train_mask, valid=val_mask)
    cache_info = {"version": 1, "kind": "stack-fit-cache", "features": fs, "name_change": name_change, "neural_columns": neural_columns,
                  "source_config_sha256": source["config_sha256"],
                  "data_meta_sha256": source["data_meta_sha256"], "source_scores_sha256": source["score_sha256"],
                  "fit_pairs": len(frame), "fit_positive": int(y.sum()), "fit_sha256": infer._sha(out / "fit.npz"),
                  "normalizer_sha256": infer._sha(out / "normalizer.json")}
    infer._write(out / "cache.json", cache_info)
    if cache_only:
        return cache_info
    start = time.monotonic()
    train_set = lgb.Dataset(x[train_mask], label=y[train_mask], weight=weight[train_mask], feature_name=fs)
    valid_set = lgb.Dataset(x[val_mask], label=y[val_mask], weight=weight[val_mask], feature_name=fs, reference=train_set)
    model = lgb.train(params, train_set, num_boost_round=trees, valid_sets=[valid_set], callbacks=[lgb.early_stopping(30, verbose=False)])
    rounds = model.best_iteration or trees
    model = lgb.train(params, lgb.Dataset(x, label=y, weight=weight, feature_name=fs), num_boost_round=rounds)
    model.save_model(out / "lgb.txt")
    metadata = {"version": 1, "kind": "postgate-pairwise-rich-stack", "features": fs,
                 "name_change": name_change, "neural_columns": neural_columns,
                "source_config_sha256": source["config_sha256"], "data_meta_sha256": source["data_meta_sha256"],
                "source_scores_sha256": source["score_sha256"], "fit_pairs": len(frame), "fit_positive": int(y.sum()),
                "fit_partition": {"modulus": 3, "remainder": 0}, "calibration_partition": {"modulus": 3, "remainder": 1},
                "validation_partition": {"modulus": 3, "remainder": 2}, "base_development_reuse": "fold0 used in prior base-model development",
                "inner_split": "owner/reference disjoint 1-in-5 groups inside fitting partition; refit fitting partition after early stopping",
                "fit_targets": "fold2 targets, fold0 partition0 targets, partition0 orphans; calibration/validation owners excluded",
                "selection": "max(gate,neural)>=.01; all positives and hard negatives; low-score negatives target-sampled at 1/20 with weight20",
                 "rounds": rounds, "params": params, "fit_seconds": time.monotonic() - start,
                 "parameters_source_sha256": infer._sha(parameters) if parameters is not None else None,
                "files": {name: infer._sha(out / name) for name in ("lgb.txt", "normalizer.json")}}
    infer._write(out / "metadata.json", metadata)
    return metadata


def bundle(root):
    root = path(root)
    m = infer._json(root / "metadata.json")
    if (m.get("version") != 1 or m.get("kind") != "postgate-pairwise-rich-stack" or
            m.get("features") != feature_names(m.get("name_change", False), m.get("neural_columns", [])) or
            m.get("fit_partition") != {"modulus": 3, "remainder": 0} or set(m.get("files", {})) != {"lgb.txt", "normalizer.json"}):
        raise ValueError("invalid pairwise stack model")
    for name, sha in m["files"].items():
        if path(name).name != name or infer._sha(root / name) != sha:
            raise ValueError("pairwise stack files changed")
    return m, infer._sha(root / "metadata.json")


def score(prepared, model_dir, out, cache, threads=8):
    prepared, model_dir, out = path(prepared), path(model_dir), path(out)
    source = post.verified(prepared)
    meta, digest = bundle(model_dir)
    if (source.get("stack_model") or source["config_sha256"] != meta["source_config_sha256"] or
            source["data_meta_sha256"] != meta["data_meta_sha256"] or out.exists()):
        raise ValueError("stack input configuration mismatch or existing output")
    data, split = path(source["data"]), source["split"]
    refs = pl.read_parquet(data / split / "ref.parquet")
    name_change, neural_columns = meta.get("name_change", False), meta.get("neural_columns", [])
    maker = gfeat if name_change else rfeat
    state = maker.prep(refs, data, split, model_dir / "normalizer.json", cache)
    targets = raw_targets(data, split)
    model = lgb.Booster(model_file=str(model_dir / "lgb.txt"))
    if model.feature_name() != meta["features"]:
        raise ValueError("stack feature order mismatch")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    writer = None
    start, rows, scored = time.monotonic(), 0, 0
    try:
        for batch in pq.ParquetFile(prepared).iter_batches(batch_size=100000):
            frame = pl.from_arrow(batch)
            mask = frame.select(active()).to_series().to_numpy()
            probs = frame["prob"].to_numpy().copy()
            if mask.any():
                x = matrix(frame.filter(pl.Series(mask)), state, targets, threads, name_change, neural_columns)
                probs[mask] = model.predict(x, num_threads=threads)
            table = frame.with_columns(pl.Series("stack_prob", probs.astype(np.float32))).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
            scored += int(mask.sum())
            if rows % 1000000 == 0:
                print({"rows": rows, "rescored": scored, "seconds": round(time.monotonic() - start, 1)}, flush=True)
    finally:
        if writer is not None:
            writer.close()
    if rows != source["pairs"]:
        raise ValueError("stack score coverage changed")
    tmp.replace(out)
    config = source.get("config") or infer._json(path(source["runs"][0]["path"]) / "manifest.json")["config"]
    config = {**config, "stack": {"sha256": digest, "kind": meta["kind"]}}
    config["model"] = {"sha256": hh.sha256(json.dumps({"base": config["model"]["sha256"], "stack": digest}, sort_keys=True).encode()).hexdigest()}
    source = {**source, "config": config, "config_sha256": hh.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
              "parent_scores_sha256": source["score_sha256"], "score_sha256": infer._sha(out),
              "stack_model": {"sha256": digest, "kind": meta["kind"]}, "calibration_partition": meta["calibration_partition"],
              "validation_partition": meta["validation_partition"], "rescored_pairs": scored, "rescore_seconds": time.monotonic() - start}
    if split == "train":
        fit_refs = refs.filter((pl.col("fold") == 0) & pl.Series(post.partition(refs["rid"].to_numpy(), 3) == 1))
        source["fit_reference_counts"] = dict(fit_refs.group_by("co").len().iter_rows())
    infer._write(out.with_suffix(".json"), source)
    return source


def check():
    selected = path(__file__).resolve().parents[1] / "reports/optuna-search.json"
    expected = infer._json(selected)["best"]["details"]["training_params"]
    assert fit_params(12, selected) == {**expected, "num_threads": 12}
    assert fit_params(12)["num_leaves"] == 31
    q = np.arange(1000)
    a, b, c = (set(q[post.partition(q, 3) == i]) for i in range(3))
    assert a and b and c and not (a & b or b & c or a & c) and len(a | b | c) == len(q)
    assert not set(features) & set(tfeat.ff[:13]) and len(features) == len(set(features))
    assert .04 < sample_targets(np.arange(10000)).mean() < .06
    refs = pl.DataFrame({"rid": [0, 1], "nm": ["alpha", "alpha"], "ad": ["27 rue duc", "29 rue duc"], "co": ["france"] * 2})
    targets = pl.DataFrame({"rid": [0, 1], "nm": ["alpha fils", "alpha"], "ad": ["27 rue duc", "29 rue duc"], "co": ["france"] * 2})
    state = {"base": tfeat.prep(refs), "mapping": {}, "words": {}, "ref_counts": np.array([[2, 2], [2, 2]]),
             "target_counts": np.array([[2], [2]])}
    rows = pl.DataFrame({"qid": [0, 1, 0, 1], "tid": [0, 0, 1, 1], "gate_prob": [.9, .1, .3, .9], "neural_prob": [.95, .2, .4, .95]})
    whole = matrix(rows, state, targets, 1)
    chunks = np.concatenate([matrix(rows.slice(i, 1), state, targets, 1) for i in range(len(rows))])
    np.testing.assert_array_equal(whole, chunks)
    print("pairwise stack checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("fit", "score", "cache", "check"))
    p.add_argument("--scores", type=path)
    p.add_argument("--normalizer", type=path, default=path("artifacts/norm2.json"))
    p.add_argument("--model", type=path, default=path("artifacts/stack2"))
    p.add_argument("--out", type=path)
    p.add_argument("--cache", type=path, default=path("cache"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--parameters", type=path)
    a = p.parse_args()
    if a.command == "check":
        check()
    elif a.command in ("fit", "cache") and a.scores:
        print(json.dumps(fit(a.scores, a.normalizer, a.model, a.cache, a.threads, trees=a.trees,
                             cache_only=a.command == "cache", parameters=a.parameters), indent=2))
    elif a.command == "score" and a.scores and a.out:
        print(json.dumps(score(a.scores, a.model, a.out, a.cache, a.threads), indent=2))
    else:
        p.error("scores are required; scoring also requires out")
