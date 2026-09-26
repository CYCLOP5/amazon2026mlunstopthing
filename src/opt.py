"""parallel cached search on incidence-complete source1 macro f0.5"""
import argparse as ap
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import shutil as sh
import tempfile as tf
import time
from pathlib import Path as path

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

import decode
import infer
import post
import post_eval
import rfeat
import stack2


def prepare(prepared, root, cache, threads=8):
    prepared, root, cache = path(prepared), path(root), path(cache)
    source = post.verified(prepared, "train")
    fit = infer._json(root / "fit/cache.json")
    if fit["source_scores_sha256"] != source["score_sha256"]:
        raise ValueError("fitting cache uses different scores")
    data = path(source["data"])
    refs = pl.read_parquet(data / "train/ref.parquet")
    wanted = refs.filter((pl.col("fold") == 0) & pl.Series(post.partition(refs["rid"].to_numpy(), 3) == 1))
    raw = pl.scan_parquet(prepared)
    touched = raw.join(wanted.select(pl.col("rid").alias("qid")).lazy(), on="qid", how="inner").select("tid").unique()
    frame = raw.join(touched, on="tid", how="semi").sort("tid", "qid").collect(engine="streaming")
    state = rfeat.prep(refs, data, "train", root / "fit/normalizer.json", cache)
    targets = stack2.raw_targets(data, "train")
    active = frame.select(stack2.active()).to_series().to_numpy()
    x = stack2.matrix(frame.filter(pl.Series(active)), state, targets, threads)
    lookup = np.full(len(refs), -1, np.int32)
    lookup[wanted["rid"].to_numpy()] = np.arange(len(wanted))
    qid = frame["qid"].to_numpy()
    local = lookup[qid]
    labels = np.where(local >= 0, frame["y"].to_numpy(), 0).astype(np.uint8)
    np.savez_compressed(root / "eval.npz", x=x, active=np.flatnonzero(active), qid=qid,
                        tid=frame["tid"].to_numpy(), local=local, y=labels, raw=frame["prob"].to_numpy(),
                        country=frame["co"].to_numpy().astype(str), segment=frame["seg"].to_numpy(),
                        degree=wanted["deg"].to_numpy())
    metadata = {"version": 1, "kind": "source1-macro-optuna-cache", "fit": fit,
                "countries": dict(wanted.group_by("co").len().iter_rows()), "references": len(wanted),
                "targets": frame["tid"].n_unique(), "pairs": len(frame), "active_pairs": int(active.sum()),
                "target_scope": "all candidates of every target touching a search reference; all possible false merges retained",
                "label_scope": "fold0 partition1 only; other reference labels removed; partition2 excluded from search",
                "files": {n: infer._sha(root / n) for n in ("fit/fit.npz", "fit/normalizer.json", "eval.npz")}}
    infer._write(root / "cache.json", metadata)
    return metadata


def metric(e, probability, countries, threads=1):
    selected = e["local"] >= 0
    edges = np.linspace(post.logit([.02])[0], post.logit([.999])[0], 25)
    bins = np.searchsorted(edges, post.logit(probability), side="right")
    hist = {}
    for country in countries:
        for segment in range(3):
            take = selected & (e["country"] == country) & (e["segment"] == segment)
            hist[f"{country}|{segment}"] = [np.bincount(bins[take], minlength=len(edges) + 1).astype(float),
                                           np.bincount(bins[take], weights=e["y"][take], minlength=len(edges) + 1)]
    curves = post.curves(hist, hist, countries, countries, edges, transfer=False)
    calibrated = post.adjust(probability, e["country"], e["segment"], {"curves": curves})
    top = decode.winners(e["qid"], e["tid"], calibrated, e["raw"])
    top = top[selected[top]]
    keep, _ = decode.choose(e["local"][top], e["tid"][top], calibrated[top], e["raw"][top], threads=threads)
    n = len(e["degree"])
    result = post_eval.metric(e["local"][top], e["y"][top], keep, e["degree"], np.ones(n, bool))
    return {"macro_f05": result["macro_f05"], "precision": result["pair_precision"], "recall": result["pair_recall"],
            "pairs": result["pairs"]}


def load(root):
    root = path(root)
    meta = infer._json(root / "cache.json")
    if meta.get("kind") != "source1-macro-optuna-cache" or meta.get("version") != 1:
        raise ValueError("invalid search cache")
    for name, sha in meta["files"].items():
        if ".." in path(name).parts or path(name).is_absolute() or infer._sha(root / name) != sha:
            raise ValueError("search cache changed")
    fit = dict(np.load(root / "fit/fit.npz", allow_pickle=False))
    evaluate = dict(np.load(root / "eval.npz", allow_pickle=False))
    if fit["x"].shape[1] != len(stack2.features) or evaluate["x"].shape[1] != len(stack2.features):
        raise ValueError("search feature contract mismatch")
    return meta, fit, evaluate


def storage(journal):
    return optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(journal)))


def worker(root, journal, models, count, threads, seed):
    meta, fit, evaluate = load(root)
    study = optuna.load_study(study_name="source1-macro-f05", storage=storage(journal),
                              sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=16, constant_liar=True))
    x, y, w = fit["x"], fit["y"], fit["weight"]
    train_mask, val_mask = fit["train"], fit["valid"]

    def objective(trial):
        params = {"objective": "binary", "metric": "binary_logloss", "seed": 19, "verbosity": -1,
                  "num_threads": threads, "deterministic": True, "force_col_wise": True, "bagging_freq": 1,
                  "learning_rate": trial.suggest_float("learning_rate", .02, .12, log=True),
                  "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63, 127]),
                  "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10, -1]),
                  "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 400, log=True),
                  "lambda_l2": trial.suggest_float("lambda_l2", .001, 100., log=True),
                  "bagging_fraction": trial.suggest_float("bagging_fraction", .6, 1.),
                  "feature_fraction": trial.suggest_float("feature_fraction", .65, 1.)}
        maximum = trial.suggest_categorical("round_limit", [400, 800, 1200])
        train_set = lgb.Dataset(x[train_mask], label=y[train_mask], weight=w[train_mask], feature_name=stack2.features)
        val_set = lgb.Dataset(x[val_mask], label=y[val_mask], weight=w[val_mask], feature_name=stack2.features, reference=train_set)
        model = lgb.train(params, train_set, num_boost_round=maximum, valid_sets=[val_set], callbacks=[lgb.early_stopping(40, verbose=False)])
        rounds = model.best_iteration or maximum
        model = lgb.train(params, lgb.Dataset(x, label=y, weight=w, feature_name=stack2.features), num_boost_round=rounds)
        probability = evaluate["raw"].copy()
        probability[evaluate["active"]] = model.predict(evaluate["x"], num_threads=threads)
        result = metric(evaluate, probability, meta["countries"], threads)
        model.save_model(str(path(models) / f"trial-{trial.number}.txt"))
        infer._write(path(models) / f"trial-{trial.number}.json", {"number": trial.number, "params": params,
                                                                   "rounds": rounds, "metrics": result})
        trial.set_user_attr("rounds", rounds)
        trial.set_user_attr("training_params", params)
        trial.set_user_attr("metrics", result)
        return result["macro_f05"]

    study.optimize(objective, n_trials=count, gc_after_trial=True)


def run(root, out, trials=64, workers=8, threads=8):
    root, out = path(root).resolve(), path(out).resolve()
    if min(trials, workers, threads) < 1 or workers > trials or workers * threads > (os.cpu_count() or 1):
        raise ValueError("invalid trial parallelism")
    if (out / "study.json").exists():
        raise ValueError("study output already exists")
    out.mkdir(parents=True, exist_ok=True)
    models = out / "trials"
    models.mkdir(exist_ok=True)
    sh.copyfile(root / "fit/normalizer.json", out / "normalizer.json")
    meta, _, _ = load(root)
    start = time.monotonic()
    with tf.TemporaryDirectory() as temp:
        scratch = path(temp)
        journal = scratch / "study.journal"
        study = optuna.create_study(study_name="source1-macro-f05", storage=storage(journal), direction="maximize")
        study.enqueue_trial({"learning_rate": .05, "num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 100,
                             "lambda_l2": 10., "bagging_fraction": .8, "feature_fraction": .9, "round_limit": 400})
        try:
            with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                jobs = [pool.submit(worker, root, journal, models, trials // workers + (i < trials % workers), threads, 42 + i)
                        for i in range(workers)]
                for job in jobs:
                    job.result()
        finally:
            sh.copyfile(journal, out / "study.journal")
            rows = [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params, "details": t.user_attrs} for t in study.trials]
            infer._write(out / "trial-records.json", rows)
        if len(rows) != trials or any(t["state"] != "COMPLETE" for t in rows):
            raise ValueError("parallel search did not complete each requested trial exactly once")
        best = study.best_trial
        sh.copyfile(models / f"trial-{best.number}.txt", out / "lgb.txt")
    metadata = {"version": 1, "kind": "postgate-pairwise-rich-stack", "features": stack2.features,
                "source_config_sha256": meta["fit"]["source_config_sha256"], "data_meta_sha256": meta["fit"]["data_meta_sha256"],
                "source_scores_sha256": meta["fit"]["source_scores_sha256"], "fit_pairs": meta["fit"]["fit_pairs"],
                "fit_positive": meta["fit"]["fit_positive"], "fit_partition": {"modulus": 3, "remainder": 0},
                "calibration_partition": {"modulus": 3, "remainder": 1}, "validation_partition": {"modulus": 3, "remainder": 2},
                "params": best.user_attrs["training_params"], "rounds": best.user_attrs["rounds"],
                "selection": "optuna maximizes complete-incidence source1 macro f0.5 on partition1; partition2 excluded",
                "files": {name: infer._sha(out / name) for name in ("lgb.txt", "normalizer.json")}}
    infer._write(out / "metadata.json", metadata)
    report = {"version": 1, "best_trial": best.number, "best_search_macro_f05": best.value, "trials": rows,
              "workers": workers, "threads_per_worker": threads, "seconds": time.monotonic() - start,
              "references": meta["references"], "targets": meta["targets"], "pairs": meta["pairs"],
              "cache_sha256": infer._sha(root / "cache.json"), "public_score": None}
    infer._write(out / "study.json", report)
    return report


def claim(journal):
    study = optuna.load_study(study_name="claim-check", storage=storage(journal))
    trial = study.ask()
    time.sleep(.02)
    study.tell(trial, float(trial.number))
    return trial.number


def check():
    e = {"qid": np.array([0, 2, 1, 0, 1, 3]), "tid": np.array([0, 0, 1, 2, 2, 3]),
         "local": np.array([0, -1, 1, 0, 1, -1]), "y": np.array([1, 0, 1, 0, 0, 0]),
         "raw": np.array([.95, .7, .95, .1, .15, .999]), "country": np.array(["us"] * 6),
         "segment": np.zeros(6, np.uint8), "degree": np.array([1, 1])}
    full = metric(e, e["raw"], {"us": 2})
    reduced = {k: v[:-1] if k != "degree" else v for k, v in e.items()}
    assert full == metric(reduced, reduced["raw"], {"us": 2}) and full["macro_f05"] == 1.
    assert not np.any(e["y"][e["local"] < 0])
    with tf.TemporaryDirectory() as tmp:
        journal = path(tmp) / "study.journal"
        study = optuna.create_study(study_name="claim-check", storage=storage(journal), direction="maximize")
        study.enqueue_trial({})
        with cf.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as pool:
            claimed = list(pool.map(claim, [journal] * 8))
        assert len(set(claimed)) == 8 and len(study.trials) == 8
        assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    print("incidence-complete search checks passed")


if __name__ == "__main__":
    parser = ap.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "check"))
    parser.add_argument("--scores", type=path, default=path("artifacts/post-train.parquet"))
    parser.add_argument("--root", type=path, default=path("artifacts/optuna-cache"))
    parser.add_argument("--cache", type=path, default=path("cache"))
    parser.add_argument("--out", type=path, default=path("artifacts/optuna-study"))
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "prepare":
        print(json.dumps(prepare(args.scores, args.root, args.cache, args.threads), indent=2))
    else:
        result = run(args.root, args.out, args.trials, args.workers, args.threads)
        print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))
