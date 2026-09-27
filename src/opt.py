'parallel cached search on incidence-complete source1 macro f0.5'
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


def prepare(prepared, root, cache, threads=8, remainder=1):
    if remainder not in (1, 2):
        raise ValueError("cache partition must be search or development")
    prepared, root, cache = path(prepared), path(root), path(cache)
    src = post.verified(prepared, "train")
    fit = infer._json(root / "fit/cache.json")
    if fit["source_scores_sha256"] != src["score_sha256"]:
        raise ValueError("fitting cache uses different scores")
    data = path(src["data"])
    refs = pl.read_parquet(data / "train/ref.parquet")
    wanted = refs.filter((pl.col("fold") == 0) & pl.Series(post.partition(refs["rid"].to_numpy(), 3) == remainder))
    raw = pl.scan_parquet(prepared)
    touched = raw.join(wanted.select(pl.col("rid").alias("qid")).lazy(), on="qid", how="inner").select("tid").unique()
    frame = raw.join(touched, on="tid", how="semi").sort("tid", "qid").collect(engine="streaming")
    name_change, neural_columns = fit.get("name_change", False), fit.get("neural_columns", [])
    if fit["features"] != stack2.feature_names(name_change, neural_columns):
        raise ValueError("fitting feature contract changed")
    maker = stack2.gfeat if name_change else rfeat
    state = maker.prep(refs, data, "train", root / "fit/normalizer.json", cache)
    targets = stack2.raw_targets(data, "train")
    active = frame.select(stack2.active()).to_series().to_numpy()
    x = stack2.matrix(frame.filter(pl.Series(active)), state, targets, threads, name_change, neural_columns)
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
                 "label_scope": f"fold0 partition{remainder} only; other reference labels removed",
                 "purpose": "search" if remainder == 1 else "development",
                 "reference_partition": {"modulus": 3, "remainder": remainder},
                "files": {n: infer._sha(root / n) for n in ("fit/fit.npz", "fit/normalizer.json", "eval.npz")}}
    infer._write(root / "cache.json", metadata)
    return metadata


def metric(e, probability, countries, threads=1, folds=1):
    if not isinstance(folds, int) or folds < 1:
        raise ValueError("invalid calibration fold count")
    sel = e["local"] >= 0
    edges = np.linspace(post.logit([.02])[0], post.logit([.999])[0], 25)
    bins = np.searchsorted(edges, post.logit(probability), side="right")
    n = len(e["degree"])
    if not n or folds > n:
        raise ValueError("insufficient references for calibration folds")
    group = post.partition(np.arange(n), folds)
    row_group = np.full(len(sel), -1, np.int32)
    row_group[sel] = group[e["local"][sel]]
    total, hits, count = 0., 0, 0
    for fold in range(folds):
        anchors = group == fold
        if not anchors.any():
            continue
        fitting = sel if folds == 1 else sel & (row_group != fold)
        if not fitting.any():
            raise ValueError("empty cross-calibration fitting fold")
        hist = {}
        for country in countries:
            for segment in range(3):
                take = fitting & (e["country"] == country) & (e["segment"] == segment)
                hist[f"{country}|{segment}"] = [np.bincount(bins[take], minlength=len(edges) + 1).astype(float),
                                               np.bincount(bins[take], weights=e["y"][take], minlength=len(edges) + 1)]
        curves = post.curves(hist, hist, countries, countries, edges, transfer=False)
        calibrated = post.adjust(probability, e["country"], e["segment"], {"curves": curves})
        top = decode.winners(e["qid"], e["tid"], calibrated, probability)
        top = top[sel[top] & (row_group[top] == fold)]
        keep, _ = decode.choose(e["local"][top], e["tid"][top], calibrated[top], probability[top], threads=threads)
        res = post_eval.metric(e["local"][top], e["y"][top], keep, e["degree"], anchors)
        total += res["macro_f05"] * int(anchors.sum())
        hits += int(e["y"][top][keep].sum())
        count += res["pairs"]
    truth = int(e["degree"].sum())
    return {"macro_f05": total / n, "precision": hits / count if count else 1.,
            "recall": hits / truth if truth else 1., "pairs": count}


def parameters(trial, threads, names, wide=False):
    params = {"objective": "binary", "metric": "binary_logloss", "seed": 19, "verbosity": -1,
              "num_threads": threads, "deterministic": True, "force_col_wise": True, "bagging_freq": 1,
              "learning_rate": trial.suggest_float("learning_rate", .008 if wide else .02, .15 if wide else .12, log=True),
              "num_leaves": trial.suggest_categorical("num_leaves", [7, 15, 31, 63, 127, 255] if wide else [15, 31, 63, 127]),
              "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8, 10, 12, -1] if wide else [6, 8, 10, -1]),
              "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10 if wide else 20, 1000 if wide else 400, log=True),
              "lambda_l2": trial.suggest_float("lambda_l2", .0001 if wide else .001, 300. if wide else 100., log=True),
              "bagging_fraction": trial.suggest_float("bagging_fraction", .5 if wide else .6, 1.),
              "feature_fraction": trial.suggest_float("feature_fraction", .5 if wide else .65, 1.)}
    if wide:
        params.update({"lambda_l1": trial.suggest_float("lambda_l1", 0., 20.),
                       "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0., 1.),
                       "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", .0001, 10., log=True),
                       "scale_pos_weight": trial.suggest_float("scale_pos_weight", .3, 3., log=True),
                       "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", .6, 1.),
                       "max_bin": trial.suggest_categorical("max_bin", [127, 255, 511]),
                       "path_smooth": trial.suggest_categorical("path_smooth", [0., .1, 1., 10.]),
                       "extra_trees": trial.suggest_categorical("extra_trees", [False, True])})
        group = trial.suggest_categorical("feature_set", ["all", "stable", "scores", "large", "no_bge"])
        keep = np.ones(len(names), float)
        for i, name in enumerate(names):
            if group == "stable" and name.startswith("g_"):
                keep[i] = 0
            if group == "scores" and name not in ("gate_logit", "neural_logit") and not name.startswith("logit_np_"):
                keep[i] = 0
            if group in ("large", "no_bge"):
                if name == "neural_logit":
                    keep[i] = 0
                if name.startswith("logit_np_m"):
                    member = int(name.removeprefix("logit_np_m"))
                    if (group == "large" and member >= 6) or (group == "no_bge" and member in (6, 7, 8)):
                        keep[i] = 0
        params["feature_contri"] = keep.tolist()
    return params


def load(root, search_only=False):
    root = path(root)
    meta = infer._json(root / "cache.json")
    if meta.get("kind") != "source1-macro-optuna-cache" or meta.get("version") != 1:
        raise ValueError("invalid search cache")
    if search_only and (meta.get("purpose", "search") != "search" or
                        meta.get("reference_partition", {"modulus": 3, "remainder": 1}) != {"modulus": 3, "remainder": 1}):
        raise ValueError("development cache cannot be used for search")
    for name, sha in meta["files"].items():
        if ".." in path(name).parts or path(name).is_absolute() or infer._sha(root / name) != sha:
            raise ValueError("search cache changed")
    fit = dict(np.load(root / "fit/fit.npz", allow_pickle=False))
    evaluate = dict(np.load(root / "eval.npz", allow_pickle=False))
    names = stack2.feature_names(meta["fit"].get("name_change", False), meta["fit"].get("neural_columns", []))
    if meta["fit"]["features"] != names or fit["x"].shape[1] != len(names) or evaluate["x"].shape[1] != len(names):
        raise ValueError("search feature contract mismatch")
    return meta, fit, evaluate


def storage(journal):
    return optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(journal)))


def worker(root, journal, models, count, threads, seed, wide=False, folds=1):
    meta, fit, evaluate = load(root, search_only=True)
    names = meta["fit"]["features"]
    study = optuna.load_study(study_name="source1-macro-f05", storage=storage(journal),
                              sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=16, constant_liar=True))
    x, y, w = fit["x"], fit["y"], fit["weight"]
    train_mask, val_mask = fit["train"], fit["valid"]

    def objective(trial):
        params = parameters(trial, threads, names, wide)
        maximum = trial.suggest_categorical("round_limit", [400, 800, 1200, 2000] if wide else [400, 800, 1200])
        train_set = lgb.Dataset(x[train_mask], label=y[train_mask], weight=w[train_mask], feature_name=names)
        val_set = lgb.Dataset(x[val_mask], label=y[val_mask], weight=w[val_mask], feature_name=names, reference=train_set)
        model = lgb.train(params, train_set, num_boost_round=maximum, valid_sets=[val_set], callbacks=[lgb.early_stopping(40, verbose=False)])
        rounds = model.best_iteration or maximum
        model = lgb.train(params, lgb.Dataset(x, label=y, weight=w, feature_name=names), num_boost_round=rounds)
        prob = evaluate["raw"].copy()
        prob[evaluate["active"]] = model.predict(evaluate["x"], num_threads=threads)
        res = metric(evaluate, prob, meta["countries"], threads, folds)
        model.save_model(str(path(models) / f"trial-{trial.number}.txt"))
        infer._write(path(models) / f"trial-{trial.number}.json", {"number": trial.number, "params": params,
                                                                   "rounds": rounds, "metrics": res})
        trial.set_user_attr("rounds", rounds)
        trial.set_user_attr("training_params", params)
        trial.set_user_attr("metrics", res)
        return res["macro_f05"]

    study.optimize(objective, n_trials=count, gc_after_trial=True)


def run(root, out, trials=64, workers=8, threads=8, selected_file=None, wide=False, folds=1, seed=42):
    root, out = path(root).resolve(), path(out).resolve()
    if min(trials, workers, threads) < 1 or workers > trials or workers * threads > (os.cpu_count() or 1):
        raise ValueError("invalid trial parallelism")
    if (out / "study.json").exists():
        raise ValueError("study output already exists")
    out.mkdir(parents=True, exist_ok=True)
    models = out / "trials"
    models.mkdir(exist_ok=True)
    sh.copyfile(root / "fit/normalizer.json", out / "normalizer.json")
    meta, _, _ = load(root, search_only=True)
    if not isinstance(folds, int) or not 1 <= folds <= meta["references"]:
        raise ValueError("invalid calibration fold count")
    if wide and (meta["fit"].get("neural_columns") != [f"np_m{i}" for i in range(15)] or
                 meta["fit"]["source_config_sha256"] != "7106e3923df375cb62753ba0c8bbf77fb5bf0ab2f7f7b1db98fe49931ff26017"):
        raise ValueError("wide feature ablations require the frozen 15-member learned ensemble")
    start = time.monotonic()
    with tf.TemporaryDirectory() as temp:
        scratch = path(temp)
        journal = scratch / "study.journal"
        study = optuna.create_study(study_name="source1-macro-f05", storage=storage(journal), direction="maximize")
        selected_file = selected_file or path(__file__).resolve().parents[1] / "reports/optuna-search.json"
        selected_params = stack2.fit_params(threads, selected_file)
        sel = {key: selected_params[key] for key in ("learning_rate", "num_leaves", "max_depth", "min_data_in_leaf",
                                                        "lambda_l2", "bagging_fraction", "feature_fraction")}
        sel["round_limit"] = 800
        defaults = {"lambda_l1": 0., "min_gain_to_split": 0., "min_sum_hessian_in_leaf": .001,
                    "scale_pos_weight": 1., "feature_fraction_bynode": 1., "max_bin": 255,
                    "path_smooth": 0., "extra_trees": False, "feature_set": "all"} if wide else {}
        sel.update({key: selected_params.get(key, value) for key, value in defaults.items()})
        if wide:
            sel["feature_set"] = infer._json(selected_file).get("feature_set", "all")
        study.enqueue_trial(sel)
        if trials > 1:
            study.enqueue_trial({"learning_rate": .05, "num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 100,
                                  "lambda_l2": 10., "bagging_fraction": .8, "feature_fraction": .9, "round_limit": 400, **defaults})
        try:
            with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                jobs = [pool.submit(worker, root, journal, models, trials // workers + (i < trials % workers), threads, seed + i, wide, folds)
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
    metadata = {"version": 1, "kind": "postgate-pairwise-rich-stack", "features": meta["fit"]["features"],
                 "name_change": meta["fit"].get("name_change", False), "neural_columns": meta["fit"].get("neural_columns", []),
                "source_config_sha256": meta["fit"]["source_config_sha256"], "data_meta_sha256": meta["fit"]["data_meta_sha256"],
                "source_scores_sha256": meta["fit"]["source_scores_sha256"], "fit_pairs": meta["fit"]["fit_pairs"],
                "fit_positive": meta["fit"]["fit_positive"], "fit_partition": {"modulus": 3, "remainder": 0},
                "calibration_partition": {"modulus": 3, "remainder": 1}, "validation_partition": {"modulus": 3, "remainder": 2},
                 "params": best.user_attrs["training_params"], "rounds": best.user_attrs["rounds"],
                 "calibration_folds": folds, "feature_set": best.params.get("feature_set", "all"),
                "selection": "optuna maximizes complete-incidence source1 macro f0.5 on partition1; partition2 excluded",
                "files": {name: infer._sha(out / name) for name in ("lgb.txt", "normalizer.json")}}
    infer._write(out / "metadata.json", metadata)
    report = {"version": 1, "best_trial": best.number, "best_search_macro_f05": best.value, "trials": rows,
              "workers": workers, "threads_per_worker": threads, "seconds": time.monotonic() - start,
              "references": meta["references"], "targets": meta["targets"], "pairs": meta["pairs"],
               "cache_sha256": infer._sha(root / "cache.json"), "public_score": None}
    report["queued_selection_sha256"] = infer._sha(selected_file)
    report.update({"wide": wide, "calibration_folds": folds, "seed": seed,
                   "objective_version": 2, "tie_breaker": "candidate model probability, matching production decoding"})
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
    tied = {"qid": np.array([0, 1]), "tid": np.array([0, 0]), "local": np.array([0, -1]),
            "y": np.array([1, 0]), "raw": np.array([.99, .01]), "country": np.array(["us", "us"]),
            "segment": np.zeros(2, np.uint8), "degree": np.array([1])}
    assert metric(tied, np.array([.91, .93]), {"us": 1})["macro_f05"] == 0.
    cross = {"qid": np.array([0, 1]), "tid": np.array([0, 1]), "local": np.array([0, 1]),
             "y": np.array([0, 1]), "raw": np.array([.8, .2]), "country": np.array(["us", "us"]),
             "segment": np.zeros(2, np.uint8), "degree": np.array([0, 1])}
    assert metric(cross, cross["raw"], {"us": 2})["macro_f05"] == .5
    assert metric(cross, cross["raw"], {"us": 2}, folds=2)["macro_f05"] == 0.
    names = ["gate_logit", "neural_logit", "logit_np_m0", "logit_np_m6"]
    fixed = {"learning_rate": .05, "num_leaves": 15, "max_depth": 4, "min_data_in_leaf": 20,
             "lambda_l2": 1., "bagging_fraction": 1., "feature_fraction": 1., "lambda_l1": 0.,
             "min_gain_to_split": 0., "min_sum_hessian_in_leaf": .001, "scale_pos_weight": 1.,
             "feature_fraction_bynode": 1., "max_bin": 255, "path_smooth": 0., "extra_trees": False,
             "feature_set": "large"}
    params = parameters(optuna.trial.FixedTrial(fixed), 1, names, wide=True)
    assert params["feature_contri"] == [1., 0., 1., 0.]
    x = np.random.default_rng(19).normal(size=(256, len(names)))
    model = lgb.train(params, lgb.Dataset(x, label=x[:, 0] > 0, feature_name=names), num_boost_round=8)
    changed = x.copy()
    changed[:, [1, 3]] += 100
    assert np.array_equal(model.predict(x), model.predict(changed))
    with tf.TemporaryDirectory() as tmp:
        root = path(tmp) / "cache"
        (root / "fit").mkdir(parents=True)
        neural_columns = ["np_m0", "np_m1"]
        names = stack2.feature_names(True, neural_columns)
        x = np.zeros((40, len(names)), np.float32)
        y = np.arange(40) % 2
        train_mask = np.arange(40) < 30
        np.savez_compressed(root / "fit/fit.npz", x=x, y=y, weight=np.ones(40), train=train_mask, valid=~train_mask)
        infer._write(root / "fit/normalizer.json", {})
        np.savez_compressed(root / "eval.npz", **e, x=np.zeros((len(e["raw"]), len(names)), np.float32), active=np.arange(len(e["raw"])))
        fit_meta = {"features": names, "name_change": True, "neural_columns": neural_columns,
                    "source_config_sha256": "check", "source_scores_sha256": "check", "data_meta_sha256": "check",
                    "fit_pairs": 40, "fit_positive": 20}
        infer._write(root / "cache.json", {"version": 1, "kind": "source1-macro-optuna-cache", "fit": fit_meta,
                     "countries": {"us": 2}, "references": 2, "targets": 4, "pairs": 6,
                     "files": {n: infer._sha(root / n) for n in ("fit/fit.npz", "fit/normalizer.json", "eval.npz")}})
        out = path(tmp) / "model"
        res = run(root, out, trials=2, workers=1, threads=1)
        model_meta, _ = stack2.bundle(out)
        assert model_meta["features"] == names and model_meta["neural_columns"] == neural_columns and model_meta["name_change"]
        exp = infer._json(path(__file__).resolve().parents[1] / "reports/optuna-search.json")["best"]["params"]
        assert res["trials"][0]["params"] == exp
        metadata = infer._json(root / "cache.json")
        metadata["purpose"] = "development"
        metadata["reference_partition"] = {"modulus": 3, "remainder": 2}
        infer._write(root / "cache.json", metadata)
        load(root)
        try:
            load(root, search_only=True)
            raise AssertionError("development labels entered search")
        except ValueError as exc:
            assert "development cache" in str(exc)
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
    parser.add_argument("--parameters", type=path)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--cal-folds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "prepare":
        print(json.dumps(prepare(args.scores, args.root, args.cache, args.threads, args.partition), indent=2))
    else:
        result = run(args.root, args.out, args.trials, args.workers, args.threads, args.parameters,
                     args.wide, args.cal_folds, args.seed)
        print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))
