"""cpu-only, selected-query screening on cached safe teammate features.

run from the repo root with .venv/bin/python src/tune.py --check or --extended.
fold 1 targets are not scored; fold 0 comparisons are diagnostic only.
"""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path as path

import numpy as np
import polars as pl

try:
    from match import _blend, _top
    from tfeat import contract
    from train import _tops
except ImportError:
    from src.match import _blend, _top
    from src.tfeat import contract
    from src.train import _tops


root = path(__file__).resolve().parents[1]
features = root / "artifacts/teammate-benchmark/features"
gate = root / "artifacts/upgraded-gate"
out = root / "artifacts/overnight-tune"
report = root / "reports/overnight-tune"
neural_scores = root / "artifacts/teammate-benchmark/neural_gate_paired_scores.parquet"
seed = 42
precisions = (0.99, 0.995, 0.999)


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(split, names):
    files = sorted((features / split).glob("*.parquet"))
    if not files:
        raise ValueError(f"missing cached {split} feature parts")
    d = pl.concat([pl.read_parquet(p, columns=["s1_id", "o_id", "label", *names]) for p in files], rechunk=True)
    qid, tid, y = (d[c].to_numpy() for c in ("s1_id", "o_id", "label"))
    if d.select("s1_id", "o_id").n_unique() != len(d) or not set(np.unique(y)) == {0, 1}:
        raise ValueError(f"invalid or duplicate {split} candidate pairs")
    x = d.select(names).to_numpy().astype(np.float32, copy=False)
    if np.isinf(x).any():
        raise ValueError(f"infinite {split} features")
    return x, y, qid, tid, files


def _owners():
    paths = [root / f"cache/data/train/s{s}.parquet" for s in (2, 3)]
    parts = [pl.read_parquet(p, columns=["rid", "own"]) for p in paths]
    ids = np.concatenate([d["rid"].to_numpy() for d in parts])
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("prepared target ids are not consecutive")
    return np.concatenate([d["own"].to_numpy() for d in parts]), paths


def _split(qid, tid, owner, fraction=0.2):
    """split both anchor and true target owner; discard crossing edges."""
    # group aliases by actual owner; group orphans by tid.
    def held(ids):
        a = ids.astype(np.uint64)
        a = (a ^ (a >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        a = (a ^ (a >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        return ((a ^ (a >> np.uint64(31))) % np.uint64(100)) < fraction * 100

    target_group = np.where(owner[tid] >= 0, owner[tid], tid.astype(np.int64) + len(owner))
    anchor_held, target_held = held(qid), held(target_group)
    fit = ~anchor_held & ~target_held
    stop = anchor_held & target_held
    if not fit.any() or not stop.any() or np.intersect1d(qid[fit], qid[stop]).size or np.intersect1d(tid[fit], tid[stop]).size or np.intersect1d(target_group[fit], target_group[stop]).size:
        raise ValueError("internal early-stop groups overlap")
    return fit, stop, target_group


def _matched(y, tid, qid, prob, denominator):
    """threshold target-top1 at each score; count all selected positive owners."""
    ix = _tops(tid, qid, prob)
    order = ix[np.lexsort((qid[ix], -prob[ix]))]
    scores, hits = prob[order], y[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    selected = ends + 1
    true = np.cumsum(hits)[ends]
    precision = true / selected
    out = {"top1_candidates": len(order)}
    for p in precisions:
        ok = np.flatnonzero(precision >= p)
        j = ok[np.argmax(true[ok])] if len(ok) else None
        out[f"{p:.3f}"] = None if j is None else {
            "cut": float(scores[ends[j]]), "selected": int(selected[j]), "true_positive": int(true[j]),
            "precision": float(precision[j]), "recall_all_selected_positive_owners": float(true[j] / denominator),
        }
    for cut in (0.5, 0.8, 0.95):
        chosen = prob[ix] >= cut
        n, h = int(chosen.sum()), int(y[ix[chosen]].sum())
        out[f"fixed_{cut:.2f}"] = {"selected": n, "true_positive": h, "precision": h / n if n else 1.0,
                                     "recall_all_selected_positive_owners": h / denominator}
    return out


def _score(name, prob, y, qid, tid, owner, neural, results, predictions):
    if len(prob) != len(y) or not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError(f"invalid probability from {name}")
    denom = int((owner[np.unique(tid)] >= 0).sum())
    positive = int(y.sum())
    if positive > denom:
        raise ValueError("more candidate positives than actual positive targets")
    ix = _tops(tid, qid, prob)
    ranked = pl.DataFrame({"qid": qid, "tid": tid, "label": y, "gate": prob.astype(np.float32)})
    top3 = _top(ranked, 3)
    info = {"candidate_pairs": len(y), "targets": len(np.unique(tid)), "positive_target_denominator": denom,
            "candidate_true_owner_pairs": positive, "candidate_recall": positive / denom,
            "gate_top1_recall": float(y[ix].sum() / denom),
            "gate_top3_recall": float(top3["label"].sum() / denom),
            "gate_high_precision": _matched(y, tid, qid, prob, denom)}
    if neural is not None:
        # exact join; never score a partial neural intersection.
        joined = top3.join(neural, on=["qid", "tid"], how="left", validate="1:1")
        if joined["prob"].null_count() or (joined["label"] != joined["neural_label"]).any():
            raise ValueError("missing or label-inconsistent neural score")
        blend = _blend(joined["prob"].to_numpy(), joined["gate"].to_numpy(), 0.6)
        info["neural_top3_blend_high_precision"] = _matched(joined["label"].to_numpy(), joined["tid"].to_numpy(),
                                                                joined["qid"].to_numpy(), blend, denom)
    results[name] = info
    predictions[name] = prob.astype(np.float32)
    print(name, "top3", round(info["gate_top3_recall"], 5),
          "recall@.995", info["gate_high_precision"]["0.995"], flush=True)


def check():
    names = json.loads((gate / "metadata.json").read_text())["feature_names"]
    contract(json.loads((gate / "metadata.json").read_text()))
    assert len(names) == 54 and not set(names) & {"label", "s1_id", "o_id", "own"}
    assert not any(n.startswith("s1_") for n in names)
    # aliases stay together; missing owners group by tid.
    owner = np.array([-1, 1, 1, -1], dtype=np.int32)
    qid = np.arange(100, dtype=np.int32).repeat(4)
    tid = np.tile(np.arange(4), 100)
    fit, stop, groups = _split(qid, tid, owner, 0.5)
    assert not np.intersect1d(groups[fit], groups[stop]).size
    assert not np.intersect1d(qid[fit], qid[stop]).size
    assert not np.intersect1d(tid[fit], tid[stop]).size
    assert groups[tid == 1][0] == groups[tid == 2][0]
    y = np.array([1, 0, 0], dtype=np.uint8)
    m = _matched(y, np.array([0, 0, 1]), np.array([1, 2, 3]), np.array([.9, .1, .8]), 2)
    assert m["0.990"]["recall_all_selected_positive_owners"] == 0.5
    print("tune checks passed")


def run(threads=4, rounds=450):
    if not 1 <= threads <= 8 or rounds < 1:
        raise ValueError("threads must be 1..8 and rounds positive")
    start = time.monotonic()
    metadata = json.loads((gate / "metadata.json").read_text())
    names, _ = contract(metadata)
    if len(names) != 54 or any(n.startswith("s1_") or n in {"label", "s1_id", "o_id", "own", "fold", "qid", "tid"} or n.endswith("_id") for n in names):
        raise ValueError("unsafe feature contract")
    x, y, q, t, train_files = _load("train", names)
    xv, yv, qv, tv, val_files = _load("validation", names)
    owner, owner_files = _owners()
    ref = pl.read_parquet(root / "cache/data/train/ref.parquet", columns=["rid", "fold"])
    fold = ref["fold"].to_numpy()
    val_owners = owner[np.unique(tv)]
    if (not np.array_equal(ref["rid"].to_numpy(), np.arange(len(ref))) or (fold[q] != 2).any() or
            (fold[val_owners[val_owners >= 0]] != 0).any() or
            (owner[t[y == 1]] != q[y == 1]).any() or (owner[tv[yv == 1]] != qv[yv == 1]).any()):
        raise ValueError("feature labels or fold assignments contradict prepared truth")
    if np.intersect1d(t, tv).size or np.intersect1d(q, val_owners[val_owners >= 0]).size:
        raise ValueError("train and validation share target ids or true owners")
    fit, stop, groups = _split(q, t, owner)
    if not set(np.unique(y[fit])) == {0, 1} or not set(np.unique(y[stop])) == {0, 1}:
        raise ValueError("internal split lacks both classes")
    neural = pl.read_parquet(neural_scores, columns=["qid", "tid", "label", "prob"]).rename({"label": "neural_label"})
    validation_pairs = pl.DataFrame({"qid": qv, "tid": tv, "label": yv})
    paired = validation_pairs.join(neural, on=["qid", "tid"], how="left", validate="1:1")
    if len(neural) != len(yv) or neural.select("qid", "tid").n_unique() != len(neural) or paired["prob"].null_count() or (paired["label"] != paired["neural_label"]).any():
        raise ValueError("neural scores do not cover exactly the cached validation pairs")
    del paired
    out.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "selected-query fixed lexical candidates, fold0 screening reuse; NOT full-pool validation or leaderboard",
               "features": names, "folds": {"fit": 2, "internal_early_stop": 2, "screening_target_true_owners": 0,
                                             "screening_negative_competitor_ref_folds": {str(i): int((fold[qv] == i).sum()) for i in (0, 1, 2)},
                                             "locked_audit_targets_untouched": 1},
               "split": {"method": "deterministic 20% actual owner/anchor hash; orphan targets grouped by tid; crossing edges unused; refit on all fold2 pairs after selecting iteration",
                         "fit_pairs": int(fit.sum()), "early_stop_pairs": int(stop.sum()), "crossing_pairs": int((~(fit | stop)).sum()),
                         "fit_positive": int(y[fit].sum()), "early_stop_positive": int(y[stop].sum()),
                         "fit_owners": int(np.unique(q[fit]).size), "stop_owners": int(np.unique(q[stop]).size),
                         "fit_target_groups": int(np.unique(groups[fit]).size), "stop_target_groups": int(np.unique(groups[stop]).size)},
               "threads": threads, "round_cap": rounds, "seeds": [7, 23],
               "inputs_sha256": {str(p.relative_to(root)): _sha(p) for p in [gate / "metadata.json", gate / metadata["model_files"]["lgb"], neural_scores,
                                                                         root / "cache/data/train/ref.parquet", *train_files, *val_files, *owner_files]},
               "variants": {}, "scores": {}, "timing_seconds": {},
               "limitations": ["Sampled fold0 owners and orphans are not the full target pool; pair precision, candidate recall, and top3 retention are only diagnostics.",
                               "No valid full-reference macro f0.5: many target distractors/true links are absent from this selected-query corpus.",
                               "No stacking: no genuinely held-out level1 meta-fit plus disjoint cutoff/audit population was created; in-sample base scores would leak.",
                                "Validation candidate negatives include fold1/2 references as competitors; no fold1-owned targets are scored or fold1 audit labels accessed.",
                                "Fold0 reused to screen many candidates; no fold1 audit or deployment/promotion inference."]}
    results, predictions = summary["scores"], {}
    import lightgbm as lgb
    baseline = lgb.Booster(model_file=str(gate / metadata["model_files"]["lgb"]))
    if baseline.feature_name() != names:
        raise ValueError("saved baseline model feature order mismatch")
    tic = time.monotonic()
    _score("existing_safe_lgb", baseline.predict(xv, num_threads=threads), yv, qv, tv, owner, neural, results, predictions)
    summary["timing_seconds"]["existing_safe_lgb"] = round(time.monotonic() - tic, 2)
    summary["variants"]["existing_safe_lgb"] = {"model": str((gate / metadata["model_files"]["lgb"]).relative_to(root)), "trees": baseline.num_trees(), "note": "prior checkpoint stopped on fold0; not independent"}
    del baseline

    base = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                max_bin=255, num_threads=threads, verbosity=-1, seed=7)
    # original safe gate already uses bagging_freq=1.
    grid = {
        "lgb_repro": {},
        "lgb_small_leaf": {"num_leaves": 31, "min_data_in_leaf": 100},
        "lgb_regularized": {"num_leaves": 63, "min_data_in_leaf": 200, "lambda_l2": 5.0, "max_depth": 8},
        "lgb_seed_bag": {"seed": 23, "bagging_fraction": 0.65, "feature_fraction": 0.75},
        "lgb_extra_splits": {"num_leaves": 63, "extra_trees": True},
    }
    for name, change in grid.items():
        tic = time.monotonic()
        params = base | change
        dfit = lgb.Dataset(x[fit], y[fit], feature_name=names, free_raw_data=True)
        dstop = lgb.Dataset(x[stop], y[stop], reference=dfit, free_raw_data=True)
        probe = lgb.train(params, dfit, num_boost_round=rounds, valid_sets=[dstop],
                          callbacks=[lgb.early_stopping(40, verbose=False)])
        trees = max(1, probe.best_iteration)
        del probe, dfit, dstop
        full = lgb.train(params, lgb.Dataset(x, y, feature_name=names), num_boost_round=trees)
        model_file = out / f"{name}.txt"
        full.save_model(str(model_file))
        _score(name, full.predict(xv, num_threads=threads), yv, qv, tv, owner, neural, results, predictions)
        summary["variants"][name] = {"family": "lightgbm", "parameters": params, "internal_best_iteration": trees, "model": model_file.name}
        summary["timing_seconds"][name] = round(time.monotonic() - tic, 2)
        print(name, "seconds", summary["timing_seconds"][name], flush=True)
        del full

    from catboost import CatBoostClassifier as catboost_classifier
    for name, change in {"cat_depth4": {"depth": 4, "l2_leaf_reg": 10},
                         "cat_depth6_bag": {"depth": 6, "l2_leaf_reg": 3, "bootstrap_type": "Bernoulli", "subsample": 0.8, "rsm": 0.8}}.items():
        tic = time.monotonic()
        params = dict(iterations=rounds, learning_rate=0.05, loss_function="Logloss", eval_metric="Logloss",
                      random_seed=7, thread_count=threads, allow_writing_files=False, verbose=False) | change
        probe = catboost_classifier(**params)
        probe.fit(x[fit], y[fit], eval_set=(x[stop], y[stop]), early_stopping_rounds=40, verbose=False)
        trees = probe.best_iteration_ + 1
        del probe
        full = catboost_classifier(**(params | {"iterations": trees}))
        full.fit(x, y, verbose=False)
        model_file = out / f"{name}.cbm"
        full.save_model(str(model_file))
        _score(name, full.predict_proba(xv, thread_count=threads)[:, 1], yv, qv, tv, owner, neural, results, predictions)
        summary["variants"][name] = {"family": "catboost", "parameters": params, "internal_best_iteration": trees, "model": model_file.name}
        summary["timing_seconds"][name] = round(time.monotonic() - tic, 2)
        print(name, "seconds", summary["timing_seconds"][name], flush=True)
        del full

    # sample whole target groups; diagnostic only.
    from sklearn.ensemble import ExtraTreesClassifier as extra_trees_classifier
    import joblib
    tic = time.monotonic()
    tids = np.unique(t)
    rng = np.random.default_rng(seed)
    rng.shuffle(tids)
    chosen = np.zeros(len(owner), dtype=bool)
    chosen[tids[:max(1, int(len(tids) * 160_000 / len(t)))]] = True
    sample = chosen[t]
    params = dict(n_estimators=64, max_depth=14, min_samples_leaf=20, max_features=0.5,
                   bootstrap=False, n_jobs=threads, random_state=seed)
    extra = extra_trees_classifier(**params).fit(x[sample], y[sample])
    model_file = out / "extra_trees_diagnostic.joblib"
    joblib.dump(extra, model_file, compress=3)
    _score("extra_trees_diagnostic", extra.predict_proba(xv)[:, 1], yv, qv, tv, owner, neural, results, predictions)
    summary["variants"]["extra_trees_diagnostic"] = {"family": "sklearn extra trees (diagnostic only; eligibility unverified)", "parameters": params, "sampled_whole_target_pairs": int(sample.sum()), "sampled_targets": int(chosen.sum()), "model": model_file.name}
    summary["timing_seconds"]["extra_trees_diagnostic"] = round(time.monotonic() - tic, 2)
    del extra, sample, chosen

    # fixed ensembles on identical rows; no learned stack.
    for name, a, b in [("lgb_mean", "lgb_repro", "lgb_regularized"),
                       ("lgb_cat_mean", "lgb_repro", "cat_depth4")]:
        tic = time.monotonic()
        pa, pb = predictions[a], predictions[b]
        _score(name, (pa + pb) / 2, yv, qv, tv, owner, neural, results, predictions)
        _score(name + "_logit", _blend(pa, pb, 0.5), yv, qv, tv, owner, neural, results, predictions)
        summary["variants"][name] = {"kind": "arithmetic probability mean", "components": [a, b]}
        summary["variants"][name + "_logit"] = {"kind": "equal-weight logit blend", "components": [a, b]}
        summary["timing_seconds"][name] = round(time.monotonic() - tic, 2)

    pl.DataFrame({"qid": qv, "tid": tv, "label": yv, **predictions}).write_parquet(out / "validation_predictions.parquet", compression="zstd")
    summary["timing_seconds"]["total"] = round(time.monotonic() - start, 2)
    summary["peak_process_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)
    capped = [name for name, variant in summary["variants"].items() if variant.get("internal_best_iteration") == rounds]
    if capped:
        summary["limitations"].append(f"Internal early stopping hit the {rounds}-round cap for {', '.join(capped)}; these are budget-limited comparisons.")
    report.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# overnight cpu tuning — selected-query screening only", "",
             "fold2 owner-disjoint internal early stopping; refit each tree on all fold2 pairs. fold0 reused across variants. no fold1-owned targets or audit labels used. no full-pool or 99.8% claim.", "",
             "| variant | top1 recall | top3 recall | recall @ precision ≥ .995 | neural blend recall @ precision ≥ .995 | seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, score in results.items():
        def recall(key):
            point = score[key]["0.995"]
            return f"{point['recall_all_selected_positive_owners']:.4f}" if point else "n/a"
        lines.append(f"| {name} | {score['gate_top1_recall']:.4f} | {score['gate_top3_recall']:.4f} | {recall('gate_high_precision')} | {recall('neural_top3_blend_high_precision')} | {summary['timing_seconds'].get(name, '—')} |")
    lines += ["", "**outcome:** no new tree or fixed ensemble improved the existing safe gate's .995-precision recall, with or without the same cached neural blend. these sampled results do not justify replacing it.", "",
              f"selected positive target denominator: {results['existing_safe_lgb']['positive_target_denominator']}; candidate recall: {results['existing_safe_lgb']['candidate_recall']:.4f}.",
              f"internal split: {summary['split']['fit_pairs']} fit / {summary['split']['early_stop_pairs']} early stop / {summary['split']['crossing_pairs']} unused crossing pairs.",
              f"total {summary['timing_seconds']['total']}s; peak process rss {summary['peak_process_rss_gib']} GiB; {threads} training threads.",
              f"budget-limited fits hit the {rounds}-round cap: {', '.join(capped) if capped else 'none'}.", "",
              "validation includes fold1 source1 ids **only as negative competitor references** (see json counts), never fold1-owned targets. no owner macro: the sampled candidate population omits many reference links and distractors. sklearn extra trees is diagnostic only pending model-license eligibility. no learned stacking: the available base fit predictions have no independent level1 meta-fit and separate cutoff/audit owner partitions.",
              "", "run from the repo root: `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 .venv/bin/python -u src/tune.py --threads 4 --rounds 450 > artifacts/overnight-tune/run.log 2>&1`; small check: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python src/tune.py --check`.",
              "", "sources and detailed thresholds/parameters: [json](overnight-tune.json); saved models and exact-pair predictions: `artifacts/overnight-tune/`."]
    report.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("complete", summary["timing_seconds"], "peak GiB", summary["peak_process_rss_gib"], flush=True)


def extend(threads=12, rounds=1400):
    if not 1 <= threads <= 12 or not 451 <= rounds <= 1400:
        raise ValueError("extended threads must be 1..12 and rounds 451..1400")
    start = time.monotonic()
    dest = root / "artifacts/overnight-tune-extended"
    doc = root / "reports/overnight-tune-extended"
    metadata = json.loads((gate / "metadata.json").read_text())
    names, _ = contract(metadata)
    if len(names) != 54 or any(n.startswith("s1_") or n.endswith("_id") or n in {"label", "own", "fold", "qid", "tid"} for n in names):
        raise ValueError("unsafe feature contract")
    x, y, q, t, train_files = _load("train", names)
    xv, yv, qv, tv, val_files = _load("validation", names)
    owner, owner_files = _owners()
    folds = pl.read_parquet(root / "cache/data/train/ref.parquet", columns=["rid", "fold"])
    fold = folds["fold"].to_numpy()
    val_owner = owner[np.unique(tv)]
    if (not np.array_equal(folds["rid"].to_numpy(), np.arange(len(fold))) or
            (fold[q] != 2).any() or (fold[val_owner[val_owner >= 0]] != 0).any() or
            (owner[t[y == 1]] != q[y == 1]).any() or (owner[tv[yv == 1]] != qv[yv == 1]).any() or
            np.intersect1d(t, tv).size):
        raise ValueError("cached folds, owners, or pair labels disagree")
    fit, stop, groups = _split(q, t, owner)
    if not set(np.unique(y[fit])) == {0, 1} or not set(np.unique(y[stop])) == {0, 1}:
        raise ValueError("early-stop split lacks a class")
    original = json.loads(report.with_suffix(".json").read_text())
    prior_file = out / "validation_predictions.parquet"
    prior = pl.read_parquet(prior_file)
    if (original["features"] != names or original["split"]["early_stop_pairs"] != int(stop.sum()) or
            original["split"]["fit_pairs"] != int(fit.sum()) or len(prior) != len(yv) or
            any(not np.array_equal(prior[col].to_numpy(), array) for col, array in
                (("qid", qv), ("tid", tv), ("label", yv)))):
        raise ValueError("prior sweep and extended fixed candidate population disagree")
    neural = pl.read_parquet(neural_scores, columns=["qid", "tid", "label", "prob"]).rename({"label": "neural_label"})
    paired = pl.DataFrame({"qid": qv, "tid": tv, "label": yv}).join(neural, on=["qid", "tid"], how="left", validate="1:1")
    if len(neural) != len(yv) or neural.select("qid", "tid").n_unique() != len(neural) or paired["prob"].null_count() or (paired["label"] != paired["neural_label"]).any():
        raise ValueError("cached neural pairs do not exactly match validation")
    del paired
    dest.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "lexical selected-query fold0 screening reuse; not full-pool, not official validation",
               "input_sha256": {str(p.relative_to(root)): _sha(p) for p in [gate / "metadata.json", prior_file, report.with_suffix(".json"), neural_scores,
                                                                            root / "cache/data/train/ref.parquet", *owner_files, *train_files, *val_files]},
               "features": names, "threads": threads, "round_cap": rounds,
               "split": {"method": "prior sweep's deterministic actual-owner and anchor split; crossing edges dropped; final refit uses all fold2 pairs",
                         "fit_pairs": int(fit.sum()), "early_stop_pairs": int(stop.sum()), "crossing_pairs": int((~(fit | stop)).sum()),
                         "fit_positive": int(y[fit].sum()), "early_stop_positive": int(y[stop].sum()),
                         "fit_target_groups": int(np.unique(groups[fit]).size), "stop_target_groups": int(np.unique(groups[stop]).size)},
               "models": {}, "scores": {}, "seconds": {},
               "limitations": ["selected-query pairs omit full-pool distractors and links; high-precision pair recall is diagnostic only",
                               "fold0 screening is reused across variants, so best-of-grid estimates are optimistic",
                               "no full-reference macro f0.5 or 99.8 percent claim; fold1-owned targets are not scored or used for selection",
                               "fold1/2 references are present as negative competitors on fold0-owned targets; no learned stack is fit"]}
    results, predictions = summary["scores"], {}
    for name in ("existing_safe_lgb", "cat_depth4", "cat_depth6_bag"):
        _score(name, prior[name].to_numpy(), yv, qv, tv, owner, neural, results, predictions)
        summary["models"][name] = {"source": str(prior_file.relative_to(root)), "reused": True}

    import lightgbm as lgb
    base = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=1.0, max_bin=255, num_threads=threads, verbosity=-1, seed=7)
    missing = (np.nan_to_num(x[:, names.index("addr_missing2")]) > 0) | (np.nan_to_num(x[:, names.index("addr_empty2")]) > 0)
    hard = (y == 0) & (np.nan_to_num(x[:, names.index("n_ratio")]) >= 90) & (np.nan_to_num(x[:, names.index("a_ratio")]) < 60)
    weights = {"lgb_hard_neg_2x": np.where(hard, 2.0, 1.0).astype(np.float32),
               "lgb_missing_addr_2x": np.where(missing, 2.0, 1.0).astype(np.float32)}
    for name, weight in weights.items():
        tic = time.monotonic()
        dfit = lgb.Dataset(x[fit], y[fit], weight=weight[fit], feature_name=names)
        dstop = lgb.Dataset(x[stop], y[stop], reference=dfit)
        probe = lgb.train(base, dfit, num_boost_round=min(rounds, 650), valid_sets=[dstop],
                          callbacks=[lgb.early_stopping(70, verbose=False)])
        trees = max(1, probe.best_iteration)
        del probe, dfit, dstop
        full = lgb.train(base, lgb.Dataset(x, y, weight=weight, feature_name=names), num_boost_round=trees)
        model_file = dest / f"{name}.txt"
        full.save_model(str(model_file))
        _score(name, full.predict(xv, num_threads=threads), yv, qv, tv, owner, neural, results, predictions)
        summary["models"][name] = {"model": model_file.name, "parameters": base, "best_iteration": trees,
                                    "internal_round_cap": min(rounds, 650), "weight_rule": "2x when negative n_ratio>=90 and a_ratio<60" if name == "lgb_hard_neg_2x" else "2x when addr_missing2 or addr_empty2",
                                    "upweighted_pairs": int(hard.sum() if name == "lgb_hard_neg_2x" else missing.sum())}
        summary["seconds"][name] = round(time.monotonic() - tic, 2)
        print(name, "seconds", summary["seconds"][name], "trees", trees, flush=True)
        del full

    from catboost import CatBoostClassifier as catboost_classifier
    for name, change in (("cat_depth4_1400", {"depth": 4, "l2_leaf_reg": 10}),
                         ("cat_depth6_bag_1400", {"depth": 6, "l2_leaf_reg": 3, "bootstrap_type": "Bernoulli", "subsample": 0.8, "rsm": 0.8})):
        tic = time.monotonic()
        params = dict(iterations=rounds, learning_rate=0.05, loss_function="Logloss", eval_metric="Logloss", random_seed=7,
                      thread_count=threads, allow_writing_files=False, verbose=False) | change
        probe = catboost_classifier(**params)
        probe.fit(x[fit], y[fit], eval_set=(x[stop], y[stop]), early_stopping_rounds=80, verbose=False)
        trees = probe.best_iteration_ + 1
        del probe
        full = catboost_classifier(**(params | {"iterations": trees}))
        full.fit(x, y, verbose=False)
        model_file = dest / f"{name}.cbm"
        full.save_model(str(model_file))
        _score(name, full.predict_proba(xv, thread_count=threads)[:, 1], yv, qv, tv, owner, neural, results, predictions)
        summary["models"][name] = {"model": model_file.name, "parameters": params, "internal_best_iteration": trees,
                                    "capped": trees == rounds, "note": "same depth and bootstrap as 450-round prior; owner-disjoint early stop then all-fold2 refit"}
        summary["seconds"][name] = round(time.monotonic() - tic, 2)
        print(name, "seconds", summary["seconds"][name], "trees", trees, flush=True)
        del full

    pl.DataFrame({"qid": qv, "tid": tv, "label": yv, **predictions}).write_parquet(dest / "validation_predictions.parquet", compression="zstd")
    summary["seconds"]["total"] = round(time.monotonic() - start, 2)
    summary["peak_process_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)
    summary["limitations"].append("cat_depth4_1400 hit its round cap; cat_depth6_bag_1400 peaked eight rounds before cap, so neither fit establishes complete convergence")
    doc.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# extended overnight cpu screening", "", "same cached safe 54-feature lexical selected-query pairs, fold2 owner-disjoint early stop and full-fold2 refit. first-sweep predictions reused, not retrained. fold0 only; no full-pool or leaderboard result.", "",
             "| gate | top3 retention | recall at precision ≥ .995 | neural top3 blend recall at precision ≥ .995 | seconds | trees |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, scores in results.items():
        def recall(key):
            point = scores[key]["0.995"]
            return f"{point['recall_all_selected_positive_owners']:.4f}" if point else "n/a"
        model = summary["models"][name]
        lines.append(f"| {name} | {scores['gate_top3_recall']:.4f} | {recall('gate_high_precision')} | {recall('neural_top3_blend_high_precision')} | {summary['seconds'].get(name, 'reused')} | {model.get('internal_best_iteration', model.get('best_iteration', 'reused'))} |")
    lines += ["", "**outcome:** catboost depth6 improved from .8947 to .9087 recall at .995 precision (neural blend .9275 to .9323) but remained below the reused safe lgb (.9121; blend .9332). depth4 improved .8520 to .8597 and hit 1400 rounds; it is still budget-limited. both weighted lgb variants also remained below the safe gate. none merits promotion from this sampled screen.",
              "", f"positive-target denominator {results['existing_safe_lgb']['positive_target_denominator']}; candidate recall {results['existing_safe_lgb']['candidate_recall']:.4f}; total {summary['seconds']['total']}s; peak process rss {summary['peak_process_rss_gib']} gib.",
              "", "these precision cutoffs were selected on the same sampled fold0 pairs and are not independently audited. no owner macro or stack is valid on this population. fold1 source1 ids appear only as negative competitors; no fold1 target labels were used.",
              "", "run from repo root: `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12 .venv/bin/python -u src/tune.py --extended --threads 12 --rounds 1400 > artifacts/overnight-tune-extended/run.log 2>&1`.",
              "", "parameters, input hashes and all operating points: [json](overnight-tune-extended.json); models and pair predictions: `artifacts/overnight-tune-extended/`."]
    doc.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("extended complete", summary["seconds"], "peak gib", summary["peak_process_rss_gib"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--rounds", type=int)
    args = parser.parse_args()
    threads = args.threads if args.threads is not None else (12 if args.extended else 4)
    rounds = args.rounds if args.rounds is not None else (1400 if args.extended else 450)
    if args.check:
        check()
    elif args.extended:
        extend(threads, rounds)
    else:
        run(threads, rounds)
