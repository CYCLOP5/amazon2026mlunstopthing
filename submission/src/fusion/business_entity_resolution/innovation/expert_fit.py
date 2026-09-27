'fit and evaluate cpu heads using frozen expert logits on a fixed pair pool'
import gc
import json
from pathlib import Path

import numpy as np
import polars as pl

from er.stack import decode
from er.stack.features import feature_names
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import export, predict, tuning_anchors
from graph_resolution.pipeline import compare_audit, subset_for_tuning
from innovation.common import log, logit, read_parent, replace_scores
from innovation.train import grouped_fit, training_partitions


EXPERT_FEATURES = ("expert_lg", "expert_rank", "expert_margin")
NO_OTHER = -14.0


def join_expert_scores(features: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    'join one finite expert logit to every prepared pair, rejecting pool drift'
    keys = ["qid", "tid"]
    if not set(keys + ["expert_lg"]) <= set(scores.columns):
        raise ValueError("Expert score parts require qid, tid, and expert_lg")
    if features.select(*keys).n_unique() != features.height:
        raise ValueError("Prepared pair table contains duplicate qid/tid pairs")
    score = scores.select(*keys, "expert_lg").with_columns(
        pl.col("qid").cast(features.schema["qid"]), pl.col("tid").cast(features.schema["tid"]),
        pl.col("expert_lg").cast(pl.Float32))
    expected = features.select(pl.col("qid").cast(pl.Int64), pl.col("tid").cast(pl.Int64))
    score_keys = score.select(pl.col("qid").cast(pl.Int64), pl.col("tid").cast(pl.Int64))
    if score.select(*keys).n_unique() != score.height:
        raise ValueError("Expert score parts contain duplicate qid/tid pairs")
    missing = expected.join(score_keys, on=keys, how="anti").height
    extra = score_keys.join(expected, on=keys, how="anti").height
    if missing or extra or score.height != features.height:
        raise ValueError(f"Expert score coverage mismatch: missing={missing}, extra={extra}, "
                         f"scores={score.height}, prepared={features.height}")
    if score["expert_lg"].null_count() or not np.isfinite(score["expert_lg"].to_numpy()).all():
        raise ValueError("Expert logits must all be finite and non-null")
    out = features.join(score, on=keys, how="left", maintain_order="left", validate="1:1")
    if out.height != features.height or out["expert_lg"].null_count():
        raise ValueError("Expert score join did not preserve prepared pair coverage")
    return out


def add_competition_features(frame: pl.DataFrame) -> pl.DataFrame:
    'add target-local expert rank and best-other margin from logits alone'
    if "expert_lg" not in frame.columns:
        raise ValueError("Missing expert_lg")
    if frame.select("qid", "tid").n_unique() != frame.height:
        raise ValueError("Duplicate pair before expert competition features")
    values = frame["expert_lg"].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Expert logits must be finite")
    top = frame.group_by("tid").agg(
        pl.col("expert_lg").top_k(2).alias("_expert_top"))
    out = frame.join(top, on="tid", how="left", maintain_order="left").with_columns(
        pl.col("_expert_top").list.get(0).alias("_expert_best"),
        pl.col("_expert_top").list.get(1, null_on_oob=True).fill_null(NO_OTHER).alias("_expert_second"),
        pl.col("expert_lg").rank("min", descending=True).over("tid").cast(pl.Float32).alias("expert_rank"))
    out = out.with_columns(
        pl.when(pl.col("expert_lg") >= pl.col("_expert_best"))
        .then(pl.col("expert_lg") - pl.col("_expert_second"))
        .otherwise(pl.col("expert_lg") - pl.col("_expert_best"))
        .cast(pl.Float32).alias("expert_margin"))
    return out.drop("_expert_top", "_expert_best", "_expert_second")


def load_prepared_with_expert(prepared, expert, split):
    'load prepared pairs and complete score shards, checking the exact key set'
    root = Path(prepared)
    if not (root / "_SUCCESS").exists():
        raise RuntimeError("Prepared pair features are incomplete")
    score_root = Path(expert)
    if not (score_root / "_SUCCESS").exists():
        raise RuntimeError("Expert scoring has not completed")
    report_path = score_root / "expert_report.json"
    if not report_path.exists():
        raise RuntimeError("Missing expert_report.json")
    paths = sorted((score_root / split).glob("part_*.parquet"))
    if not paths:
        raise ValueError(f"No expert score parts for {split}")
    scores = pl.concat([pl.read_parquet(path) for path in paths], how="vertical_relaxed")
    base = pl.read_parquet(root / split / "features.parquet")
    return add_competition_features(join_expert_scores(base, scores)), paths


def _settings(rounds):
    return {"cv": 3, "seed": 42, "fit_fold": 0, "audit_fold": 1, "tune_holdout_buckets": 5,
        "fit": {"num_boost_round": int(rounds), "early_stopping_rounds": 80},
        "lgbm": {"learning_rate": .04, "num_leaves": 63, "min_data_in_leaf": 80,
                 "lambda_l2": 5., "feature_fraction": .9, "deterministic": True,
                 "force_col_wise": True, "seed": 42}}


def _metric_country_deltas(accepted, reference, anchors, countries):
    res = {}
    for country in countries:
        sub = anchors.filter(pl.col("co") == country).select("qid", "deg")
        res[str(country)] = decode.score(accepted, sub)["macro_f05"] - decode.score(reference, sub)["macro_f05"]
    return res


def blend_predictions(frame: pl.DataFrame, kind: str, weight: float) -> pl.DataFrame:
    'Create a calibrated head/parent logit blend for prepared pairs'
    col = "control_head" if kind == "control" else "expert_head"
    if col not in frame.columns or "parent_p" not in frame.columns:
        raise ValueError(f"Missing blend inputs for {kind}")
    p = 1. / (1. + np.exp(-np.clip((1. - weight) * logit(frame["parent_p"].to_numpy()) +
                                    weight * logit(frame[col].to_numpy()), -30., 30.)))
    return frame.select("qid", "tid").with_columns(pl.Series("p", p.astype(np.float32)))


def _fit_columns(frame):
    base = [c for c in feature_names(frame.columns)
            if c not in {"own", "co", "parent_p", *EXPERT_FEATURES}
            and not c.startswith("_")]
    return base, base + list(EXPERT_FEATURES)


def run(data, parent, prepared, expert, output, rounds=1000):
    'Fit control/expert CPU heads, tune safe blends, audit, and export full pools'
    output, prepared, expert = Path(output), Path(prepared), Path(expert)
    bundle = output / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    if not (prepared / "_SUCCESS").exists() or not (expert / "_SUCCESS").exists():
        raise RuntimeError("Preparation and expert scoring must both finish before fitting")
    expert_report_path = expert / "expert_report.json"
    if not expert_report_path.exists():
        raise RuntimeError("Missing expert_report.json")
    expert_report = json.loads(expert_report_path.read_text())
    tr, train_parts = load_prepared_with_expert(prepared, expert, "train")
    if not {"own", "fold", "y", "co", "parent_p"} <= set(tr.columns):
        raise ValueError("Prepared train features lack labels, folds, owner, country, or parent score")
    base_cols, expert_cols = _fit_columns(tr)
    params = _settings(rounds)
    fit, groups = training_partitions(tr, load_refs(data, "train"), params["cv"])
    if fit.shape != (tr.height,) or not fit.any():
        raise ValueError("No eligible owner-grouped fit rows")
    models, predictions, training = {}, {}, {}
    for name, columns in (("control", base_cols), ("expert", expert_cols)):
        pred, fitted, info = grouped_fit(tr, columns, fit, groups, params, name + " expert-calibration head")
        models[name], predictions[name], training[name] = fitted, pred, info
        for i, model in enumerate(fitted):
            model.save_model(str(bundle / f"{name}_{i}.txt"))
    (bundle / "features.json").write_text(json.dumps({"control": base_cols, "expert": expert_cols}, indent=2))
    tr = tr.with_columns(pl.Series("control_head", predictions["control"], dtype=pl.Float32),
                         pl.Series("expert_head", predictions["expert"], dtype=pl.Float32))

    parent_train, parent_report = read_parent(parent, "train")
    parent_decision = parent_report["selected_decoder"]
    refs = load_refs(data, "train")
    anchors = refs.select(pl.col("rid").alias("qid"), "fold", "deg", "co")
    tune = tuning_anchors(anchors, params)
    audit = anchors.filter(pl.col("fold") == 1).select("qid", "deg")
    parent_accept = decode.apply(parent_decision["rule"], parent_train, parent_decision["threshold"])
    tune_pool = subset_for_tuning(parent_train, tune)
    country_tune = tune.join(anchors.select("qid", "co"), on="qid", how="left")
    tuning = {"parent": {"kind": "parent", "weight": 0., "rule": parent_decision["rule"],
        "threshold": parent_decision["threshold"], "tune": decode.score(parent_accept, tune),
        "tune_country_delta_vs_parent": {str(c): 0. for c in country_tune["co"].unique().sort().to_list()},
        "eligible": True}}
    for kind in ("control", "expert"):
        for weight in (.25, .5, 1.):
            update = blend_predictions(tr, kind, weight)
            patched = replace_scores(tune_pool, update)
            decision = decode.tune(patched, tune, rules=("top1_threshold", "expected_f"), floors=(.3, .5))
            accepted = decode.apply(decision["rule"], patched, decision["threshold"])
            key = f"{kind}_{weight:g}"
            deltas = _metric_country_deltas(accepted, parent_accept, country_tune,
                                            country_tune["co"].unique().sort().to_list())
            tuning[key] = {"kind": kind, "weight": weight, "rule": decision["rule"],
                "threshold": decision["threshold"], "tune": decision,
                "tune_country_delta_vs_parent": deltas}
            log(f"Expert calibration {key}: tune F0.5 {decision['macro_f05']:.7f}; country delta {deltas}")
            del patched, accepted, update

    parent_tune_score = tuning["parent"]["tune"]["macro_f05"]
    for key in (k for k in tuning if k.startswith("control_")):
        gain = tuning[key]["tune"]["macro_f05"] - parent_tune_score
        country_delta = tuning[key]["tune_country_delta_vs_parent"]
        tuning[key]["promotion_gain_vs_parent"] = gain
        tuning[key]["eligible"] = (gain >= .0001 and
                                    min(country_delta.values(), default=0.) >= -.0002)
    all_control_keys = ["parent"] + [k for k in tuning if k.startswith("control_")]
    eligible_control_keys = ["parent"] + [k for k in tuning if k.startswith("control_") and tuning[k]["eligible"]]
    best_control = max(eligible_control_keys, key=lambda k: tuning[k]["tune"]["macro_f05"])
    control_score_ceiling_key = max(all_control_keys, key=lambda k: tuning[k]["tune"]["macro_f05"])
    control_score_ceiling = tuning[control_score_ceiling_key]["tune"]["macro_f05"]
    if best_control == "parent":
        control_accept = parent_accept
    else:
        base_kind = tuning[best_control]["kind"]
        base_weight = tuning[best_control]["weight"]
        base_patch = replace_scores(tune_pool, blend_predictions(tr, base_kind, base_weight))
        control_accept = decode.apply(tuning[best_control]["rule"], base_patch,
                                      tuning[best_control]["threshold"])
        del base_patch
    for key in (k for k in tuning if k.startswith("expert_")):
        gain = tuning[key]["tune"]["macro_f05"] - control_score_ceiling
        candidate_patch = replace_scores(tune_pool, blend_predictions(tr, "expert", tuning[key]["weight"]))
        candidate_accept = decode.apply(tuning[key]["rule"], candidate_patch, tuning[key]["threshold"])
        country_delta_control = _metric_country_deltas(candidate_accept, control_accept, country_tune,
                                                       country_tune["co"].unique().sort().to_list())
        country_delta_parent = tuning[key]["tune_country_delta_vs_parent"]
        tuning[key]["tune_country_delta_vs_selected_control"] = country_delta_control
        tuning[key]["promotion_gain_vs_control_ceiling"] = gain
        tuning[key]["eligible"] = (gain >= .0001 and
            min(country_delta_parent.values(), default=0.) >= -.0002 and
            min(country_delta_control.values(), default=0.) >= -.0002)
        del candidate_patch, candidate_accept
    del control_accept
    eligible_expert = [k for k in tuning if k.startswith("expert_") and tuning[k]["eligible"]]
    winner = max(eligible_expert, key=lambda k: tuning[k]["tune"]["macro_f05"]) if eligible_expert else best_control
    decision = tuning[winner]
    if winner == "parent":
        final_train = parent_train
    else:
        final_train = replace_scores(parent_train, blend_predictions(tr, decision["kind"], decision["weight"]))
    accepted = decode.apply(decision["rule"], final_train, decision["threshold"])
    audit_by_country = {}
    for country in anchors.filter(pl.col("fold") == 1)["co"].unique().sort():
        sub = anchors.filter((pl.col("fold") == 1) & (pl.col("co") == country)).select("qid", "deg")
        audit_by_country[str(country)] = {"parent": decode.score(parent_accept, sub),
                                          "selected": decode.score(accepted, sub)}
    comparison = compare_audit(accepted, parent_accept, audit)
    comparison["reference"] = "bold_zebra graph sibling refinement with frozen decoder"
    report = {"selected": winner, "decision": decision, "best_control": best_control,
        "control_score_ceiling": {"trial": control_score_ceiling_key, "macro_f05": control_score_ceiling},
        "promotion_policy": {"controls": "promote only if tune gain vs parent >= 0.0001 and every tune-country delta vs parent >= -0.0002",
            "expert": "promote only if tune gain vs highest parent/control score (including ineligible controls) >= 0.0001 and every country delta vs parent and selected eligible control >= -0.0002",
            "selection_uses_audit": False},
        "tuning": tuning, "training": training, "audit": decode.score(accepted, audit),
        "parent_audit": decode.score(parent_accept, audit), "audit_by_country": audit_by_country,
        "comparison": comparison, "expert_report": expert_report,
        "expert_parts": {"train": len(train_parts)},
        "limits": ["Tune and audit populations contain labeled US/India only; French transfer is unmeasured.",
                   "Fold 1 has informed previous experiments and is not a fresh blind estimate.",
                   "Expert logits cover only selected prepared pairs; unscored parent candidates retain their original scores."]}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (bundle / "decision.json").write_text(json.dumps(decision, indent=2))
    final_train.write_parquet(output / "validation_predictions.parquet")
    log(f"EXPERT FIT audit {report['audit']}; selected={winner}; paired={comparison}")

    del tr, parent_train, final_train, accepted, parent_accept, predictions
    gc.collect()
    te, test_parts = load_prepared_with_expert(prepared, expert, "test")
    parent_test, _ = read_parent(parent, "test")
    if winner == "parent":
        final_test = parent_test
    else:
        p = predict(models[decision["kind"]], te, expert_cols if decision["kind"] == "expert" else base_cols)
        te = te.with_columns(pl.Series("control_head" if decision["kind"] == "control" else "expert_head",
                                       p, dtype=pl.Float32))
        final_test = replace_scores(parent_test, blend_predictions(te, decision["kind"], decision["weight"]))
    final_test.write_parquet(output / "test_predictions.parquet")
    report["expert_parts"]["test"] = len(test_parts)
    report["export"] = export(final_test, "p", decision["rule"], decision["threshold"],
        load_refs(data, "test"), load_targets(data, "test"), str(output / "output"))
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "_SUCCESS").write_text("complete\n")
    log("Expert calibration complete")
    return report
