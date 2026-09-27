#!/usr/bin/env python3
"""read-only gate and neural gap diagnostic for cached fold-0 rows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path as path

for thread_env in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
    os.environ[thread_env] = "4"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))

import numpy as np
import polars as pl


root = path(__file__).resolve().parents[1]
bench = root / "artifacts/teammate-benchmark"
data = root / "cache/data"
out = root / "artifacts/overnight-gap"
safe_score = "lgb_a_no_s1_aggregate"
cuts = (0.50, 0.80, 0.90, 0.95, 0.99)
precision_targets = (0.990, 0.995, 0.999)
weights = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def top_k(frame: pl.DataFrame, score: str, k: int | None) -> pl.DataFrame:
    """return deterministic per-target rank prefixes."""
    ranked = frame.sort(["tid", score, "qid"], descending=[False, True, False])
    return ranked if k is None else ranked.group_by("tid", maintain_order=True).head(k)


def blend(neural: np.ndarray, gate: np.ndarray, weight: float) -> np.ndarray:
    """match the production weighted-logit blend without importing training code."""
    if weight == 0:
        return gate.astype(np.float32, copy=False)
    if weight == 1:
        return neural.astype(np.float32, copy=False)
    eps = np.finfo(np.float32).eps
    neural, gate = (np.clip(x, eps, 1 - eps) for x in (neural, gate))
    logits = weight * (np.log(neural) - np.log1p(-neural))
    logits += (1 - weight) * (np.log(gate) - np.log1p(-gate))
    return (1 / (1 + np.exp(-logits))).astype(np.float32, copy=False)


def metric_at(top1: pl.DataFrame, truth: pl.DataFrame, score: str, cut: float) -> dict:
    """score one prediction per target against known linked and empty targets."""
    z = top1.join(truth, on="tid", how="inner", validate="1:1").with_columns(
        (pl.col(score) >= cut).alias("pred"),
        ((pl.col(score) >= cut) & (pl.col("label") == 1)).alias("tp"),
    )
    predicted = int(z["pred"].sum())
    tp = int(z["tp"].sum())
    linked = int(z["linked"].sum())
    fp, fn = predicted - tp, linked - tp
    per_target = np.where(
        z["linked"].to_numpy(),
        1.25 * z["tp"].to_numpy() / (z["pred"].to_numpy() + 0.25),
        (~z["pred"].to_numpy()).astype(float),
    )
    return {
        "cut": cut,
        "target_macro_f05_diagnostic": float(per_target.mean()),
        "selected_pairs": predicted,
        "true_positive_pairs": tp,
        "false_positive_pairs": fp,
        "false_negative_pairs": fn,
        "pair_precision": tp / predicted if predicted else 1.0,
        "pair_recall_known_linked_targets": tp / linked if linked else 1.0,
        "pair_f05_diagnostic": 1.25 * tp / (1.25 * tp + fp + .25 * fn) if tp else 0.0,
    }


def matched_precision(top1: pl.DataFrame, truth: pl.DataFrame, score: str) -> dict:
    """describe complete-score cut points; do not select a deployment cutoff."""
    z = top1.join(truth, on="tid", how="inner", validate="1:1").sort(
        [score, "qid"], descending=[True, False]
    )
    scores = z[score].to_numpy()
    labels = z["label"].to_numpy().astype(np.int64, copy=False)
    linked = int(z["linked"].sum())
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    selected = ends + 1
    tp = np.cumsum(labels)[ends]
    precision = tp / selected
    result = {"linked_targets": linked}
    for target in precision_targets:
        valid = np.flatnonzero(precision >= target)
        if not len(valid):
            result[f"{target:.3f}"] = None
            continue
        best = valid[np.argmax(tp[valid])]
        point = metric_at(top1, truth, score, float(scores[ends[best]]))
        result[f"{target:.3f}"] = point
    return result


def evaluate(name: str, ranked: pl.DataFrame, truth: pl.DataFrame, score: str, scope: str) -> dict:
    top1 = top_k(ranked, score, 1)
    return {
        "scope": scope,
        "top1_rows": len(top1),
        "fixed_cuts": {f"{cut:.2f}": metric_at(top1, truth, score, cut) for cut in cuts},
        "matched_precision_descriptive_only": matched_precision(top1, truth, score),
        "name": name,
    }


def load_targets(pairs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    tids, qids = pairs["tid"].unique(), pairs["qid"].unique()
    target_columns = ["rid", "co", "nm", "ad", "nn", "an", "own", "sr"]
    target_scan = pl.concat(
        [
            pl.scan_parquet(data / "train/s2.parquet").select(target_columns),
            pl.scan_parquet(data / "train/s3.parquet").select(target_columns),
        ]
    ).filter(pl.col("rid").is_in(tids.implode()))
    targets = target_scan.collect(engine="streaming").rename({"rid": "tid"})
    refs = pl.scan_parquet(data / "train/ref.parquet").select(["rid", "nn", "an"]).filter(
        pl.col("rid").is_in(qids.implode())
    ).collect(engine="streaming").rename({"rid": "qid", "nn": "q_nn", "an": "q_an"})
    if targets["tid"].n_unique() != len(targets) or refs["qid"].n_unique() != len(refs):
        raise ValueError("source identifiers are not unique")
    if len(targets) != tids.len() or len(refs) != qids.len():
        raise ValueError("cached source data does not cover every diagnostic pair")
    return targets, refs


def artifact_consistency(pairs: pl.DataFrame, shortlist: pl.DataFrame) -> dict:
    """inspect the supplied stored top-3 file without trusting it as input."""
    stored = pl.read_parquet(bench / "neural_gate_blended_top3.parquet")
    old = stored.filter(pl.col("variant") == safe_score).select("tid", "qid", "label", "prob", "gate", "blend")
    new = shortlist.select("tid", "qid", "label", "prob", "gate", "blend")
    joined = old.join(new, on=["tid", "qid"], how="inner", suffix="_new", validate="1:1")
    max_blend_delta = float((joined["blend"] - joined["blend_new"]).abs().max()) if len(joined) else None
    return {
        "paired_rows": len(pairs),
        "stored_rows": len(stored),
        "stored_variants": sorted(stored["variant"].unique().to_list()),
        "safe_gate_rows": len(old),
        "recomputed_safe_gate_rows": len(new),
        "joined_safe_gate_rows": len(joined),
        "max_abs_blend_delta_at_weight_0_6": max_blend_delta,
        "consistent": len(old) == len(new) == len(joined) and (max_blend_delta is not None and max_blend_delta < 1e-6),
    }


def summary(frame: pl.DataFrame) -> dict:
    n = len(frame)
    flags = ("candidate_miss", "gate_top3_loss", "wrong_top1", "cutoff_miss_080", "wrong_top1_selected_080")
    return {"targets": n, **{flag: int(frame[flag].sum()) for flag in flags}, **{
        f"{flag}_rate": float(frame[flag].mean()) if n else 0.0 for flag in flags
    }}


def error_patterns(
    pairs: pl.DataFrame, truth: pl.DataFrame, shortlist: pl.DataFrame, blend06: pl.DataFrame,
    targets: pl.DataFrame, refs: pl.DataFrame,
) -> tuple[dict, pl.DataFrame]:
    positive = pairs.group_by("tid").agg((pl.col("label").sum() > 0).alias("candidate_positive"))
    retained = shortlist.group_by("tid").agg((pl.col("label").sum() > 0).alias("top3_positive"))
    top1 = top_k(blend06, "score", 1).select("tid", "qid", "label", "score")
    z = truth.join(positive, on="tid", how="left").join(retained, on="tid", how="left").join(
        top1, on="tid", how="left", validate="1:1"
    ).join(targets, on="tid", how="inner", validate="1:1").join(refs, on="qid", how="inner", validate="m:1").with_columns(
        pl.col("candidate_positive").fill_null(False),
        pl.col("top3_positive").fill_null(False),
        pl.col("nm").fill_null("").str.strip_chars().eq("").alias("empty_name"),
        pl.col("ad").fill_null("").str.strip_chars().eq("").alias("blank_address"),
    ).with_columns(
        (pl.col("linked") & ~pl.col("candidate_positive")).alias("candidate_miss"),
        (pl.col("linked") & pl.col("candidate_positive") & ~pl.col("top3_positive")).alias("gate_top3_loss"),
        (pl.col("linked") & pl.col("top3_positive") & (pl.col("label") == 0)).alias("wrong_top1"),
        (pl.col("linked") & (pl.col("label") == 1) & (pl.col("score") < 0.80)).alias("cutoff_miss_080"),
        ((pl.col("label") == 0) & (pl.col("score") >= 0.80)).alias("wrong_top1_selected_080"),
        ((pl.col("nn") != "") & (pl.col("nn") == pl.col("q_nn"))).alias("same_normalized_name"),
        ((pl.col("an") != "") & (pl.col("an") == pl.col("q_an"))).alias("same_normalized_address"),
    )
    by_country = [{"country": row[0], **summary(group)} for row, group in z.group_by("co", maintain_order=True)]
    by_field = []
    for field in ("empty_name", "blank_address"):
        for value, group in z.group_by(field, maintain_order=True):
            by_field.append({"field": field, "value": bool(value[0]), **summary(group)})
    wrong = z.filter(pl.col("wrong_top1"))
    hard = {
        "wrong_top1_rows": len(wrong),
        "same_normalized_name": int(wrong["same_normalized_name"].sum()),
        "same_normalized_address": int(wrong["same_normalized_address"].sum()),
        "same_name_different_address": int((wrong["same_normalized_name"] & ~wrong["same_normalized_address"]).sum()),
        "same_address_different_name": int((wrong["same_normalized_address"] & ~wrong["same_normalized_name"]).sum()),
    }
    keep = ["tid", "qid", "co", "sr", "linked", "label", "score", "empty_name", "blank_address",
            "same_normalized_name", "same_normalized_address", "candidate_miss", "gate_top3_loss", "wrong_top1",
            "cutoff_miss_080", "wrong_top1_selected_080"]
    return {"overall": summary(z), "by_country": by_country, "by_field": by_field, "hard_negative_flags": hard}, z.filter(
        pl.any_horizontal([pl.col(flag) for flag in ("candidate_miss", "gate_top3_loss", "wrong_top1", "cutoff_miss_080", "wrong_top1_selected_080")])
    ).select(keep)


def report_md(result: dict) -> str:
    ceiling, models, errors = result["ceiling"], result["models"], result["errors"]
    rows = [
        "# overnight gap diagnostic",
        "",
        "all metrics below use only the supplied selected fold-0 cached rows. they are diagnostic evidence, not full-pool validation or a deployable threshold selection.",
        "",
        "## reported public facts",
        "",
        "- baseline public score: 0.964 (team-reported)",
        "- upgraded public score: 0.969 (team-reported)",
        "- stale local reports are not treated as contradicting those team-reported facts.",
        "",
        "## causal gap",
        "",
        f"- {ceiling['linked_targets']:,} linked targets and {ceiling['empty_targets']:,} empty targets are represented.",
        f"- the fixed candidate universe contains the true pair for {ceiling['candidate_oracle_pair_recall']:.4%} of linked targets; the remaining {ceiling['candidate_miss_count']:,} cannot be recovered by any gate, blend, or cutoff.",
        f"- the safe gate retains {ceiling['safe_gate_retention']['top3']:.4%} at top-3; {errors['overall']['gate_top3_loss']:,} otherwise recoverable true pairs are removed before the neural score can rank them.",
        f"- at the existing descriptive w=0.6 blend, {errors['overall']['wrong_top1']:,} targets have a true pair in the safe top-3 but a wrong top-1. at cut 0.80, {errors['overall']['cutoff_miss_080']:,} correct top-1 pairs fall below the cut and {errors['overall']['wrong_top1_selected_080']:,} wrong top-1 pairs remain selected.",
        "",
        "## score comparisons",
        "",
        "the target-macro diagnostic averages one selected-row decision per target, including correct empty-target decisions. it is not source1 macro and cannot be compared to the leaderboard metric. pair f0.5 is reported separately from aggregate selected-pair tp, fp, and fn; it is also diagnostic-only.",
        "",
        "| score | candidate scope | target-macro diagnostic @ .80 | pair f0.5 diagnostic @ .80 | precision @ .80 | recall @ .80 | recall at >= .995 precision |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, model in models.items():
        fixed = model["fixed_cuts"]["0.80"]
        point = model["matched_precision_descriptive_only"]["0.995"]
        recall = "n/a" if point is None else f"{point['pair_recall_known_linked_targets']:.4f}"
        rows.append(f"| {name} | {model['scope']} | {fixed['target_macro_f05_diagnostic']:.4f} | {fixed['pair_f05_diagnostic']:.4f} | {fixed['pair_precision']:.4f} | {fixed['pair_recall_known_linked_targets']:.4f} | {recall} |")
    rows.extend([
        "",
        "## error strata",
        "",
        "| country | targets | candidate miss | top-3 loss | wrong top-1 | cutoff miss @ .80 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in sorted(errors["by_country"], key=lambda value: value["targets"], reverse=True):
        rows.append(f"| {row['country']} | {row['targets']:,} | {row['candidate_miss']:,} | {row['gate_top3_loss']:,} | {row['wrong_top1']:,} | {row['cutoff_miss_080']:,} |")
    rows.extend([
        "",
        "## field strata",
        "",
        "| field | value | targets | candidate miss | top-3 loss | wrong top-1 | cutoff miss @ .80 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in errors["by_field"]:
        rows.append(f"| {row['field']} | {str(row['value']).lower()} | {row['targets']:,} | {row['candidate_miss']:,} | {row['gate_top3_loss']:,} | {row['wrong_top1']:,} | {row['cutoff_miss_080']:,} |")
    empty = [row for row in errors["by_field"] if row["field"] == "empty_name" and row["value"]]
    if not empty:
        rows.append("\nno target had an empty name in this population, so an empty-name error rate is not estimable here.")
    hard = errors["hard_negative_flags"]
    rows.extend([
        "",
        f"among {hard['wrong_top1_rows']:,} wrong top-1 rows with a retained true pair, {hard['same_normalized_name']:,} share a normalized name and {hard['same_normalized_address']:,} share a normalized address with the wrong candidate. this is an exact-normalized-field hard-negative flag, not a semantic error taxonomy.",
        "",
        "## limits",
        "",
        "- no model, stack, blend weight, or threshold was fit or selected. all six weights and all cut points are same-label descriptive comparisons.",
        "- labels are available only for this sampled/selected fold-0 population. no full-pool metric is claimed.",
        "- the safe top-3 is upstream of the cross-encoder blend; a later score cannot recover a gate-pruned true pair.",
        "",
    ])
    return "\n".join(rows)


def check() -> None:
    toy = pl.DataFrame({"tid": [1, 1, 2, 3], "qid": [2, 1, 3, 4], "label": [0, 1, 0, 1], "score": [.9, .9, .4, .1]})
    truth = pl.DataFrame({"tid": [1, 2, 3], "linked": [True, False, True]})
    best = top_k(toy, "score", 1)
    assert best.sort("tid")["qid"].to_list() == [1, 3, 4]
    metric = metric_at(best, truth, "score", .5)
    assert metric["true_positive_pairs"] == 1 and metric["selected_pairs"] == 1
    assert np.isclose(metric["target_macro_f05_diagnostic"], 2 / 3)
    assert np.isclose(metric["pair_f05_diagnostic"], 5 / 6)
    assert np.array_equal(blend(np.array([.2], dtype=np.float32), np.array([.8], dtype=np.float32), 0), np.array([.8], dtype=np.float32))
    assert np.array_equal(blend(np.array([.2], dtype=np.float32), np.array([.8], dtype=np.float32), 1), np.array([.2], dtype=np.float32))
    print("gap checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    out.mkdir(parents=True, exist_ok=True)
    pairs = pl.read_parquet(bench / "neural_gate_paired_scores.parquet").select(
        "qid", "tid", "label", "prob", safe_score
    ).rename({safe_score: "gate"})
    if pairs.select("qid", "tid").n_unique() != len(pairs):
        raise ValueError("paired scores are not unique by candidate pair")
    targets, refs = load_targets(pairs)
    positive = pairs.group_by("tid").agg((pl.col("label").sum() > 0).alias("candidate_positive"))
    truth = targets.select("tid", (pl.col("own") >= 0).alias("linked")).join(positive, on="tid", how="left").with_columns(
        pl.col("candidate_positive").fill_null(False)
    )
    if int((truth["candidate_positive"] & ~truth["linked"]).sum()):
        raise ValueError("a labeled positive is attached to an empty target")
    shortlist = top_k(pairs, "gate", 3)
    shortlist = shortlist.with_columns(pl.Series("blend", blend(shortlist["prob"].to_numpy(), shortlist["gate"].to_numpy(), .6)))
    consistency = artifact_consistency(pairs, shortlist)
    all_retained = int(pairs["label"].sum())
    linked = int(truth["linked"].sum())
    retentions = {"top1": int(top_k(pairs, "gate", 1)["label"].sum()) / linked,
                  "top2": int(top_k(pairs, "gate", 2)["label"].sum()) / linked,
                  "top3": int(shortlist["label"].sum()) / linked,
                  "all": all_retained / linked}
    models = {
        "cross_encoder_all_candidates": evaluate("cross_encoder_all_candidates", pairs, truth, "prob", "all fixed candidates"),
        "safe_gate_top1": evaluate("safe_gate_top1", pairs, truth, "gate", "all fixed candidates"),
    }
    blend_frames = {}
    for weight in weights:
        score = f"blend_w_{weight:.1f}"
        frame = shortlist.with_columns(pl.Series("score", blend(shortlist["prob"].to_numpy(), shortlist["gate"].to_numpy(), weight)))
        name = f"safe_top3_logit_blend_w_{weight:.1f}"
        models[name] = evaluate(name, frame, truth, "score", "safe-gate top-3")
        blend_frames[weight] = frame
    errors, error_rows = error_patterns(pairs, truth, shortlist, blend_frames[.6], targets, refs)
    ceiling = {
        "targets": len(truth), "linked_targets": linked, "empty_targets": len(truth) - linked,
        "candidate_true_pairs": all_retained, "candidate_miss_count": linked - all_retained,
        "candidate_oracle_pair_recall": all_retained / linked,
        "candidate_oracle_target_macro_f05_diagnostic": (int((~truth["linked"]).sum()) + all_retained) / len(truth),
        "safe_gate_retention": retentions,
    }
    result = {
        "scope": "read-only selected fold-0 cached diagnostic; not full-pool validation",
        "constraints": {"cpu_worker_caps": 4, "cpu_affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None, "ram_target_gib_lt": 3, "writes": ["artifacts/overnight-gap", "reports/overnight-gap.json", "reports/overnight-gap.md"]},
        "team_reported_public_scores": {"baseline": 0.964, "upgraded": 0.969, "verification": "accepted team-reported facts; local reports may be stale"},
        "inputs": {"paired_scores": str(bench / "neural_gate_paired_scores.parquet"), "stored_top3": str(bench / "neural_gate_blended_top3.parquet"), "validation_scores": str(bench / "validation_scores.parquet"), "cache_data": str(data)},
        "artifact_consistency": consistency,
        "ceiling": ceiling,
        "models": models,
        "errors": errors,
        "selection_policy": "no stacking, meta-fit, blend-weight selection, or threshold selection was performed; all labeled comparisons are descriptive only",
    }
    text = json.dumps(result, indent=2) + "\n"
    (out / "metrics.json").write_text(text, encoding="utf-8")
    error_rows.write_parquet(out / "error_rows.parquet", compression="zstd")
    (root / "reports/overnight-gap.json").write_text(text, encoding="utf-8")
    (root / "reports/overnight-gap.md").write_text(report_md(result), encoding="utf-8")
    print(json.dumps({"ceiling": ceiling, "artifact_consistency": consistency, "error_rows": len(error_rows)}, indent=2))


if __name__ == "__main__":
    main()
