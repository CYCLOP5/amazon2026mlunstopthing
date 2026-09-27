#!/usr/bin/env python3
'cpu-only cached gate-width diagnostic; not a production policy selector'
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _key in ("POLARS_MAX_THREADS", "ARROW_NUM_THREADS"):
    os.environ[_key] = "6"
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:6]))

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "artifacts/teammate-benchmark"
DATA = ROOT / "cache/data"
OUT = ROOT / "artifacts/overnight-width"
SAFE_SCORE = "lgb_a_no_s1_aggregate"
PRECISION_TARGETS = (.990, .995, .999)
FIXED_WIDTHS = (1, 2, 3, 5, 10, 20)


def top_k(frame: pl.DataFrame, width: int) -> pl.DataFrame:
    'return deterministic safe-gate prefixes, matching the benchmark tie rule'
    return frame.sort(["tid", "gate", "qid"], descending=[False, True, False]).group_by(
        "tid", maintain_order=True
    ).head(width)


def blend(neural: np.ndarray, gate: np.ndarray) -> np.ndarray:
    'use the saved w=.6 weighted-logit blend without loading a model'
    eps = np.finfo(np.float32).eps
    neural, gate = (np.clip(x, eps, 1 - eps) for x in (neural, gate))
    logits = np.float32(.6) * (np.log(neural) - np.log1p(-neural))
    logits += np.float32(.4) * (np.log(gate) - np.log1p(-gate))
    return (np.float32(1) / (np.float32(1) + np.exp(-logits))).astype(np.float32, copy=False)


def top_one(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(["tid", "blend", "qid"], descending=[False, True, False]).group_by(
        "tid", maintain_order=True
    ).head(1)


def matched_precision(top1: pl.DataFrame, linked: int) -> dict:
    'describe same-fold score curves; these cuts are never selected for use'
    ranked = top1.sort(["blend", "qid"], descending=[True, False])
    scores = ranked["blend"].to_numpy()
    labels = ranked["label"].to_numpy().astype(np.int64, copy=False)
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    sel = ends + 1
    true_positive = np.cumsum(labels)[ends]
    precision = true_positive / sel
    res = {"linked_target_denominator": linked}
    for target in PRECISION_TARGETS:
        valid = np.flatnonzero(precision >= target)
        if not len(valid):
            res[f"{target:.3f}"] = None
            continue
        best = valid[np.argmax(true_positive[valid])]
        res[f"{target:.3f}"] = {
            "descriptive_score_cut": float(scores[ends[best]]),
            "selected_pairs": int(sel[best]),
            "true_positive_pairs": int(true_positive[best]),
            "pair_precision": float(precision[best]),
            "pair_recall_all_linked_targets": float(true_positive[best] / linked),
        }
    return res


def source1_counts(selected: pl.DataFrame, references: int) -> dict:
    'summarize source1 candidate sizes with all unseen references represented as zero'
    counts = selected.group_by("qid").len().sort("len")["len"].to_numpy()
    zeros = references - len(counts)
    if zeros < 0:
        raise ValueError("selected references exceed the reference corpus")

    def nearest_rank(percentile: float) -> int:
        rank = max(0, int(np.ceil(percentile * references)) - 1)
        return 0 if rank < zeros else int(counts[rank - zeros])

    return {
        "reference_corpus": references,
        "references_with_candidates": int(len(counts)),
        "references_without_candidates": int(zeros),
        "mean": float(len(selected) / references),
        "median_nearest_rank": nearest_rank(.50),
        "p95_nearest_rank": nearest_rank(.95),
        "p99_nearest_rank": nearest_rank(.99),
        "maximum": int(counts[-1]) if len(counts) else 0,
    }


def target_flags(pairs: pl.DataFrame) -> pl.DataFrame:
    tids = pairs["tid"].unique()
    columns = ["rid", "own", "ad"]
    targets = pl.concat([
        pl.scan_parquet(DATA / "train/s2.parquet").select(columns),
        pl.scan_parquet(DATA / "train/s3.parquet").select(columns),
    ]).filter(pl.col("rid").is_in(tids.implode())).collect(engine="streaming").rename({"rid": "tid"})
    if targets["tid"].n_unique() != len(targets) or len(targets) != len(tids):
        raise ValueError("cached target data does not cover the paired-score targets exactly")
    return targets.with_columns(
        (pl.col("own") >= 0).alias("linked"),
        pl.col("ad").fill_null("").str.strip_chars().eq("").alias("blank_address"),
    ).select("tid", "linked", "blank_address")


def policy_rows(pairs: pl.DataFrame, flags: pl.DataFrame, name: str) -> pl.DataFrame:
    if name.startswith("k"):
        return top_k(pairs, int(name[1:]))
    ranked = pairs.sort(["tid", "gate", "qid"], descending=[False, True, False]).with_columns(
        pl.col("gate").shift(-1).over("tid").alias("next_gate"),
        pl.int_range(pl.len()).over("tid").alias("rank"),
    ).join(flags.select("tid", "blank_address"), on="tid", how="inner", validate="m:1").with_columns(
        (pl.col("gate") - pl.col("next_gate").fill_null(pl.col("gate"))).alias("top1_gap"),
    )
    if name == "blank10":
        return ranked.filter(pl.col("rank") < pl.when(pl.col("blank_address")).then(10).otherwise(3))
    if name == "gap1_else3":
        return ranked.filter(
            ((pl.col("rank") == 0) & (pl.col("top1_gap") > .5)) |
            ((pl.col("rank") < 3) & (pl.col("top1_gap") <= .5))
        )
    if name == "gap1_else3_blank10":
        return ranked.filter(
            pl.when(pl.col("blank_address"))
            .then(pl.col("rank") < 10)
            .otherwise(
                ((pl.col("rank") == 0) & (pl.col("top1_gap") > .5)) |
                ((pl.col("rank") < 3) & (pl.col("top1_gap") <= .5))
            )
        )
    raise ValueError(f"unknown policy {name}")


def evaluate(name: str, pairs: pl.DataFrame, flags: pl.DataFrame, linked: int, candidate_truth: int, references: int) -> dict:
    sel = policy_rows(pairs, flags, name)
    scored = sel.with_columns(pl.Series("blend", blend(sel["prob"].to_numpy(), sel["gate"].to_numpy())))
    retained = int(scored["label"].sum())
    top1 = top_one(scored)
    return {
        "policy": name,
        "candidate_pairs": len(scored),
        "target_mean_candidates": float(len(scored) / flags.height),
        "true_candidate_retention_all_linked_targets": retained / linked,
        "true_candidate_retention_given_fixed_universe": retained / candidate_truth,
        "source1_candidate_counts_entire_reference_corpus": source1_counts(scored, references),
        "same_fold0_high_precision_curve_descriptive_only": matched_precision(top1, linked),
    }


def input_consistency(pairs: pl.DataFrame) -> dict:
    stored = pl.read_parquet(BENCH / "neural_gate_blended_top3.parquet").filter(
        pl.col("variant") == SAFE_SCORE
    ).select("tid", "qid", "label", "prob", "gate", "blend")
    recomputed = top_k(pairs, 3)
    recomputed = recomputed.with_columns(pl.Series("blend", blend(recomputed["prob"].to_numpy(), recomputed["gate"].to_numpy())))
    joined = stored.join(recomputed, on=["tid", "qid"], how="full", coalesce=True, suffix="_new")
    missing_stored = joined["blend"].is_null().sum()
    missing_recomputed = joined["blend_new"].is_null().sum()
    delta = (joined.filter(pl.col("blend").is_not_null() & pl.col("blend_new").is_not_null())["blend"] -
             joined.filter(pl.col("blend").is_not_null() & pl.col("blend_new").is_not_null())["blend_new"]).abs()
    return {
        "paired_rows": len(pairs),
        "stored_safe_top3_rows": len(stored),
        "recomputed_safe_top3_rows": len(recomputed),
        "stored_only_pairs": int(missing_recomputed),
        "recomputed_only_pairs": int(missing_stored),
        "max_abs_blend_delta_w_0_6": float(delta.max()) if len(delta) else None,
        "exact_complete_alignment": len(stored) == len(recomputed) and not missing_stored and not missing_recomputed and float(delta.max()) < 1e-6,
    }


def report_md(result: dict) -> str:
    rows = [
        "# overnight width diagnostic",
        "",
        "selected lexical fold-0 cached pairs only. this is not production hybrid/full-pool validation, policy promotion, or evidence for a .998 claim.",
        "",
        "## gap cause",
        "",
        f"- {result['gap_cause']['candidate_miss_count']:,} of {result['gap_cause']['linked_targets']:,} linked targets ({result['gap_cause']['candidate_miss_rate']:.2%}) have no true pair in the fixed lexical universe; width cannot recover them.",
        f"- safe k3 retains {result['gap_cause']['safe_k3_true_candidate_retention']:.2%} of all linked targets; it loses {result['gap_cause']['safe_k3_gate_loss_count']:,} true pairs after lexical retrieval, including {result['gap_cause']['blank_address_safe_k3_gate_loss_count']:,} on {result['gap_cause']['blank_address_targets']:,} raw blank-address targets.",
        "",
        "## efficiency tradeoff",
        "",
        "| policy | target mean | true retention | recall @ >= .995 precision | source1 mean | p95 | p99 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in result["policies"].items():
        curve = value["same_fold0_high_precision_curve_descriptive_only"]["0.995"]
        recall = "n/a" if curve is None else f"{curve['pair_recall_all_linked_targets']:.2%}"
        source1 = value["source1_candidate_counts_entire_reference_corpus"]
        rows.append(
            f"| {name} | {value['target_mean_candidates']:.3f} | {value['true_candidate_retention_all_linked_targets']:.2%} | {recall} | "
            f"{source1['mean']:.4f} | {source1['p95_nearest_rank']} | {source1['p99_nearest_rank']} | {source1['maximum']:,} |"
        )
    rows.extend([
        "",
        "## reading it",
        "",
        "- compare fixed k3 with `blank10`, `gap1_else3`, and `gap1_else3_blank10`; their rows show the concrete cached candidate-cost versus retention tradeoff without fitting a threshold to labels.",
        "- source1 distribution includes every reference in `train/ref.parquet`; zero-candidate references are retained in its mean and tail ranks.",
        "- this is score-only cpu work: widening raw-blank targets can recover gate-pruned candidates while gap pruning removes easy rows, but actual neural cost also depends on batching, sequence lengths, and runtime overhead rather than pair count alone.",
        "",
        "## limits",
        "",
        "- every policy ranks the complete paired lexical universe directly; no policy is scored on a pair intersection.",
        "- w=.6 uses the unchanged cached neural probabilities and safe gate scores. precision curves are descriptive same-fold0 scans, not chosen operating thresholds.",
        "- no azure, new ml job, gpu run, or production/fullpool artifact was used.",
        "",
    ])
    return "\n".join(rows)


def check() -> None:
    toy = pl.DataFrame({"tid": [1, 1, 2], "qid": [2, 1, 1], "gate": [.9, .4, .8], "prob": [.7, .2, .6], "label": [1, 0, 0]})
    assert top_k(toy, 1).sort("tid")["qid"].to_list() == [2, 1]
    assert np.isclose(blend(np.array([.5], np.float32), np.array([.5], np.float32))[0], .5)
    counts = source1_counts(toy, 5)
    assert counts["mean"] == .6 and counts["p95_nearest_rank"] == 2 and counts["p99_nearest_rank"] == 2
    print("width checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pl.read_parquet(BENCH / "neural_gate_paired_scores.parquet").select("qid", "tid", "label", "prob", SAFE_SCORE).rename({SAFE_SCORE: "gate"})
    if pairs.select("qid", "tid").n_unique() != len(pairs) or pairs.null_count().sum_horizontal()[0]:
        raise ValueError("paired scores are incomplete or duplicated")
    flags = target_flags(pairs)
    linked = int(flags["linked"].sum())
    candidate_truth = int(pairs["label"].sum())
    if int((pairs.group_by("tid").agg(pl.col("label").sum().alias("truth")).join(flags, on="tid")["truth"] > 1).sum()):
        raise ValueError("paired benchmark has multiple true owners per target")
    refs = int(pl.scan_parquet(DATA / "train/ref.parquet").select(pl.len()).collect(engine="streaming").item())
    policies = {f"k{width}": None for width in FIXED_WIDTHS}
    policies |= {"blank10": None, "gap1_else3": None, "gap1_else3_blank10": None}
    rs = {name: evaluate(name, pairs, flags, linked, candidate_truth, refs) for name in policies}
    k3 = rs["k3"]
    blank = flags.filter("blank_address")
    blank_pairs = pairs.join(blank.select("tid"), on="tid", how="inner", validate="m:1")
    blank_k3 = policy_rows(blank_pairs, blank, "k3")
    res = {
        "scope": "selected lexical fold-0 cached diagnostic; not production hybrid/full-pool validation",
        "constraints": {"cpu_threads_max": 6, "ram_target_gib_lt": 3, "gpu": False, "azure": False},
        "inputs": {"paired_scores": str(BENCH / "neural_gate_paired_scores.parquet"), "stored_safe_top3": str(BENCH / "neural_gate_blended_top3.parquet"), "raw_target_flags": "cache/data/train/s2.parquet + s3.parquet; ad.fill_null('').strip_chars() == ''"},
        "input_consistency": input_consistency(pairs),
        "policy_definitions": {"fixed": list(FIXED_WIDTHS), "blank10": "k3 normally; k10 when raw target address is blank", "gap1_else3": "k1 when safe gate top1-top2 > predeclared 0.5; otherwise k3", "gap1_else3_blank10": "blank10 overrides gap1_else3 for raw blank-address targets"},
        "gap_cause": {"targets": flags.height, "linked_targets": linked, "candidate_truth_pairs": candidate_truth, "candidate_miss_count": linked - candidate_truth, "candidate_miss_rate": (linked - candidate_truth) / linked, "safe_k3_true_candidate_retention": k3["true_candidate_retention_all_linked_targets"], "safe_k3_gate_loss_count": candidate_truth - int(policy_rows(pairs, flags, "k3")["label"].sum()), "blank_address_targets": blank.height, "blank_address_safe_k3_gate_loss_count": int(blank_pairs["label"].sum()) - int(blank_k3["label"].sum())},
        "policies": rs,
        "selection": "no policy, blend weight, or score threshold was fit or selected; high-precision curves are descriptive same-fold0 evidence only",
    }
    text = json.dumps(res, indent=2) + "\n"
    (OUT / "metrics.json").write_text(text, encoding="utf-8")
    (ROOT / "reports/overnight-width.json").write_text(text, encoding="utf-8")
    (ROOT / "reports/overnight-width.md").write_text(report_md(res), encoding="utf-8")
    print(json.dumps({"gap_cause": res["gap_cause"], "policies": {name: {"pairs": value["candidate_pairs"], "retention": value["true_candidate_retention_all_linked_targets"]} for name, value in rs.items()}}, indent=2))


if __name__ == "__main__":
    main()
