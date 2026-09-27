#!/usr/bin/env python3
'streaming cpu-only ambiguity audit for cached entity-resolution data'

import argparse
import json
import os
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
peak_rss = 0


def configure(threads: int, memory_gib: float) -> None:
    if not 1 <= threads <= 6:
        raise ValueError("threads must be between 1 and 6")
    for key in ("POLARS_MAX_THREADS", "ARROW_NUM_THREADS"):
        os.environ[key] = str(threads)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:threads]))
    if not 0 < memory_gib < 4:
        raise ValueError("memory-gib must be greater than zero and less than 4")


def imports():
    global pl, psutil
    import polars as pl
    import psutil


def monitor(stop: threading.Event) -> None:
    global peak_rss
    process = psutil.Process()
    while not stop.is_set():
        peak_rss = max(peak_rss, process.memory_info().rss)
        time.sleep(.05)
    peak_rss = max(peak_rss, process.memory_info().rss)


def target(path: Path):
    return pl.scan_parquet(path).with_columns(
        pl.col("ad").fill_null("").str.strip_chars().eq("").alias("blank_address")
    )


def rows(frame, keys):
    return frame.group_by(keys).agg(pl.len().alias("records"))


def as_rows(frame):
    return sorted(frame.collect(engine="streaming").to_dicts(), key=lambda row: tuple(str(v) for v in row.values()))


def reference_names(data: Path, split: str):
    return rows(pl.scan_parquet(data / split / "ref.parquet").select("co", "nn"), ["co", "nn"])


def reference_summary(names):
    return as_rows(names.group_by("co").agg(
        pl.len().alias("normalized_name_groups"),
        pl.col("records").sum().alias("reference_records"),
        (pl.col("records") >= 2).sum().alias("colliding_name_groups"),
        pl.when(pl.col("records") >= 2).then(pl.col("records")).otherwise(0).sum().alias("reference_records_in_name_collisions"),
    ))


def source_summary(data: Path, split: str, source: int, names):
    frame = target(data / split / f"s{source}.parquet").select("co", "nn", "blank_address")
    totals = frame.group_by("co").agg(
        pl.len().alias("target_records"),
        pl.col("blank_address").sum().alias("blank_address_records"),
    )
    name_counts = rows(frame.select("co", "nn"), ["co", "nn"])
    name_stats = name_counts.group_by("co").agg(
        pl.len().alias("normalized_name_groups"),
        (pl.col("records") >= 2).sum().alias("colliding_name_groups"),
        pl.when(pl.col("records") >= 2).then(pl.col("records")).otherwise(0).sum().alias("target_records_in_name_collisions"),
    )
    blank = frame.filter(pl.col("blank_address")).join(names, on=["co", "nn"], how="left").with_columns(
        pl.col("records").fill_null(0).alias("reference_name_count")
    )
    blank_stats = blank.group_by("co").agg(
        pl.len().alias("blank_targets"),
        (pl.col("reference_name_count") >= 1).sum().alias("blank_targets_with_reference_name"),
        (pl.col("reference_name_count") >= 2).sum().alias("blank_targets_with_colliding_reference_name"),
    )
    stats = totals.join(name_stats, on="co", how="left").join(blank_stats, on="co", how="left").with_columns(
        pl.col("blank_targets").fill_null(0),
        pl.col("blank_targets_with_reference_name").fill_null(0),
        pl.col("blank_targets_with_colliding_reference_name").fill_null(0),
    ).with_columns(
        (pl.col("blank_address_records") / pl.col("target_records")).alias("blank_address_rate"),
        (pl.col("target_records_in_name_collisions") / pl.col("target_records")).alias("normalized_name_collision_rate"),
        pl.when(pl.col("blank_targets") > 0).then(
            pl.col("blank_targets_with_colliding_reference_name") / pl.col("blank_targets")
        ).otherwise(0.0).alias("blank_reference_name_collision_rate"),
    ).with_columns(pl.lit(source).alias("source"))
    return as_rows(stats)


def fold0_inputs(data: Path, names):
    refs = pl.scan_parquet(data / "train/ref.parquet").select(pl.col("rid").alias("own"), "fold")
    groups = []
    blank = []
    for src in (2, 3):
        labeled = target(data / "train" / f"s{src}.parquet").select("co", "nm", "ad", "nn", "own", "blank_address").filter(
            pl.col("own") >= 0
        ).join(refs, on="own", how="inner").filter(pl.col("fold") == 0)
        exact = labeled.group_by("co", "nm", "ad").agg(
            pl.len().alias("target_records"), pl.col("own").n_unique().alias("true_owners")
        ).filter(pl.col("true_owners") >= 2)
        groups.extend(as_rows(exact.group_by("co").agg(
            pl.len().alias("ambiguous_exact_input_groups"),
            pl.col("target_records").sum().alias("targets_in_ambiguous_exact_input_groups"),
            pl.col("true_owners").sum().alias("owner_assignments_across_groups"),
        ).with_columns(pl.lit(src).alias("source"))))
        blank.extend(as_rows(labeled.filter(pl.col("blank_address")).join(names, on=["co", "nn"], how="left").with_columns(
            pl.col("records").fill_null(0).alias("reference_name_count")
        ).group_by("co").agg(
            pl.len().alias("fold0_linked_blank_targets"),
            (pl.col("reference_name_count") >= 2).sum().alias("fold0_linked_blank_targets_with_colliding_reference_name"),
        ).with_columns(pl.lit(src).alias("source"))))
    return groups, blank


def loss_summary(fullpool: dict):
    res = []
    for blank, values in fullpool["losses"]["by_target_blank_address"].items():
        res.append({
            "blank_address": blank == "true",
            **values,
            "gate_loss_rate": values["gate_lost"] / values["linked_targets"],
            "wrong_top1_rate_after_gate": values["wrong_top1"] / values["postgate_true_targets"],
        })
    return res


def report(data: Path, memory_gib: float) -> dict:
    fullpool = json.loads((ROOT / "reports/overnight-fullpool.json").read_text(encoding="utf-8"))
    res = {
        "scope": "read-only cached-data eda; all prevalence uses every train/test source record; labeled associations use fold0 only",
        "constraints": {
            "cpu_only": True,
            "polars_streaming": True,
            "threads": int(os.environ["POLARS_MAX_THREADS"]),
            "memory_cap_gib": memory_gib,
            "no_fold1_inspection": True,
        },
        "fullpool_context": {
            "postgate_oracle_macro_f05": fullpool["postgate_oracle"]["macro_f05"],
            "postgate_oracle_below_target_0_998": fullpool["postgate_oracle"]["macro_f05"] < 0.998,
            "fold0_target_blank_losses": loss_summary(fullpool),
        },
        "splits": {},
    }
    for split in ("train", "test"):
        names = reference_names(data, split)
        res["splits"][split] = {
            "reference_normalized_name_collisions_by_country": reference_summary(names),
            "target_prevalence_by_source_country": [
                row for source in (2, 3) for row in source_summary(data, split, source, names)
            ],
        }
    exact, blank = fold0_inputs(data, reference_names(data, "train"))
    res["fold0_labeled_input_associations"] = {
        "exact_raw_target_input_multiple_owner_groups_by_source_country": exact,
        "linked_blank_target_reference_name_collisions_by_source_country": blank,
        "interpretation": "an exact group has equal country, source, raw name, and raw address across targets but multiple labeled owners. it proves target-only ambiguity; it does not alone prove identical full candidate-pair features or a global model ceiling.",
    }
    return res


def markdown(result: dict) -> str:
    loss = {row["blank_address"]: row for row in result["fullpool_context"]["fold0_target_blank_losses"]}
    all_gate_lost = loss[True]["gate_lost"] + loss[False]["gate_lost"]
    all_wrong_top1 = loss[True]["wrong_top1"] + loss[False]["wrong_top1"]
    train = result["splits"]["train"]["target_prevalence_by_source_country"]
    test = result["splits"]["test"]["target_prevalence_by_source_country"]
    references = {split: result["splits"][split]["reference_normalized_name_collisions_by_country"] for split in ("train", "test")}
    exact = result["fold0_labeled_input_associations"]["exact_raw_target_input_multiple_owner_groups_by_source_country"]
    blank = result["fold0_labeled_input_associations"]["linked_blank_target_reference_name_collisions_by_source_country"]
    lines = [
        "# ambiguity audit", "",
        "## root cause", "",
        f"- saved full-pool postgate oracle macro f0.5 is {result['fullpool_context']['postgate_oracle_macro_f05']:.6f}, below .998. gate reachability is therefore already insufficient for that target within the saved postgate universe.",
        f"- retained truth is {loss[True]['postgate_true_targets']:,}/{loss[True]['linked_targets']:,} ({loss[True]['postgate_recall']:.5f}) for blank addresses versus {loss[False]['postgate_true_targets']:,}/{loss[False]['linked_targets']:,} ({loss[False]['postgate_recall']:.5f}) for nonblank addresses.",
        f"- blank targets produce {loss[True]['gate_lost']:,}/{all_gate_lost:,} ({loss[True]['gate_lost'] / all_gate_lost:.2%}) gate losses and {loss[True]['wrong_top1']:,}/{all_wrong_top1:,} ({loss[True]['wrong_top1'] / all_wrong_top1:.2%}) wrong top-1 outcomes. their within-stratum rates are {loss[True]['gate_loss_rate']:.2%} before and {loss[True]['wrong_top1_rate_after_gate']:.2%} after the gate; nonblank rates are {loss[False]['gate_loss_rate']:.2%} and {loss[False]['wrong_top1_rate_after_gate']:.2%}.",
        "- this supports blank address as a major full-pool failure stratum. it does not establish that missing address alone explains the public-score gap.",
        "",
        "## exact target ambiguities", "",
        "| source | country | exact multi-owner groups | target records | owner assignments |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in exact:
        lines.append(f"| {row['source']} | {row['co']} | {row['ambiguous_exact_input_groups']:,} | {row['targets_in_ambiguous_exact_input_groups']:,} | {row['owner_assignments_across_groups']:,} |")
    if not exact:
        lines.append("| none | none | 0 | 0 | 0 |")
    lines.extend([
        "", "equal input features force a deterministic classifier to return an equal result for those features. these target-only groups do not by themselves show equal full candidate-pair features, so they are not used as a model-theoretic upper bound.",
        "", "## blank-name collisions", "",
        "| source | country | fold0 linked blank targets | colliding reference name | rate |",
        "| ---: | --- | ---: | ---: | ---: |",
    ])
    for row in blank:
        rate = row["fold0_linked_blank_targets_with_colliding_reference_name"] / row["fold0_linked_blank_targets"] if row["fold0_linked_blank_targets"] else 0
        lines.append(f"| {row['source']} | {row['co']} | {row['fold0_linked_blank_targets']:,} | {row['fold0_linked_blank_targets_with_colliding_reference_name']:,} | {rate:.2%} |")
    lines.extend(["", "## reference name collisions", "", "| split | country | reference records | records in colliding normalized names | rate |", "| --- | --- | ---: | ---: | ---: |"])
    for split, values in references.items():
        for row in values:
            rate = row["reference_records_in_name_collisions"] / row["reference_records"]
            lines.append(f"| {split} | {row['co']} | {row['reference_records']:,} | {row['reference_records_in_name_collisions']:,} | {rate:.2%} |")
    lines.extend(["", "## corpus shift", "", "| split | source | country | targets | blank address | normalized-name collision | blank target ref-name collision |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"])
    for split, values in (("train", train), ("test", test)):
        for row in values:
            lines.append(f"| {split} | {row['source']} | {row['co']} | {row['target_records']:,} | {row['blank_address_rate']:.2%} | {row['normalized_name_collision_rate']:.2%} | {row['blank_reference_name_collision_rate']:.2%} |")
    lines.extend([
        "", "france is absent from train and present in test, so no france training prevalence or labeled error estimate exists here. all corpus-shift counts use every supplied source record; fold1 accuracy was not inspected.",
    ])
    return "\n".join(lines) + "\n"


def check() -> None:
    assert 1 <= int(os.environ["POLARS_MAX_THREADS"]) <= 6
    assert all(os.environ[key] == "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
    names = rows(pl.DataFrame({"co": ["us", "us", "india"], "nn": ["a", "a", "a"]}).lazy(), ["co", "nn"])
    summary = reference_summary(names)
    assert summary == [
        {"co": "india", "normalized_name_groups": 1, "reference_records": 1, "colliding_name_groups": 0, "reference_records_in_name_collisions": 0},
        {"co": "us", "normalized_name_groups": 1, "reference_records": 2, "colliding_name_groups": 1, "reference_records_in_name_collisions": 2},
    ]
    print("ambiguity checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "cache/data")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--memory-gib", type=float, default=3.75)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    configure(args.threads, args.memory_gib)
    imports()
    if args.check:
        check()
        return
    stop = threading.Event()
    watcher = threading.Thread(target=monitor, args=(stop,), daemon=True)
    watcher.start()
    try:
        res = report(args.data, args.memory_gib)
    finally:
        stop.set()
        watcher.join()
    res["constraints"]["peak_rss_gib"] = peak_rss / 1024**3
    if peak_rss >= int(args.memory_gib * 1024**3):
        raise MemoryError(f"peak rss {peak_rss / 1024**3:.3f} gib reached the {args.memory_gib:.3f} gib cap")
    (ROOT / "reports/overnight-ambiguity.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (ROOT / "reports/overnight-ambiguity.md").write_text(markdown(res), encoding="utf-8")
    print(json.dumps({"report": "reports/overnight-ambiguity.json", "fold0_exact_groups": len(res["fold0_labeled_input_associations"]["exact_raw_target_input_multiple_owner_groups_by_source_country"])}, indent=2))


if __name__ == "__main__":
    main()
