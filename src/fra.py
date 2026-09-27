"""france-only policy audit on complete cached candidates"""
import argparse as ap
from pathlib import Path as path

import polars as pl

import decode
import frule
import gfeat
import infer
import post


def links(file, refs, targets):
    rows = (pl.scan_csv(file, separator="\t", schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String})
            .filter(pl.col("source1_entity_id").is_in(refs["eid"].implode()))
            .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").collect(engine="streaming"))
    result = (rows.join(refs.select(pl.col("eid").alias("source1_entity_id"), pl.col("rid").alias("qid")), on="source1_entity_id")
              .join(targets.select(pl.col("eid").alias("matched_entity_ids"), pl.col("rid").alias("tid")), on="matched_entity_ids")
              .select("qid", "tid"))
    if len(result) != len(rows) or result["tid"].n_unique() != len(result):
        raise ValueError("france matches lost ids or reused targets")
    return result


def choose(pairs, recipe, attrs, threads):
    post.validate_recipe(recipe)
    raw = post.score_rows({k: pairs[k].to_numpy() for k in ("co", "gate_prob", "neural_prob", "stack_prob")}, recipe)
    p = post.adjust(raw, pairs["co"].to_numpy(), pairs["seg"].to_numpy(), recipe)
    top = decode.winners(pairs["qid"].to_numpy(), pairs["tid"].to_numpy(), p, raw)
    keep, _ = decode.choose(pairs["qid"].to_numpy()[top], pairs["tid"].to_numpy()[top], p[top], raw[top],
                            floor=recipe["floor"], exact=recipe["exact_limit"], threads=threads)
    result = pairs[top[keep]].select("qid", "tid")
    if recipe.get("rules"):
        result, _ = frule.decide(result, attrs, recipe["rules"])
    return result


def run(scores, data, normalizer, cache, out, threads=2):
    if threads < 1 or out.exists():
        raise ValueError("invalid threads or existing france audit")
    source = post.verified(scores, "test")
    original = infer._json("artifacts/submission-learned/calibration.json")
    rules = original["rules"]
    frule.validate(rules)
    if infer._sha(normalizer) != rules["normalizer_sha256"] or infer._sha(data / "meta.json") != source["data_meta_sha256"]:
        raise ValueError("france audit normalization or data changed")
    scan = pl.scan_parquet(scores)
    columns = ["qid", "tid", "co", "seg", "gate_prob", "neural_prob", "stack_prob", *[f"np_m{i}" for i in range(15)]]
    pairs = scan.filter(pl.col("co") == "france").select(columns).collect(engine="streaming")
    cross = (scan.filter(pl.col("co") != "france").select("tid")
             .join(pairs.select("tid").unique().lazy(), on="tid", how="semi").select(pl.len()).collect(engine="streaming").item())
    if cross:
        raise ValueError("france-only projection would remove competing countries")
    refs = pl.scan_parquet(data / "test/ref.parquet").filter(pl.col("co") == "france").collect()
    targets = (pl.concat([pl.scan_parquet(data / "test" / f"s{i}.parquet") for i in (2, 3)])
               .select("rid", "eid", "nm", "ad", "co", "sr")
               .join(pairs.select(pl.col("tid").alias("rid")).unique().lazy(), on="rid", how="semi").collect(engine="streaming"))
    state = gfeat.prep(refs, data, "test", normalizer, cache)
    raw_keep, _ = decode.choose(pairs["qid"].to_numpy(), pairs["tid"].to_numpy(), pairs["gate_prob"].to_numpy(), floor=.05, threads=threads)
    values = []
    for part in pairs.iter_slices(50000):
        rows = gfeat.rows(state, targets, part)
        for row in rows.select("rn", "tn", "raw_r", "raw_t", "ra", "ta", "raw_ra", "raw_ta").iter_rows():
            flags = frule.flags(row, state["frequency"].get("france", {}), state["words"].get(("france", "a"), {}), rules["same_street"])
            values.append((*flags, not bool(row[7].strip()), row[0] == row[1], row[4] == row[5] and bool(row[4])))
    attrs = pl.concat([pairs.select("qid", "tid").with_columns(pl.Series("raw_keep", raw_keep)),
                       pl.DataFrame(values, schema=["swap", "conflict", "recall", "initials", "blank", "name_exact", "address_exact"], orient="row")], how="horizontal")
    del state, values
    out.mkdir(parents=True)
    pairs.write_parquet(out / "candidates.parquet")
    attrs.write_parquet(out / "attributes.parquet")
    refs.write_parquet(out / "references.parquet")
    targets.write_parquet(out / "targets.parquet")
    baseline = links("artifacts/submission-learned/output/matching_results.tsv", refs, targets)
    expected = links("artifacts/retune-r2/france-fullstack-empirical/output/matching_results.tsv", refs, targets)
    recipes = {"gate_density": {**original, "rules": None},
               "gate_empirical": infer._json("artifacts/retune-r2/cal-gate-empirical.json"),
               "stack_density": infer._json("artifacts/retune-r2/cal-fullstack.json"),
               "stack_empirical": infer._json("artifacts/retune-r2/cal-fullstack-empirical.json")}
    results = []
    keys = ["qid", "tid"]
    fields = ["swap", "conflict", "recall", "initials", "blank", "name_exact", "address_exact"]
    for name, recipe in recipes.items():
        if (recipe["config_sha256"] != source["config_sha256"] or recipe["sources"]["test"] != source["score_sha256"] or
                recipe["data_meta_sha256"] != source["data_meta_sha256"]):
            raise ValueError("france policy does not match the frozen scores")
        for use_rules in (False, True):
            label = name + ("_rules" if use_rules else "")
            recipe = {**recipe, "rules": rules if use_rules else None}
            predicted = choose(pairs, recipe, attrs, threads)
            if label in ("gate_density_rules", "stack_empirical"):
                target = baseline if label == "gate_density_rules" else expected
                if len(predicted) != len(target) or len(predicted.join(target, on=keys, how="anti")):
                    raise ValueError(f"{label} differs from the validated full export")
            added = predicted.join(baseline, on=keys, how="anti")
            removed = baseline.join(predicted, on=keys, how="anti")
            row = {"policy": label, "matches": len(predicted), "matched_references": predicted["qid"].n_unique(),
                   "added": len(added), "removed": len(removed), "flag_counts": {}}
            for kind, frame in (("accepted", predicted), ("added", added), ("removed", removed)):
                annotated = frame.join(attrs, on=keys)
                row["flag_counts"][kind] = {field: int(annotated[field].sum()) for field in fields}
                if kind != "accepted" and len(frame):
                    examples = (annotated.sample(n=min(30, len(frame)), seed=42).join(pairs, on=keys)
                                .join(refs.select(pl.col("rid").alias("qid"), pl.col("nm").alias("reference_name"), pl.col("ad").alias("reference_address")), on="qid")
                                .join(targets.select(pl.col("rid").alias("tid"), pl.col("nm").alias("target_name"), pl.col("ad").alias("target_address")), on="tid"))
                    examples.write_csv(out / f"{label}-{kind}.tsv", separator="\t")
            predicted.write_parquet(out / f"{label}.parquet")
            results.append(row)
            print(row, flush=True)
    report = {"scope": "complete france candidate competition; no cross-country rival removed", "source_scores_sha256": source["score_sha256"],
              "data_meta_sha256": source["data_meta_sha256"], "normalizer_sha256": infer._sha(normalizer),
              "frule_code_sha256": infer._sha(path(frule.__file__)), "gfeat_code_sha256": infer._sha(path(gfeat.__file__)),
              "references": len(refs), "candidates": len(pairs), "targets": len(targets), "cross_country_rivals": cross,
              "validated_export_parity": True, "accuracy": "unmeasured; flags and neural agreement are not france labels", "results": results}
    infer._write(out / "audit.json", report)
    infer._write("reports/france-route-audit.json", report)
    return report


def check():
    pairs = pl.DataFrame({"qid": [0, 1], "tid": [0, 0], "co": ["france", "france"], "seg": [0, 0],
                          "gate_prob": [.9, .8], "neural_prob": [.8, .95], "stack_prob": [.8, .95]})
    recipe = {"version": 1, "kind": "segmented-postprocessor", "unseen_gate": False, "floor": .05,
              "neural_weight": .6, "exact_limit": 64, "known_score": "stack_prob", "countries": ["us", "india"],
              "partition": {"modulus": 3, "remainder": 1}, "rules": None,
              "curves": {"france|0": {"centres": [-20., 20.], "posterior": [0., 1.]}}}
    assert choose(pairs, recipe, None, 1).rows() == [(1, 0)]
    assert choose(pairs, {**recipe, "unseen_gate": True}, None, 1).rows() == [(0, 0)]
    print("france routing and competing-owner checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=path, default=path("artifacts/full-result/scores/stack-test.parquet"))
    p.add_argument("--data", type=path, default=path("cache/data"))
    p.add_argument("--normalizer", type=path, default=path("artifacts/full-result/stack/normalizer.json"))
    p.add_argument("--cache", type=path, default=path("cache"))
    p.add_argument("--out", type=path, default=path("artifacts/france-r2"))
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    run(a.scores, a.data, a.normalizer, a.cache, a.out, a.threads)
