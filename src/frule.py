"""explicit unlabeled-country decision experiments on authentic candidates"""

import hashlib as hh
import json
from pathlib import Path as path

import numpy as np
import polars as pl

import decode
import gfeat
import infer
import rfeat


def config(normalizer, country="france"):
    return {"version": 1, "kind": "unlabelled-name-patterns", "country": country,
            "normalizer_sha256": infer._sha(normalizer), "code_sha256": infer._sha(path(__file__)),
            "drop_swaps": True, "drop_conflicts": True, "add_initials": True, "same_street": .8}


def validate(rules):
    if (not isinstance(rules, dict) or rules.get("version") != 1 or rules.get("kind") != "unlabelled-name-patterns" or
            not isinstance(rules.get("country"), str) or not rules["country"] or
            rules.get("code_sha256") != infer._sha(path(__file__)) or
            any(type(rules.get(k)) is not bool for k in ("drop_swaps", "drop_conflicts", "add_initials")) or
            not isinstance(rules.get("same_street"), (int, float)) or not 0 < rules["same_street"] <= 1):
        raise ValueError("invalid or stale name-pattern rules")


def subsequence(a, b):
    it = iter(b)
    return all(c in it for c in a)


def flags(row, frequencies, weights, threshold):
    rn, tn, raw_r, raw_t, ra, ta, raw_ra, raw_ta = row
    values = gfeat.describe(rn, tn, raw_r, raw_t, raw_ra, raw_ta, frequencies)
    missing, extra = set(rn.split()) - set(tn.split()), set(tn.split()) - set(rn.split())
    swap = bool(values[7])
    if swap and len(missing) == len(extra) == 1:
        a, b = next(iter(missing)), next(iter(extra))
        swap = not (subsequence(a, b) or subsequence(b, a))
    street = bool(ra and ta) and rfeat.overlap(ra, ta, weights)[0] >= threshold
    safe_address = street and not bool(values[23])
    consistent = bool(values[19])
    if values[17] and safe_address:
        host = raw_t.lower()
        consistent |= any(len(w) >= 4 and frequencies.get(w, 0) < 20 and w in host for w in rn.split())
    alias = bool(values[18]) and bool(rn.split()) and set(rn.split()) <= set(tn.split())
    recall = safe_address and (consistent or alias or bool(values[3]) or bool(values[1]))
    initials = safe_address and bool(values[16] or values[17]) and consistent
    return swap, bool(values[23] and not values[24]), recall, initials


def decide(matches, candidates, rules):
    drop = pl.lit(False)
    if rules["drop_swaps"]:
        drop |= pl.col("swap")
    if rules["drop_conflicts"]:
        drop |= pl.col("conflict")
    dropped = matches.join(candidates.filter(drop).select("qid", "tid"), on=["qid", "tid"], how="semi")
    kept = matches.join(dropped, on=["qid", "tid"], how="anti")
    add = candidates.filter(~drop & ((pl.col("raw_keep") & pl.col("recall")) | (pl.col("initials") & rules["add_initials"])))
    add = add.join(kept.select("tid"), on="tid", how="anti")
    counts = add.group_by("tid").agg(pl.col("qid").n_unique().alias("owners"))
    add = add.join(counts.filter(pl.col("owners") == 1).select("tid"), on="tid", how="semi").select("qid", "tid").unique()
    result = pl.concat([kept, add])
    if result["tid"].n_unique() != len(result):
        raise ValueError("name-pattern rules assigned more than one owner")
    return result, {"dropped": len(dropped), "added": len(add)}


def apply(prepared, data, matches, rules, normalizer, cache, threads=8):
    validate(rules)
    if normalizer is None or infer._sha(normalizer) != rules["normalizer_sha256"]:
        raise ValueError("name-pattern normalization changed")
    country = rules["country"]
    refs = pl.read_parquet(path(data) / "test/ref.parquet").filter(pl.col("co") == country)
    if not len(refs):
        return matches, {"added": 0, "dropped": 0}
    pairs = pl.scan_parquet(prepared).filter(pl.col("co") == country).select("qid", "tid", "gate_prob").collect(engine="streaming")
    if not len(pairs):
        return matches, {"added": 0, "dropped": 0, "candidate_rows": 0, "country": country}
    state = gfeat.prep(refs, data, "test", normalizer, cache)
    targets = pl.concat([pl.read_parquet(path(data) / "test" / f"s{i}.parquet", columns=["rid", "nm", "ad", "co"]) for i in (2, 3)]).sort("rid")
    raw_keep, _ = decode.choose(pairs["qid"].to_numpy(), pairs["tid"].to_numpy(), pairs["gate_prob"].to_numpy(),
                                floor=.05, threads=threads)
    values = []
    for part in pairs.iter_slices(50000):
        q = targets.select(pl.all().gather(part["tid"].unique().cast(pl.UInt32)))
        rows = gfeat.rows(state, q, part)
        values.extend(flags(row, state["frequency"].get(country, {}), state["words"].get((country, "a"), {}), rules["same_street"])
                      for row in rows.select("rn", "tn", "raw_r", "raw_t", "ra", "ta", "raw_ra", "raw_ta").iter_rows())
    attributes = pl.DataFrame(values, schema=["swap", "conflict", "recall", "initials"], orient="row")
    candidates = pl.concat([pairs.select("qid", "tid").with_columns(pl.Series("raw_keep", raw_keep)), attributes], how="horizontal")
    result, stats = decide(matches, candidates, rules)
    stats.update({"country": country, "candidate_rows": len(pairs), "rules_sha256": hh.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()})
    return result, stats


def check():
    rules = {"drop_swaps": True, "drop_conflicts": True, "add_initials": True}
    matches = pl.DataFrame({"qid": [0, 1], "tid": [0, 1]})
    candidates = pl.DataFrame({"qid": [0, 1, 2, 3, 4], "tid": [0, 1, 2, 3, 3],
                               "swap": [True, False, False, False, False], "conflict": [False] * 5,
                               "recall": [False, False, True, True, True], "initials": [False, False, True, True, True],
                               "raw_keep": [False] * 5})
    result, stats = decide(matches, candidates, rules)
    assert set(result.iter_rows()) == {(1, 1), (2, 2)} and stats == {"dropped": 1, "added": 1}
    contradictory = candidates.with_columns(pl.when(pl.col("tid") == 0).then(True).otherwise(pl.col("recall")).alias("recall"),
                                            pl.when(pl.col("tid") == 0).then(True).otherwise(pl.col("raw_keep")).alias("raw_keep"))
    assert (0, 0) not in set(decide(matches, contradictory, rules)[0].iter_rows())
    assert not flags(("alpha beta gamma", "abzz", "Alpha Beta Gamma", "ABZZ", "10 main", "10 main", "10 main", "10 main"), {}, {}, .8)[3]
    assert flags(("alpha beta", "ab", "Alpha Beta", "AB", "10 main", "10 main", "10 main", "10 main"), {}, {}, .8)[3]
    print("name-pattern decision checks passed")


if __name__ == "__main__":
    check()
