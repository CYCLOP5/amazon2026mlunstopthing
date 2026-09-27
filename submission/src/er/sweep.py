"compare decision rules on an experiment's validation predictions (no retraining)"
import json
import os
import time

import numpy as np
import polars as pl

from er.decision import DECISIONS
from er.evaluation import macro_fbeta
from er.io import load_gt_pairs, read_tsv

GRIDS = {
    "argmax_threshold": [round(x, 3) for x in np.arange(0.30, 0.901, 0.025)],
    "threshold": [round(x, 3) for x in np.arange(0.50, 0.951, 0.05)],
    "expected_f": [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
}


def run(cfg, paths, apply: bool = False):
    t0 = time.time()
    va = pl.read_parquet(os.path.join(paths.exp, "val_pred.parquet"), columns=["s1_id", "o_id", "p"])
    tr = cfg["training"]
    s1 = (read_tsv(paths.raw("train", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
            .filter(((pl.col("s1_id").hash(seed=tr["hash_seed"]) % 100) >= tr["train_pct"])
                    & ((pl.col("s1_id").hash(seed=tr["hash_seed"]) % 100) < tr["val_pct"])))
    truth = load_gt_pairs(paths.raw("train", "gt")).filter(pl.col("s1_id").is_in(s1["s1_id"].implode()))
    print(f"validation entities {s1.height:,}, candidate pairs {va.height:,}", flush=True)
    rows, best = [], None
    for rule_name, grid in GRIDS.items():
        rule = DECISIONS.get(rule_name)
        for thr in grid:
            m = rule(va, thr).select("s1_id", "o_id")
            f = macro_fbeta(m, truth, s1["s1_id"])
            rows.append((rule_name, thr, f, m.height))
            if best is None or f > best[2]:
                best = (rule_name, thr, f)
        top = max((r for r in rows if r[0] == rule_name), key=lambda r: r[2])
        print(f" {rule_name:17s} best F0.5 = {top[2]:.5f} at {top[1]}  ({time.time()-t0:.0f}s)", flush=True)


    lines = ["# Decision-rule sweep (validation)", "", "| rule | parameter | macro F0.5 | predicted pairs |",
             "|---|---|---|---|"]
    per_rule_best = {}
    for r in rows:
        if r[0] not in per_rule_best or r[2] > per_rule_best[r[0]][2]:
            per_rule_best[r[0]] = r
    for r in per_rule_best.values():
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.5f} | {r[3]:,} |")
    lines += ["", "Per country (best setting of each rule):", "", "| rule | country | macro F0.5 | entities |", "|---|---|---|---|"]
    for rule_name, thr, _, _ in per_rule_best.values():
        m = DECISIONS.get(rule_name)(va, thr).select("s1_id", "o_id")
        for country in sorted(s1["country"].unique().to_list()):
            ids = s1.filter(pl.col("country") == country)["s1_id"]
            f = macro_fbeta(m.filter(pl.col("s1_id").is_in(ids.implode())),
                            truth.filter(pl.col("s1_id").is_in(ids.implode())), ids)
            lines.append(f"| {rule_name} | {country} | {f:.5f} | {len(ids):,} |")
    lines += ["", "All settings:", "", "| rule | parameter | macro F0.5 |", "|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]:.5f} |" for r in rows]
    out_dir = os.path.join(paths.exp, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "decision_sweep.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"BEST: rule={best[0]} parameter={best[1]} macro F0.5={best[2]:.5f}")
    print(f"wrote {os.path.join(out_dir, 'decision_sweep.md')}")

    if apply:
        mp = os.path.join(paths.exp, "metrics.json")
        metrics = json.load(open(mp))
        metrics.update(decision_rule=best[0], threshold=best[1], val_f05=best[2])
        json.dump(metrics, open(mp, "w"), indent=1, default=str)
        print(f"applied to {mp}. Now re-write the outputs with:\n"
              f"  python src/run.py -c <same config> --only predict,validate")
