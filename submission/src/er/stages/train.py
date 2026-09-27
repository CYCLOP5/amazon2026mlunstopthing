'stage train: fit the configured model and tune the decision threshold'
import csv
import datetime
import json
import os
import time

import numpy as np
import polars as pl

from er.decision import DECISIONS
from er.evaluation import macro_fbeta
from er.features import feature_columns
from er.io import load_gt_pairs, s1_bucket
from er.models import MODELS
from er.safe import THREADS, guard


def is_done(cfg, paths, split=None):
    return os.path.exists(os.path.join(paths.exp, "metrics.json"))


def threshold_grid(dec: dict):
    if dec.get("threshold", "auto") != "auto":
        return [float(dec["threshold"])]
    lo, hi, step = dec.get("grid", [0.1, 0.95, 0.025])
    return [float(x) for x in np.round(np.arange(lo, hi + 1e-9, step), 4)]


def run(cfg, paths, split=None):
    t0 = time.time()
    tr, mc = cfg["training"], cfg["model"]
    assert tr["val_pct"] <= tr["sample_pct"], "[training] val_pct must be <= sample_pct"
    lf = pl.scan_parquet(os.path.join(paths.features("train"), "*.parquet")).with_columns(
        s1_bucket(tr["hash_seed"]).alias("_bkt"))
    feats = [f for f in feature_columns(lf.drop("_bkt").collect_schema().names())
             if f not in set(mc.get("exclude_features", []))]
    cast = [pl.col(f).cast(pl.Float32) for f in feats]

    fit = lf.filter(pl.col("_bkt") < tr["train_pct"]).select(*cast, "label").collect()
    y_fit, X_fit = fit["label"].to_numpy(), fit.select(feats).to_numpy()
    del fit
    guard("train/load fit set")
    va = (lf.filter((pl.col("_bkt") >= tr["train_pct"]) & (pl.col("_bkt") < tr["val_pct"]))
            .select("s1_id", "o_id", *cast, "label").collect())
    X_va, y_va = va.select(feats).to_numpy(), va["label"].to_numpy()
    va = va.select("s1_id", "o_id", "label")
    guard("train/load val set")
    print(f"fit pairs {len(y_fit):,} (pos {int(y_fit.sum()):,}) | val pairs {len(y_va):,} | "
          f"{len(feats)} features | model {mc['name']}  load {time.time()-t0:.0f}s", flush=True)

    model = MODELS.get(mc["name"])(mc.get("params", {}), mc.get("fit", {}), threads=THREADS)
    model.fit(X_fit, y_fit, X_va, y_va, feats)
    n_fit, n_val = len(y_fit), len(y_va)
    del X_fit, y_fit
    va = va.with_columns(pl.Series("p", model.predict(X_va), dtype=pl.Float32))
    del X_va
    print(f"trained  t={time.time()-t0:.0f}s", flush=True)


    s1_val = (pl.scan_parquet(paths.prep_file("train", 1)).select(pl.col("entity_id").alias("s1_id"))
                .filter((s1_bucket(tr["hash_seed"]) >= tr["train_pct"]) & (s1_bucket(tr["hash_seed"]) < tr["val_pct"]))
                .collect()["s1_id"])
    truth = load_gt_pairs(paths.raw("train", "gt")).filter(pl.col("s1_id").is_in(s1_val.implode()))
    oracle = macro_fbeta(va.filter(pl.col("label") == 1).select("s1_id", "o_id"), truth, s1_val)
    rule = DECISIONS.get(cfg["decision"]["rule"])
    curve = []
    for thr in threshold_grid(cfg["decision"]):
        f = macro_fbeta(rule(va, thr).select("s1_id", "o_id"), truth, s1_val)
        curve.append((thr, f))
        print(f"  thr={thr:.3f}  F0.5={f:.4f}")
    best_thr, best_f = max(curve, key=lambda x: x[1])
    print(f"validation: {len(s1_val):,} Source-1 ids | oracle F0.5 (perfect matcher on candidates) = {oracle:.4f}")
    print(f"BEST validation macro F0.5 = {best_f:.4f} at threshold {best_thr:.3f}")

    os.makedirs(os.path.join(paths.exp, "model"), exist_ok=True)
    model.save(os.path.join(paths.exp, "model"))
    va.write_parquet(os.path.join(paths.exp, "val_pred.parquet"))
    imp = model.importance()
    metrics = {"experiment": paths.exp_name, "model": mc["name"], "val_f05": best_f, "threshold": best_thr,
               "oracle_f05": oracle, "n_features": len(feats), "features": feats,
               "fit_pairs": n_fit, "val_pairs": n_val, "val_s1": len(s1_val), "threshold_curve": curve,
               "importance": imp, "model_info": model.info, "decision_rule": cfg["decision"]["rule"],
               "hashes": {"prep": paths.h_prep, "blocking": paths.h_block, "features": paths.h_feat},
               "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "train_minutes": round((time.time() - t0) / 60, 1)}
    json.dump(metrics, open(os.path.join(paths.exp, "metrics.json"), "w"), indent=1, default=str)
    if imp:
        print("top features:", [(n, round(v)) for n, v in imp[:15]])

    board = os.path.join(paths.experiments, "leaderboard.csv")
    new = not os.path.exists(board)
    with open(board, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["trained_at", "experiment", "model", "val_f05", "threshold", "oracle_f05", "n_features",
                        "features_hash", "blocking_hash"])
        w.writerow([metrics["trained_at"], paths.exp_name, mc["name"], f"{best_f:.5f}", best_thr, f"{oracle:.5f}",
                    len(feats), paths.h_feat, paths.h_block])
