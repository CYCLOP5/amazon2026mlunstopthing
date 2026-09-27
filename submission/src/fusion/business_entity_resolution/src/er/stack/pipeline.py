'`python src/run.py stack -c configs/stack.toml` - see er/stack/__init__.py'
import json
import os
import time

import numpy as np
import polars as pl

from er.io import write_id_lists
from er.safe import THREADS, guard
from er.stack import analysis, decode, rules
from er.stack.artifacts import cache_signature, save_models, load_models, source_digest
from er.stack.recall import add_exact_candidates
from er.stack.enhanced import ambiguity_features, occupancy_features, consensus_features
from er.stack.features import (as_matrix, chunks_by_tid, confident_owners, feature_names, group_features,
                               name_frequency, normalize, score_features, text_features)
from er.stack.inputs import add_lexical, load_pairs, load_refs, load_targets

T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)



def build_round1(sc, split, out_dir):
    path = os.path.join(out_dir, f"r1_{split}.parquet")
    tpath = os.path.join(out_dir, f"tgt_{split}.parquet")
    signature = cache_signature(sc, split)
    signature_path = os.path.join(out_dir, f"cache_{split}.json")
    if os.path.exists(path) and os.path.exists(tpath) and os.path.exists(signature_path):
        if json.load(open(signature_path)) == signature:
            log(f"{split}: verified round-1 table cache")
            return

    if os.path.exists(signature_path):
        os.remove(signature_path)
    refs, tg = load_refs(sc["data"], split), load_targets(sc["data"], split)
    for label, frame in (("refs", refs), ("targets", tg)):
        ids = frame["rid"]
        if not frame.height or ids.null_count() or ids.n_unique() != frame.height or ids.min() != 0 or ids.max() != frame.height-1:
            raise ValueError(f"{split}/{label}: rid must be unique contiguous row indices starting at zero")
    n_s2 = int((tg["sr"] == 2).sum())
    if not tg.filter(pl.col("sr") == 2)["rid"].is_between(0, n_s2-1).all():
        raise ValueError("S2 rid must precede S3 rid in prepared targets")
    pairs, info = load_pairs(sc[f"{split}_roots"], tg.height, threads=min(THREADS, 32),
                             require_complete=sc.get("require_complete", True))
    if pairs["qid"].max() is not None and pairs["qid"].max() >= refs.height:
        raise ValueError("Scored candidate qid is outside the prepared reference table")
    if split == "train" and ("y" not in pairs.columns or pairs["y"].null_count() or not pairs["y"].is_in([0,1]).all()):
        raise ValueError("Training scores require complete binary y labels")
    if sc.get(f"lexical_{split}"):
        n_neural = pairs.height
        pairs, li = add_lexical(pairs, sc[f"lexical_{split}"], refs, tg, sc["lexical_top_k"], sc["lexical_min_rel"])
        info.update(li)
        if split == "train":
            own = pl.concat([pl.read_parquet(os.path.join(sc["data"], "train", f"s{sr}.parquet"), columns=["rid", "own"])
                             for sr in (2, 3)]).select(pl.col("rid").cast(pl.UInt32).alias("tid"), "own")
            pairs = pairs.join(own, on="tid", how="left").with_columns(
                (pl.col("qid").cast(pl.Int64) == pl.col("own")).cast(pl.UInt8).alias("y")).drop("own")
            fold = refs.select(pl.col("rid").alias("qid"), "fold")
            pos = pairs.filter(pl.col("y") == 1).join(fold, on="qid").filter(pl.col("fold").is_in(sc["eval_folds"]))
            info["eval_true_pairs_neural"] = int((pos["in_neural"] == 1).sum())
            info["eval_true_pairs_added_by_lexical"] = int((pos["in_neural"] == 0).sum())
        log(f"{split}: {n_neural:,} neural pairs + lexical -> {pairs.height:,}")
    ref_attr, tgt_attr = None, None
    if sc.get("exact_rescue", False):
        ref_attr = normalize(refs.select("rid", "nm", "ad", "co"))
        tgt_attr = normalize(tg.select("rid", "nm", "ad", "co"))
        pairs, recall_info = add_exact_candidates(pairs, ref_attr, tgt_attr, sc.get("exact_max_owners", 8))
        info.update(recall_info)
        if split == "train":
            own = pl.concat([pl.read_parquet(os.path.join(sc["data"], "train", f"s{sr}.parquet"), columns=["rid", "own"])
                             for sr in (2, 3)]).select(pl.col("rid").cast(pl.UInt32).alias("tid"), "own")
            pairs = pairs.join(own, on="tid", how="left").with_columns(
                (pl.col("qid").cast(pl.Int64) == pl.col("own")).cast(pl.UInt8).alias("y")).drop("own")
            folds = refs.select(pl.col("rid").alias("qid"), "fold")
            positives = pairs.filter(pl.col("y") == 1).join(folds, on="qid").filter(pl.col("fold").is_in(sc["eval_folds"]))
            info["eval_true_pairs_union"] = positives.height
        pairs = pairs.sort("tid", "qid")
    log(f"{split}: {info}")
    json.dump(info, open(os.path.join(out_dir, f"inputs_{split}.json"), "w"), indent=1)
    sf = score_features(pairs, n_s2)
    del pairs
    if split == "train":
        sf = sf.join(refs.select(pl.col("rid").alias("qid"), "fold"), on="qid", how="left")
        rel = sf.filter(pl.col("fold").is_in(sc["eval_folds"]))["tid"].unique()
        sf = sf.filter(pl.col("tid").is_in(rel.implode()))
        log(f"train: kept {sf.height:,} pairs of {rel.len():,} records that can reach folds {sc['eval_folds']}")
    guard("stack/score features")
    if ref_attr is None:
        ref_attr = normalize(refs.select("rid", "nm", "ad", "co"))
    log(f"{split}: normalised {ref_attr.height:,} Source-1 records")
    if tgt_attr is None:
        tgt_attr = normalize(tg.filter(pl.col("rid").is_in(sf["tid"].unique().implode())).select("rid", "nm", "ad", "co"))
    else:
        tgt_attr = tgt_attr.filter(pl.col("rid").is_in(sf["tid"].unique().implode()))
    log(f"{split}: normalised {tgt_attr.height:,} Source-2/3 records")
    s1f, of = name_frequency(ref_attr, tgt_attr)

    sf = sf.sort("tid", "qid")
    parts, done = [], 0
    for ch in chunks_by_tid(sf.select("qid", "tid"), sc["chunk_rows"]):
        parts.append(text_features(ch, ref_attr, tgt_attr, sc["text_groups"]))
        done += ch.height
        log(f"{split}: text features {done:,}/{sf.height:,}")
        guard("stack/text features")
    txt = pl.concat(parts)
    if sc.get("enhanced_features", False):
        extra = ambiguity_features(sf.select("qid", "tid"), ref_attr, tgt_attr)
        txt = txt.join(extra, on=["qid", "tid"], how="left")
    del parts
    r1 = (sf.join(txt, on=["qid", "tid"], how="left").join(s1f, on="qid", how="left")
            .join(of, on="tid", how="left"))
    r1 = r1.sort("tid", "qid")
    r1.write_parquet(path + ".tmp")
    os.replace(path + ".tmp", path)
    tgt_attr.select("rid", "name_core", "addr_can",
                    pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("hn1")).write_parquet(tpath)
    json.dump(signature, open(signature_path, "w"))
    log(f"{split}: round-1 table {r1.height:,} x {r1.width} written")



def lgb_params(sc):
    return {"objective": "binary", "verbose": -1, "num_threads": THREADS, **sc["lgbm"]}


def cv_fit(df: pl.DataFrame, feats: list, fit_mask: np.ndarray, sc: dict, tag: str):
    'k-fold over the fit rows grouped by source-1 entity. returns p for every row (out-of-fold'
    import lightgbm as lgb
    k = sc["cv"]
    if k < 2:
        raise ValueError("stack.cv must be at least 2")
    X = as_matrix(df, feats)
    y = df["y"].to_numpy().astype(np.float32)
    grp = (df.select((pl.col("qid").hash(sc["seed"]) % k).alias("g"))["g"].to_numpy())
    p = np.zeros(len(y), dtype=np.float64)
    other = ~fit_mask
    models, iters = [], []
    for i in range(k):
        tr, va = fit_mask & (grp != i), fit_mask & (grp == i)
        if not tr.any() or not va.any():
            raise ValueError("Empty CV partition; increase the training sample or reduce stack.cv")
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(lgb_params(sc), dtr, num_boost_round=sc["fit"]["num_boost_round"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(sc["fit"]["early_stopping_rounds"], verbose=False)])
        p[va] = m.predict(X[va], num_threads=THREADS)
        if other.any():
            p[other] += m.predict(X[other], num_threads=THREADS) / k
        models.append(m)
        iters.append(m.best_iteration)
        log(f"{tag}: fold {i + 1}/{k} best_iteration {m.best_iteration}  valid logloss "
            f"{m.best_score['valid_0']['binary_logloss']:.5f}")
        guard(f"{tag} cv")
    gain = np.mean([m.feature_importance("gain") for m in models], axis=0)
    imp = sorted(zip(feats, gain.tolist()), key=lambda x: -x[1])
    return p.astype(np.float32), models, {"best_iterations": iters, "importance": imp[:40]}


def predict(models, df: pl.DataFrame, feats: list, rows: int = 2_000_000) -> np.ndarray:
    out = []
    for off in range(0, df.height, rows):
        X = as_matrix(df.slice(off, rows), feats)
        out.append(np.mean([m.predict(X, num_threads=THREADS) for m in models], axis=0))
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def add_round1(df, p):
    return df.with_columns(pl.Series("p1", p), pl.Series("r1_lg", np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))))


def add_groups(df, tgt_small, min_p, rows, enhanced=False):
    df = df.sort("tid", "qid")
    conf = confident_owners(df, min_p)
    parts = []
    for ch in chunks_by_tid(df.select("qid", "tid", "is_s3"), rows):
        part = group_features(ch, conf, tgt_small)
        if enhanced:
            part = part.join(consensus_features(ch, conf, tgt_small), on=["qid", "tid"], how="left")
        parts.append(part)
    res = df.join(pl.concat(parts), on=["qid", "tid"], how="left", maintain_order="left")
    return occupancy_features(res) if enhanced else res


def tuning_anchors(anchors, sc):
    out = anchors.filter(pl.col("fold") == sc["fit_fold"])
    buckets = sc.get("tune_holdout_buckets", 0)
    if buckets:
        if buckets < 2:
            raise ValueError("tune_holdout_buckets must be zero (legacy) or >= 2")
        out = out.filter((pl.col("qid").hash(sc["seed"] + 991) % buckets) == 0)
    return out.select("qid", "deg")



def evaluate(tr: pl.DataFrame, anchors: pl.DataFrame, col: str, sc: dict) -> dict:
    d = tr.select("qid", "tid", pl.col(col).alias("p"), "y")
    a_tune = tuning_anchors(anchors, sc)
    a_aud = anchors.filter(pl.col("fold") == sc["audit_fold"]).select("qid", "deg")
    t = decode.tune(d, a_tune)
    acc = decode.apply(t["rule"], d, t["threshold"])
    audit = decode.score(acc.select("qid", "y"), a_aud)
    return {"tune": t, "audit": audit}


def export(test: pl.DataFrame, col: str, rule: str, thr: float, refs, tg, out_dir: str, accepted=None):
    os.makedirs(out_dir, exist_ok=True)
    d = test.select("qid", "tid", pl.col(col).alias("p"), *(["in_neural"] if "in_neural" in test.columns else []))
    acc = accepted if accepted is not None else decode.apply(rule, d.select("qid", "tid", "p"), thr)

    e1 = refs.select(pl.col("rid").alias("qid"), pl.col("eid").alias("s1_id"))
    e2 = tg.select(pl.col("rid").alias("tid"), pl.col("eid").alias("o_id"))
    cand = d.join(e1, on="qid").join(e2, on="tid").select("s1_id", "o_id")
    match = acc.select("qid", "tid").join(e1, on="qid").join(e2, on="tid").select("s1_id", "o_id")
    bad = match.join(cand, on=["s1_id", "o_id"], how="anti").height
    if bad:
        raise SystemExit(f"{bad} matches outside the candidate set")
    s1_ids = refs.sort("rid")["eid"]
    write_id_lists(match, s1_ids, "matched_entity_ids", os.path.join(out_dir, "matching_results.tsv"))
    write_id_lists(cand, s1_ids, "candidate_entity_ids", os.path.join(out_dir, "candidate_pairs.tsv"))
    per = cand.group_by("s1_id").len()["len"]
    info = {"source1": s1_ids.len(), "matches": match.height, "candidate_pairs": cand.height,
            "candidates_per_s1_mean": float(per.sum() / s1_ids.len()), "candidates_per_s1_max": int(per.max() or 0),
            "rule": rule, "threshold": thr, "score": col}
    json.dump(info, open(os.path.join(out_dir, "export.json"), "w"), indent=1)
    log(f"exported {out_dir}: {info}")
    return info



def run(cfg, paths):
    sc = cfg["stack"]
    out_dir = os.path.join(paths.exp, "stack")
    work = sc.get("work") or out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    log(f"stack experiment '{paths.exp_name}', {THREADS} threads, output {out_dir}")
    if sc.get("inference_bundle"):
        return infer(cfg, paths, out_dir, work)
    if sc["fit_fold"] == sc["audit_fold"]:
        raise ValueError("fit_fold and audit_fold must differ")
    for split in ("train", "test"):
        build_round1(sc, split, work)
    json.dump(cfg, open(os.path.join(out_dir, "config_resolved.json"), "w"), indent=2)
    bundle = os.path.join(out_dir, "bundle")
    os.makedirs(bundle, exist_ok=True)

    if os.path.exists(os.path.join(bundle, "bundle.json")):
        os.remove(os.path.join(bundle, "bundle.json"))

    tr = pl.read_parquet(os.path.join(work, "r1_train.parquet"))
    te = pl.read_parquet(os.path.join(work, "r1_test.parquet"))
    anchors = load_refs(sc["data"], "train").select(pl.col("rid").alias("qid"), "deg", "fold")
    report = {"inputs": {s: json.load(open(os.path.join(work, f"inputs_{s}.json"))) for s in ("train", "test")}}
    if report["inputs"]["train"]["config_sha256"] != report["inputs"]["test"]["config_sha256"]:
        raise ValueError("Train/test upstream score configurations differ")
    fit = (tr["fold"] == sc["fit_fold"]).to_numpy()
    if sc.get("tune_holdout_buckets", 0):
        heldout = tuning_anchors(anchors, sc)["qid"]
        fit &= ~tr["qid"].is_in(heldout.implode()).to_numpy()
    report["validation"] = {"tune_entities": tuning_anchors(anchors, sc).height,
        "tune_holdout_buckets": sc.get("tune_holdout_buckets", 0),
        "audit_fold": sc["audit_fold"], "fit_rows": int(fit.sum())}
    log(f"train table {tr.height:,} pairs ({int(fit.sum()):,} fit rows, positives {int(tr.filter(pl.Series(fit))['y'].sum()):,}); "
        f"test table {te.height:,} pairs")


    f1 = [f for f in feature_names(tr.columns) if f not in sc.get("exclude_features", [])]
    p, models, info1 = cv_fit(tr, f1, fit, sc, "round 1")
    tr, te = add_round1(tr, p), add_round1(te, predict(models, te, f1))
    report["round1"] = {"features": f1, **info1}
    save_models(bundle, "round1", models, f1)
    del models


    tgt_tr = pl.read_parquet(os.path.join(work, "tgt_train.parquet"))
    tgt_te = pl.read_parquet(os.path.join(work, "tgt_test.parquet"))
    tr = add_groups(tr, tgt_tr, sc["confident_p"], sc["chunk_rows"], sc.get("enhanced_features", False))
    te = add_groups(te, tgt_te, sc["confident_p"], sc["chunk_rows"], sc.get("enhanced_features", False))
    log("group features added")
    f2 = f1 + [c for c in tr.columns if c.startswith("sib_")] + ["r1_lg"]
    p, models, info2 = cv_fit(tr, f2, fit, sc, "round 2")
    tr = tr.with_columns(pl.Series("p2", p))
    te = te.with_columns(pl.Series("p2", predict(models, te, f2)))
    report["round2"] = {"features": f2, **info2}
    save_models(bundle, "round2", models, f2)
    del models


    methods = {"base (neural blend as delivered)": "prob", "round 1 stacker": "p1", "round 2 stacker (groups)": "p2"}
    report["methods"] = {}
    for name, col in methods.items():
        r = evaluate(tr, anchors, col, sc)
        report["methods"][name] = {"score": col, **r}
        log(f"{name:34s} tune F0.5 {r['tune']['macro_f05']:.5f} ({r['tune']['rule']} @ {r['tune']['threshold']:.4f})"
            f" | AUDIT fold {sc['audit_fold']} F0.5 {r['audit']['macro_f05']:.5f}  P {r['audit']['pair_precision']:.4f}"
            f"  R {r['audit']['pair_recall']:.4f}")
    best = max(report["methods"], key=lambda k: report["methods"][k]["tune"]["macro_f05"])
    b = report["methods"][best]
    report["selected"] = best
    json.dump({"version": 1, "code_sha256": source_digest(), "stack": sc, "selected": b,
               "inputs": report["inputs"], "validation": report["validation"]},
              open(os.path.join(bundle, "bundle.json"), "w"), indent=2)


    if sc.get("rules", True):
        rr = rules.evaluate(tr, anchors, b["score"], b["tune"]["rule"], b["tune"]["threshold"], sc)
        report["rules"] = rr
        for k, v in rr["audit"].items():
            log(f"rules {k:28s} tune F0.5 {rr['tune'][k]['macro_f05']:.5f} | AUDIT F0.5 {v['macro_f05']:.5f}"
                f"  P {v['pair_precision']:.4f}  R {v['pair_recall']:.4f}")

    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1, default=str)
    keep = [c for c in ("qid", "tid", "y", "fold", "prob", "gate_prob", "neural_prob", "p1", "p2") if c in tr.columns]
    tr.filter(pl.col("fold").is_in(sc["eval_folds"])).select(keep).write_parquet(os.path.join(out_dir, "val_pred.parquet"))
    te.select([c for c in keep if c in te.columns]).write_parquet(os.path.join(out_dir, "test_pred.parquet"))
    col, rule, thr = b["score"], b["tune"]["rule"], b["tune"]["threshold"]
    acc_tr = decode.apply(rule, tr.select("qid", "tid", pl.col(col).alias("p"), "y"), thr)
    acc_te = decode.apply(rule, te.select("qid", "tid", pl.col(col).alias("p")), thr)
    if sc.get("analysis", True):
        rep = analysis.run(tr, te, acc_tr, acc_te, col, sc, out_dir, tgt_tr)
        report["analysis"] = {k: v for k, v in rep.items() if not k.startswith("drift")}
        log(f"analysis: {rep['by_cause']}")
        log(f"analysis: wrong {rep['wrong_by_cause']}; not-candidate links {rep['not_candidate_links']:,}, "
            f"same-name sibling would cover {rep['not_candidate_covered_by_same_name_sibling']:,}")
        log(f"analysis: acceptance {rep['acceptance']}")
    refs_te, tg_te = load_refs(sc["data"], "test"), load_targets(sc["data"], "test")
    report["export"] = export(te, col, rule, thr, refs_te, tg_te, paths.output, accepted=acc_te)
    if sc.get("rules", True):
        acc = rules.apply_all(te, b["score"], b["tune"]["rule"], b["tune"]["threshold"])
        report["export_rules"] = export(te, b["score"], b["tune"]["rule"], b["tune"]["threshold"], refs_te, tg_te,
                                        os.path.join(paths.exp, "output_rules"), accepted=acc)
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1, default=str)
    log(f"selected on fold {sc['fit_fold']}: {best}; audit fold {sc['audit_fold']} F0.5 {b['audit']['macro_f05']:.5f}. "
        f"Upload {paths.output}/matching_results.tsv (validate first).")


def infer(cfg, paths, out_dir, work):
    'replay a trained stacker on scored test candidates without training data or fitting'
    requested = cfg["stack"]
    bundle = requested["inference_bundle"]
    meta = json.load(open(os.path.join(bundle, "bundle.json")))
    if meta["version"] != 1:
        raise ValueError("Unsupported bundle version")
    if meta["code_sha256"] != source_digest():
        raise ValueError("Bundle was trained with different source code; use its original code revision")
    sc = dict(meta["stack"])

    for key in ("data", "test_roots", "lexical_test", "chunk_rows", "require_complete"):
        sc[key] = requested[key]
    build_round1(sc, "test", work)
    inputs = json.load(open(os.path.join(work, "inputs_test.json")))
    if inputs["config_sha256"] != meta["inputs"]["test"]["config_sha256"]:
        raise ValueError("Inference scores use a different upstream model configuration")
    te = pl.read_parquet(os.path.join(work, "r1_test.parquet"))
    models, f1 = load_models(bundle, "round1")
    te = add_round1(te, predict(models, te, f1))
    del models
    b = meta["selected"]
    if b["score"] == "p2":
        tgt = pl.read_parquet(os.path.join(work, "tgt_test.parquet"))
        te = add_groups(te, tgt, sc["confident_p"], sc["chunk_rows"], sc.get("enhanced_features", False))
        models, f2 = load_models(bundle, "round2")
        te = te.with_columns(pl.Series("p2", predict(models, te, f2)))
    refs, tg = load_refs(sc["data"], "test"), load_targets(sc["data"], "test")
    info = export(te, b["score"], b["tune"]["rule"], b["tune"]["threshold"], refs, tg, paths.output)
    json.dump(info, open(os.path.join(out_dir, "inference.json"), "w"), indent=2)
    return info
