#!/usr/bin/env python3
"""cpu-only full-pool postgate diagnosis, blend selection, and held-out stacking."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path as path


def _threads(n):
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    if n < 1 or n > available:
        raise ValueError("threads must be between 1 and available cpus")
    for key in ("POLARS_MAX_THREADS", "ARROW_NUM_THREADS"):
        os.environ[key] = str(n)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def _imports():
    global np, pl, pq, infer
    import numpy as np
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(int(os.environ["ARROW_NUM_THREADS"]))
    pa.set_io_thread_count(1)
    try:
        import infer
    except ImportError:
        from src import infer


weights = (0., .2, .4, .6, .8, 1.)
root = path(__file__).resolve().parents[1]


def _hash(z):
    return hashlib.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()


def _held(ids):
    x = np.asarray(ids, dtype=np.uint64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return ((x ^ (x >> np.uint64(31))) % np.uint64(2)) == 0


def _sample(ids):
    x = np.asarray(ids, dtype=np.uint64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return ((x ^ (x >> np.uint64(31))) % np.uint64(50)) == 0


def _fit_edges(q, t, own, fold):
    groups = np.where(own[t] >= 0, own[t], t.astype(np.int64) + len(fold))
    return ((fold[q] == 0) & ~_held(q) & ~_held(groups) &
            ((own[t] < 0) | (fold[np.maximum(own[t], 0)] == 0)))


def _logit(x):
    eps = np.finfo(np.float32).eps
    x = np.clip(x, eps, 1 - eps)
    return np.log(x) - np.log1p(-x)


def _blend(gate, neural, weight):
    if weight == 1:
        return neural
    z = np.float32(1 - weight) * _logit(gate) + np.float32(weight) * _logit(neural)
    return (np.float32(1) / (np.float32(1) + np.exp(-z))).astype(np.float32)


def _stack(gate, neural, model):
    x = model["intercept"] + model["coefficients"][0] * _logit(gate) + model["coefficients"][1] * _logit(neural)
    return (1 / (1 + np.exp(-np.clip(x, -80, 80)))).astype(np.float32)


def _input(data, bases):
    paths, indices, owners = [], [], []
    for base in bases:
        base = base.resolve()
        index = infer._json(base / "runs.json")
        owner = index.get("owner")
        if (index.get("version") != 2 or not isinstance(owner, dict) or
                index.get("owner_sha256") != _hash(owner) or owner.get("split") != "train" or
                owner.get("data_meta_sha256") != infer._sha(data / "meta.json") or
                index.get("complete") is not True or
                index.get("verification", {}).get("status") != "exact_coverage" or
                index.get("verification", {}).get("full") != index.get("full") or
                not isinstance(index.get("runs"), list) or not index["runs"]):
            raise ValueError("incomplete or invalid training run index")
        owners.append(owner)
        indices.append({"path": str(base / "runs.json"), "sha256": infer._sha(base / "runs.json")})
        for entry in index["runs"]:
            if not isinstance(entry, dict) or entry.get("childstatus") != "succeeded" or entry.get("config") != owner:
                raise ValueError("invalid run index entry")
            rel = entry.get("runpath")
            if not isinstance(rel, str) or path(rel).is_absolute() or ".." in path(rel).parts or path(rel).parts[:1] != ("countries",):
                raise ValueError("unsafe run path")
            p = (base / rel).resolve()
            if not p.is_relative_to((base / "countries").resolve()) or p in paths:
                raise ValueError("invalid or duplicate run path")
            paths.append(p)
    expected = infer._ids(data, "train")
    runs = infer._runs(data, paths, "train", expected)
    cfg = runs[0]["manifest"]["config"]
    numeric = cfg.get("numeric")
    gate, neural, model, ks = (cfg.get(k) for k in ("gate", "neural", "model", "k"))
    if (cfg.get("version") != 1 or not all(isinstance(x, dict) for x in (numeric, gate, neural, model, ks)) or
            not isinstance(cfg.get("retrievers"), list) or not cfg["retrievers"] or
            not all(isinstance(x, dict) and {"model", "revision", "feature", "format", "family"} <= x.keys()
                    for x in cfg["retrievers"])):
        raise ValueError("not a saved hybrid production run")
    blend = numeric.get("blend")
    if not isinstance(blend, dict):
        raise ValueError("invalid hybrid numeric configuration")
    weight = blend.get("neural_weight")
    if (numeric.get("gate") != "mean_probability" or numeric.get("neural") != "sigmoid_probability" or
            numeric.get("topk_ties") != "qid_asc" or blend.get("method") != "weighted_logit" or
            isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= 1 or
            not all(isinstance(x.get("sha256"), str) and len(x["sha256"]) == 64 for x in (gate, neural)) or
            model.get("sha256") != _hash({"gate": gate["sha256"], "neural": neural["sha256"], "neural_weight": weight}) or
            any(o.get("neural_weight") != weight or o.get("k_lex") != ks.get("lexical") or
                o.get("k_dense") != ks.get("dense") or o.get("k_gate") != ks.get("gate") for o in owners)):
        raise ValueError("not a saved hybrid production run")
    source = {"indices": indices, "data": str(data), "data_meta_sha256": infer._sha(data / "meta.json"),
              "config": cfg,
              "config_sha256": runs[0]["manifest"]["config_sha256"],
              "runs": [{"path": str(r["dir"]), "manifest_sha256": infer._sha(r["dir"] / "manifest.json")} for r in runs]}
    return runs, source


def _truth(data):
    ref = pl.read_parquet(data / "train/ref.parquet", columns=["rid", "deg", "fold", "co", "blank"])
    if not np.array_equal(ref["rid"].to_numpy(), np.arange(len(ref), dtype=np.uint32)) or not {0, 1}.issubset(set(ref["fold"])):
        raise ValueError("invalid reference ids or missing tune/audit folds")
    owners, blanks = [], []
    start = 0
    for sr in (2, 3):
        t = pl.scan_parquet(data / "train" / f"s{sr}.parquet").select(
            "rid", "own", pl.col("ad").fill_null("").str.strip_chars().eq("").alias("blank")
        ).collect(engine="streaming")
        if not np.array_equal(t["rid"].to_numpy(), np.arange(start, start + len(t), dtype=np.uint32)):
            raise ValueError("invalid target ids")
        owners.append(t["own"].to_numpy().astype(np.int32, copy=False))
        blanks.append(t["blank"].to_numpy())
        start += len(t)
    own = np.concatenate(owners)
    if (own < -1).any() or (own >= len(ref)).any():
        raise ValueError("invalid target ownership")
    if not np.array_equal(np.bincount(own[own >= 0], minlength=len(ref)), ref["deg"].to_numpy()):
        raise ValueError("target ownership disagrees with reference degrees")
    return ref, own, np.concatenate(blanks)


def _batches(runs, own, fold, which):
    cols = ["qid", "tid", "gate_prob", "neural_prob", "prob", "y", "sr"]
    for run in runs:
        for p in run["parts"]:
            meta = next(x for x in run["manifest"]["parts"] if run["dir"] / x["name"] == p)
            if not set(cols).issubset(pq.ParquetFile(p).schema.names):
                raise ValueError(f"missing saved hybrid scores {p}")
            last, carry = -1, None
            for batch in pq.ParquetFile(p).iter_batches(batch_size=65536, columns=cols):
                d = pl.from_arrow(batch)
                if carry is not None:
                    d = pl.concat([carry, d])
                t = d["tid"].to_numpy()
                if len(t) == 0:
                    continue
                if (t[1:] < t[:-1]).any() or t[0] <= last:
                    raise ValueError(f"unsorted targets {p}")
                cut = int(np.searchsorted(t, t[-1], side="left"))
                carry = d.slice(cut)
                d = d.slice(0, cut)
                if len(d):
                    last = int(d["tid"][-1])
                    yield _valid(d, own, fold, which, meta)
            if carry is not None:
                yield _valid(carry, own, fold, which, meta)


def _valid(d, own, fold, which, meta):
    q, t, y = (d[x].to_numpy() for x in ("qid", "tid", "y"))
    if len(t) and ((t < 0).any() or (q < 0).any() or t.max() >= len(own) or q.max() >= len(fold)):
        raise ValueError("saved pair outside prepared ids")
    chosen = np.isin(fold[q], which)
    if len(t) and ((d["sr"] != meta["source"]).any() or
                   ((own[t[chosen]] == q[chosen]).astype(np.uint8) != y[chosen]).any()):
        raise ValueError("saved pair labels/source disagree with prepared truth")
    for name in ("gate_prob", "neural_prob", "prob"):
        x = d[name].to_numpy()
        if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError(f"invalid {name}")
    return q, t, y, d["gate_prob"].to_numpy(), d["neural_prob"].to_numpy(), d["prob"].to_numpy()


def _top(q, t, y, score, fold, allowed):
    # rank against every reference first; fold filtering happens only afterwards.
    ix = np.lexsort((q, -score, t))
    ix = ix[np.r_[True, t[ix][1:] != t[ix][:-1]]]
    ix = ix[allowed[fold[q[ix]]]]
    return q[ix], t[ix], y[ix], score[ix]


def _join(parts):
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4)) if parts else (
        np.empty(0, np.uint32), np.empty(0, np.uint32), np.empty(0, np.uint8), np.empty(0, np.float32))


def _fixed(rows, deg, mask, threshold):
    q, _, y, score = rows
    chosen = mask[q] & (score >= threshold)
    pn = np.bincount(q[chosen], minlength=len(deg))
    tp = np.bincount(q[chosen], weights=y[chosen], minlength=len(deg))
    d = deg[mask]
    f = np.where(d == 0, (pn[mask] == 0).astype(float), 1.25 * tp[mask] / np.maximum(pn[mask] + .25 * d, 1))
    n, hit, truth = int(pn[mask].sum()), int(tp[mask].sum()), int(d.sum())
    return {"threshold": float(threshold), "macro_f05": float(f.mean()), "pair_precision": hit / n if n else 1.,
            "pair_recall": hit / truth if truth else 1., "pairs": n, "true_pairs": truth}


def _curve(rows, deg, mask):
    q, _, y, score = rows
    take = mask[q]
    q, y, score = q[take], y[take].astype(np.int64), score[take]
    n = int(mask.sum())
    if not n:
        raise ValueError("empty threshold population")
    base, truth = int(((deg == 0) & mask).sum()), int(deg[mask].sum())
    if not len(q):
        return {"threshold": float(np.nextafter(1., np.inf)), "macro_f05": base / n,
                "pair_precision": 1., "pair_recall": 0. if truth else 1., "pairs": 0, "true_pairs": truth}
    ix = np.lexsort((-score, q))
    q, y, score = q[ix], y[ix], score[ix]
    start = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
    first = np.zeros(len(q), dtype=bool)
    first[start] = True
    groups = np.maximum.accumulate(np.where(first, np.arange(len(q)), 0))
    pred = np.arange(len(q)) - groups + 1
    hits = np.cumsum(y)
    hits -= np.where(groups > 0, np.r_[0, np.cumsum(y)[:-1]][groups], 0)
    d = deg[q]
    f = np.where(d == 0, 0., 1.25 * hits / (pred + .25 * d))
    before = np.empty(len(q), dtype=np.float64)
    before[start] = (d[start] == 0).astype(float)
    other = np.ones(len(q), dtype=bool)
    other[start] = False
    before[other] = f[np.flatnonzero(other) - 1]
    delta = f - before
    ix = np.argsort(-score, kind="stable")
    score, delta, y = score[ix], delta[ix], y[ix]
    ends = np.r_[np.flatnonzero(score[1:] != score[:-1]), len(score) - 1]
    totals = np.clip(base + np.cumsum(delta)[ends], 0, n)
    hits = np.cumsum(y)[ends]
    selected = ends + 1
    precision = hits / selected
    initial = {"threshold": float(np.nextafter(float(score[0]), np.inf)), "macro_f05": base / n,
               "pair_precision": 1., "pair_recall": 0. if truth else 1., "pairs": 0, "true_pairs": truth}
    best = max(range(len(ends)), key=lambda i: (totals[i] / n, precision[i], float(score[ends[i]])))
    result = {"threshold": float(score[ends[best]]), "macro_f05": float(totals[best] / n),
              "pair_precision": float(precision[best]), "pair_recall": float(hits[best] / truth) if truth else 1.,
              "pairs": int(selected[best]), "true_pairs": truth}
    return max((initial, result), key=lambda z: (z["macro_f05"], z["pair_precision"], z["threshold"]))


def _scan(runs, ref, own, which, model=None, fit=False):
    fold = ref["fold"].to_numpy()
    allowed = np.isin(np.arange(3), which)
    stored = {w: [] for w in weights} if model is None else {"stack": []}
    stored["v1"] = []
    keep = np.zeros(len(own), dtype=bool)
    fit_x, fit_y, fit_w = [], [], []
    for q, t, y, gate, neural, prob in _batches(runs, own, fold, which):
        if 0 in which:
            eligible = fold[q] == 0
            keep[t[eligible][y[eligible] == 1]] = True
        base = _blend(gate, neural, .6)
        if not np.allclose(prob, _blend(gate, neural, runs[0]["manifest"]["config"]["numeric"]["blend"]["neural_weight"]), atol=2e-6):
            raise ValueError("saved probability contradicts hybrid configuration")
        stored["v1"].append(_top(q, t, y, base, fold, allowed))
        if model is None:
            for w in weights:
                stored[w].append(_top(q, t, y, _blend(gate, neural, w), fold, allowed))
        if fit:
            # no fold1/2 owner targets enter meta fit; tune still sees the full target pool.
            fit_indices = np.flatnonzero(_fit_edges(q, t, own, fold))
            selected = fit_indices[(y[fit_indices] == 1) | _sample(t[fit_indices])]
            if len(selected):
                fit_x.append(np.column_stack((_logit(gate[selected]), _logit(neural[selected]))))
                fit_y.append(y[selected])
                fit_w.append(np.where(y[selected] == 1, 1., 50.).astype(np.float32))
        if model is not None:
            stored["stack"].append(_top(q, t, y, _stack(gate, neural, model), fold, allowed))
    rows = {key: _join(value) for key, value in stored.items()}
    if not fit:
        return rows, keep, None
    from sklearn.linear_model import LogisticRegression as logistic_regression
    x, y, w = np.concatenate(fit_x), np.concatenate(fit_y), np.concatenate(fit_w)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("meta fit needs positive and negative pairs")
    learner = logistic_regression(C=1., max_iter=200).fit(x, y, sample_weight=w)
    if not learner.n_iter_[0] < 200:
        raise ValueError("meta fit did not converge")
    model = {"features": ["gate_logit", "neural_logit"], "coefficients": learner.coef_[0].tolist(),
             "intercept": float(learner.intercept_[0]), "epsilon": float(np.finfo(np.float32).eps),
             "fit_pairs": len(y), "fit_positive": int(y.sum()), "fit_negative_sample": "target hash modulo 50; inverse-probability weight 50",
              "split": "even deterministic group hash held out; fold0 anchors and fold0-owned/orphan target groups only; crossing edges and fold1/2-owned target negatives excluded"}
    return rows, keep, model


def _strata(ref, own, blank, keep, base, deg):
    fold, countries = ref["fold"].to_numpy(), ref["co"].to_numpy()
    q, t, y, s = base
    true = (own >= 0) & (fold[np.maximum(own, 0)] == 0)
    top, at = np.zeros(len(own), dtype=bool), np.zeros(len(own), dtype=bool)
    top[t[y == 1]] = True
    at[t[(y == 1) & (s >= .8)]] = True
    selected = (s >= .8)
    pred = np.bincount(q[selected], minlength=len(ref))
    tp = np.bincount(q[selected & (y == 1)], minlength=len(ref))

    def errors(target_mask):
        truth = int(target_mask.sum())
        retained = int((keep & target_mask).sum())
        return {"linked_targets": truth, "postgate_true_targets": retained,
                "gate_lost": int((target_mask & ~keep).sum()),
                "wrong_top1": int((target_mask & keep & ~top).sum()),
                "cutoff_lost": int((target_mask & top & ~at).sum()),
                "postgate_recall": retained / truth if truth else 1.}

    def summary(mask):
        d = deg[mask]
        f = np.where(d == 0, (pred[mask] == 0).astype(float), 1.25 * tp[mask] / np.maximum(pred[mask] + .25 * d, 1))
        return {"anchors": int(mask.sum()), "macro_f05_at_0_8": float(f.mean()) if len(d) else None,
                **errors(true & mask[np.maximum(own, 0)])}

    tune = fold == 0
    result = {"overall": summary(tune), "by_country": {}, "by_anchor_has_blank_alias": {},
              "by_target_blank_address": {}, "by_singleton_anchor": {}}
    for co in np.unique(countries[tune]):
        mask = tune & (countries == co)
        result["by_country"][str(co)] = summary(mask)
    for value in (False, True):
        mask = tune & (ref["blank"].to_numpy().astype(bool) == value)
        result["by_anchor_has_blank_alias"][str(value).lower()] = summary(mask)
        result["by_target_blank_address"][str(value).lower()] = errors(true & (blank == value))
        result["by_singleton_anchor"][str(value).lower()] = summary(tune & ((deg == 0) == value))
    return result


def _report(z):
    lines = ["# overnight full-pool diagnosis", "", "complete training target corpus; target top1 is chosen before anchor filtering.",
             "fold0 owner groups split for stack fit and separate cutoff selection; fold1 stays locked until --audit.",
             "base models trained on fold2 and developed on fold0; fold0 blend comparisons reuse development labels.",
             "meta fit uses sampled fold0-owned/orphan negatives with inverse sampling weights; fold1/2-owned targets are excluded from meta fit but included in full-pool tune negatives. no raw features regenerated.", "",
             f"selected: {z['selection']['variant']} @ {z['selection']['threshold']:.7g} (held-out fold0 macro {z['selection']['tune']['macro_f05']:.6f})",
             f"frozen v1 w=.6 @ .8: {z['v1']['macro_f05']:.6f} fold0 macro", "",
             "| variant | held-out fold0 macro | full fold0 macro | cutoff |", "| --- | ---: | ---: | ---: |"]
    for k, v in z["variants"].items():
        lines.append(f"| {k} | {v['tune']['macro_f05']:.6f} | {v['fold0_at_tune_cut']['macro_f05']:.6f} | {v['tune']['threshold']:.7g} |")
    stats = z["losses"]["overall"]
    lines += ["", f"postgate oracle recall: {stats['postgate_recall']:.6f}; gate-lost: {stats['gate_lost']}; wrong-top1: {stats['wrong_top1']}; cutoff-lost: {stats['cutoff_lost']}.",
              "loss tables by country, anchor with any blank alias (macro), individual blank target (errors only), and singleton anchor (=degree 0) are in the json report.",
              "oracle is constrained to saved postgate pairs, not the earlier lexical candidate pool."]
    if "audit" in z:
        lines += ["", f"one-time fold1 audit: finalist {z['audit']['finalist']['macro_f05']:.6f}, frozen v1 {z['audit']['v1']['macro_f05']:.6f} macro f0.5."]
    return "\n".join(lines) + "\n"


def run(data, bases, out, report, audit=False):
    data = data.resolve()
    runs, source = _input(data, bases)
    ref, own, blank = _truth(data)
    deg, fold = ref["deg"].to_numpy(), ref["fold"].to_numpy()
    selection_file = out / "selection.json"
    if audit:
        if not selection_file.is_file() or not report.with_suffix(".json").is_file():
            raise ValueError("freeze a fold0 selection before audit")
        if (out / "audit.json").exists() or (out / "audit.started").exists():
            raise ValueError("fold1 audit has already been recorded")
        saved = infer._json(selection_file)
        if saved["source"] != source:
            raise ValueError("frozen selection input changed")
        chosen = saved["selection"]
        with (out / "audit.started").open("x", encoding="utf-8") as lock:
            lock.write(infer._sha(selection_file) + "\n")
        model = saved["stack"] if chosen["variant"] == "stack" else None
        rows, _, _ = _scan(runs, ref, own, (1,), model=model) if model else _scan(runs, ref, own, (1,))
        variant = rows["stack"] if model else rows[float(chosen["variant"].split("w=")[1])]
        m = fold == 1
        result = {"finalist": _fixed(variant, deg, m, chosen["threshold"]),
                  "v1": _fixed(rows["v1"], deg, m, .8), "source": source, "selection_sha256": infer._sha(selection_file)}
        infer._write(out / "audit.json", result)
        z = infer._json(report.with_suffix(".json"))
        z["audit"] = result
    else:
        if selection_file.exists():
            raise ValueError("selection already frozen; use --audit")
        rows, keep, model = _scan(runs, ref, own, (0,), fit=True)
        stack_rows, _, _ = _scan(runs, ref, own, (0,), model=model)
        rows["stack"] = stack_rows["stack"]
        full, tune = fold == 0, (fold == 0) & _held(np.arange(len(ref)))
        variants = {}
        for key, scores in rows.items():
            if key == "v1":
                continue
            name = "stack" if key == "stack" else f"w={key:.1f}"
            best = _curve(scores, deg, tune)
            variants[name] = {"tune": best, "fold0_at_tune_cut": _fixed(scores, deg, full, best["threshold"]),
                              "fold0_optimized_descriptive": _curve(scores, deg, full)}
        winner = max(variants, key=lambda name: (variants[name]["tune"]["macro_f05"],
                                                 variants[name]["tune"]["pair_precision"], name))
        selection = {"variant": winner, "threshold": variants[winner]["tune"]["threshold"],
                     "tune": variants[winner]["tune"]}
        retained = np.bincount(own[keep], minlength=len(ref))
        oracle_f = np.where(deg == 0, 1., 1.25 * retained / np.maximum(retained + .25 * deg, 1))
        truth = int(deg[full].sum())
        z = {"scope": "complete hybrid postgate train target pool; source1 macro f0.5",
             "source": source, "selection": selection, "stack": model, "variants": variants,
             "v1": _fixed(rows["v1"], deg, full, .8),
             "postgate_oracle": {"macro_f05": float(oracle_f[full].mean()),
                                 "pair_recall": int(retained[full].sum()) / truth if truth else 1.},
             "losses": _strata(ref, own, blank, keep, rows["v1"], deg),
             "limitations": ["base models fit fold2 but developed on fold0; fold0 screening is not independent",
                             "stack fitted only on fold0 owner/ref-disjoint groups, sampled negatives weighted 50; excluding other-owner-fold negatives shifts meta fit calibration; fold1 audit untouched",
                             "target-top1 is computed on all references, including non-tune folds, before filtering"]}
        infer._write(selection_file, {"source": source, "selection": selection, "stack": model})
    infer._write(report.with_suffix(".json"), z)
    report.with_suffix(".md").parent.mkdir(parents=True, exist_ok=True)
    report.with_suffix(".md").write_text(_report(z), encoding="utf-8")
    print(json.dumps({"selection": z["selection"], "audit": z.get("audit"), "report": str(report.with_suffix('.json'))}, indent=2))


def check():
    assert os.environ["POLARS_MAX_THREADS"] == os.environ["ARROW_NUM_THREADS"]
    assert all(os.environ[key] == "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
    assert np.allclose(_blend(np.array([.2], np.float32), np.array([.8], np.float32), .5), [.5])
    strata_ref = pl.DataFrame({"rid": [0, 1, 2, 3], "deg": [0, 2, 1, 0], "blank": [0, 1, 0, 0],
                               "fold": [0, 0, 0, 1], "co": ["us"] * 4})
    strata_own = np.array([1, 1, 2, -1])
    strata_blank = np.array([True, False, False, True])
    strata_keep = np.array([True, False, True, False])
    strata_rows = (np.array([1, 2, 0, 0]), np.arange(4), np.array([1, 0, 0, 0]), np.full(4, .9))
    loss = _strata(strata_ref, strata_own, strata_blank, strata_keep, strata_rows,
                   strata_ref["deg"].to_numpy())
    assert loss["by_singleton_anchor"]["true"]["anchors"] == 1
    assert loss["by_singleton_anchor"]["true"]["linked_targets"] == 0
    assert loss["by_anchor_has_blank_alias"]["true"]["linked_targets"] == 2
    assert np.isclose(loss["by_anchor_has_blank_alias"]["true"]["macro_f05_at_0_8"], 1.25 / 1.5)
    assert loss["by_target_blank_address"]["true"]["linked_targets"] == 1
    assert loss["by_target_blank_address"]["false"]["gate_lost"] == 1
    assert loss["by_target_blank_address"]["false"]["wrong_top1"] == 1
    assert "macro_f05_at_0_8" not in loss["by_target_blank_address"]["true"]
    ref = pl.DataFrame({"rid": [0, 1, 2, 3], "deg": [0, 2, 1, 0], "fold": [0, 0, 0, 1]})
    q = np.array([1, 2, 1, 0, 2], np.uint32)
    t = np.array([10, 10, 11, 12, 13], np.uint32)
    y = np.array([1, 0, 1, 0, 0], np.uint8)
    s = np.array([.8, .9, .8, .1, .9], np.float32)
    top = _top(q, t, y, s, ref["fold"].to_numpy(), np.array([True, False, False]))
    assert top[0].tolist() == [2, 1, 0, 2]  # fold filtering must not rescue the lower-ranked true pair.
    mask = np.array([True, True, True, False])
    deg = ref["deg"].to_numpy()
    fixed = _fixed(top, deg, mask, .8)
    assert np.isclose(fixed["macro_f05"], (1 + 1.25 / 1.5) / 3)
    best = _curve(top, deg, mask)
    assert np.isclose(best["macro_f05"], fixed["macro_f05"])
    rng = np.random.default_rng(12)
    for _ in range(20):
        qid = rng.integers(0, 8, 30, dtype=np.uint32)
        scores = rng.choice(np.array([.1, .5, .9], np.float32), 30)
        sample = (qid, np.arange(30, dtype=np.uint32), rng.integers(0, 2, 30, dtype=np.uint8), scores)
        degrees = rng.integers(0, 4, 8)
        all_anchors = np.ones(8, bool)
        got = _curve(sample, degrees, all_anchors)
        brute = max((_fixed(sample, degrees, all_anchors, float(cut)) for cut in (1.1, .1, .5, .9)),
                    key=lambda result: result["macro_f05"])
        assert np.isclose(got["macro_f05"], brute["macro_f05"])
    ids = np.arange(10000)
    owner = np.array([1, 1, -1, 2])
    groups = np.where(owner >= 0, owner, np.arange(4) + 10)
    fit = ~_held(ids)[:, None] & ~_held(groups)[None, :]
    tune = _held(ids)[:, None] & _held(groups)[None, :]
    assert not (fit & tune).any() and groups[0] == groups[1]
    assert not np.intersect1d(ids[np.where(fit)[0]], ids[np.where(tune)[0]]).size
    assert not np.intersect1d(groups[np.where(fit)[1]], groups[np.where(tune)[1]]).size
    fold = np.array([0] * 24 + [1, 2], np.uint8)
    fit_qid = int(np.flatnonzero(~_held(np.arange(24)))[0])
    fit_own = np.array([24, 25, fit_qid, -1])
    edges = _fit_edges(np.full(4, fit_qid), np.arange(4), fit_own, fold)
    assert edges.tolist()[:3] == [False, False, True]
    with tempfile.TemporaryDirectory() as tmp:
        import run as runner
        import match as matcher
        import sys
        import types
        from unittest.mock import patch
        home = path(tmp)
        data = home / "data"
        (data / "train").mkdir(parents=True)
        infer._write(data / "meta.json", {"check": True})
        nref, ntarget = 30, 3000
        own = np.arange(ntarget, dtype=np.int32) % 26
        own[np.arange(ntarget) % 3 != 0] = -1
        degrees = np.bincount(own[own >= 0], minlength=nref)
        truth_blank = np.bincount(own[(np.arange(ntarget) % 7 == 0) & (own >= 0)], minlength=nref) > 0
        refs = pl.DataFrame({"rid": np.arange(nref, dtype=np.uint32), "deg": degrees,
                             "fold": np.array([0] * 24 + [1] * 2 + [2] * 4, np.uint8),
                             "co": ["us"] * nref, "blank": truth_blank.astype(np.uint8)})
        refs.write_parquet(data / "train/ref.parquet")
        for sr, ids in ((2, np.arange(0, 1500)), (3, np.arange(1500, ntarget))):
            pl.DataFrame({"rid": ids.astype(np.uint32), "own": own[ids],
                          "ad": np.where(ids % 7 == 0, "", "main")}).write_parquet(data / "train" / f"s{sr}.parquet")
        gate_dir, neural_dir = home / "gate", home / "neural"
        gate_dir.mkdir(); neural_dir.mkdir()
        (gate_dir / "weights.bin").write_bytes(b"gate")
        (neural_dir / "weights.bin").write_bytes(b"neural")
        with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
            cfg, config_hash = matcher._cfg(data, {"sha256": "a" * 64, "feature_names": []},
                                             {"sha256": "b" * 64},
                                             [{"model": matcher.embed.mod0, "revision": matcher.embed.rev0}],
                                             10, 50, 20, .6, "cpu")
        bases = []
        for shard in range(4):
            base = home / f"q{shard}" / "train"
            lo, hi = shard * 750, (shard + 1) * 750
            owner = runner._owner(data, gate_dir, neural_dir, "train", ["us"], lo, hi, 10, 50, 20,
                                  ["e5"], "cpu", 2, 64, 32, 4096, .6)
            base.mkdir(parents=True)
            index = runner._index(base, owner, False)
            rel = runner._runrel(runner._dir("us"))
            rd = base / rel
            (rd / "parts").mkdir(parents=True)
            ids = np.arange(lo, hi, dtype=np.uint32)
            q, t, labels, scores = [], [], [], []
            for tid in ids:
                for qid in (int(tid % 24), int((tid + 7) % 30), int(own[tid])):
                    if qid < 0 or (tid == 0 and qid == own[tid]) or (qid in q[-2:] and t[-1] == tid):
                        continue
                    q.append(qid); t.append(int(tid)); labels.append(int(qid == own[tid]))
                    scores.append(.85 if qid == own[tid] else .12)
            gate = np.array(scores, np.float32)
            neural = np.where(np.array(labels) == 1, .8, .2).astype(np.float32)
            part = rd / "parts/part_0000000.parquet"
            pl.DataFrame({"qid": np.array(q, np.uint32), "tid": np.array(t, np.uint32),
                          "gate_prob": gate, "neural_prob": neural, "prob": _blend(gate, neural, .6),
                          "y": np.array(labels, np.uint8), "sr": np.where(np.array(t) < 1500, 2, 3).astype(np.uint8)}).write_parquet(part)
            coverage = rd / "parts/part_0000000.npy"
            infer._npy(coverage, ids)
            manifest = {"version": infer.ver, "split": "train", "config": cfg, "config_sha256": config_hash,
                        "parts": [{"name": "parts/part_0000000.parquet", "coverage": "parts/part_0000000.npy",
                                   "queries": len(ids), "pairs": len(t), "source": 2 if shard < 2 else 3,
                                   "pair_sha256": infer._sha(part), "coverage_sha256": infer._sha(coverage)}]}
            infer._write(rd / "manifest.json", manifest)
            index.update({"complete": True,
                          "verification": {"status": "exact_coverage", "full": False, "targets": len(ids)},
                          "runs": [{"country": "us", "rid_start": lo, "rid_stop": hi,
                                    "runpath": rel, "config": owner, "childstatus": "succeeded", "returncode": 0,
                                    "command": []}]})
            infer._write(base / "runs.json", index)
            bases.append(base)
        runs, source = _input(data, bases)
        assert len(runs) == 4 and source["config"] == cfg
        for base in bases:
            manifest_file = base / runner._runrel(runner._dir("us")) / "manifest.json"
            changed = infer._json(manifest_file)
            changed["config"]["model"]["sha256"] = "0" * 64
            changed["config_sha256"] = _hash(changed["config"])
            infer._write(manifest_file, changed)
        try:
            _input(data, bases)
        except ValueError as error:
            assert "not a saved hybrid production run" in str(error)
        else:
            raise AssertionError("invalid hybrid model hash accepted")
        for base in bases:
            manifest_file = base / runner._runrel(runner._dir("us")) / "manifest.json"
            changed = infer._json(manifest_file)
            changed["config"] = cfg
            changed["config_sha256"] = config_hash
            infer._write(manifest_file, changed)
        out, report = home / "out", home / "report"
        run(data, bases, out, report)
        first = infer._json(report.with_suffix(".json"))
        assert first["losses"]["overall"]["gate_lost"] == 1
        assert first["postgate_oracle"]["macro_f05"] < 1
        assert first["stack"]["fit_positive"] > 0 and "audit" not in first
        run(data, bases, out, report, audit=True)
        assert "audit" in infer._json(report.with_suffix(".json"))
        try:
            run(data, bases, out, report, audit=True)
        except ValueError as error:
            assert "already been recorded" in str(error)
        else:
            raise AssertionError("repeated audit accepted")
    print("fullgap checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--data", type=path, default=root / "cache/data")
    parser.add_argument("--runs", type=path, nargs=4, default=[root / f"artifacts/upgraded-train-q{i}/train" for i in range(4)])
    parser.add_argument("--out", type=path, default=root / "artifacts/overnight-fullpool")
    parser.add_argument("--report", type=path, default=root / "reports/overnight-fullpool")
    args = parser.parse_args()
    _threads(args.threads)
    _imports()
    if args.check:
        check()
        return
    if args.audit and not (args.out / "selection.json").exists():
        raise ValueError("run fold0 selection before --audit")
    # four independently downloaded run indices may each own a target subset;
    # inspect their runpaths, then let infer._runs verify their exact union.
    run(data=args.data, bases=args.runs, out=args.out, report=args.report, audit=args.audit)


if __name__ == "__main__":
    main()
