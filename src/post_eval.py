'development-only scorer checks and explicit orphan-density stress tests'
import argparse as ap
from hashlib import sha256
import json
from pathlib import Path as path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

import decode
import infer
import post


def metric(q, y, keep, deg, anchors):
    count = np.bincount(q[keep], minlength=len(deg))
    true = np.bincount(q[keep], weights=y[keep], minlength=len(deg))
    denominator = count + .25 * deg
    f = np.divide(1.25 * true, denominator, out=np.ones(len(deg)), where=denominator > 0)
    sel = int(count.sum())
    truth = int(deg[anchors].sum())
    return {"macro_f05": float(f[anchors].mean()), "pair_precision": float(true.sum() / sel) if sel else 1.,
            "pair_recall": float(true.sum() / truth) if truth else 1., "pairs": sel, "true_pairs": truth}


def run(prepared, out, threads=8):
    prepared, out = path(prepared), path(out)
    src = post.verified(prepared, "train")
    data = path(src["data"])
    refs = pl.read_parquet(data / "train/ref.parquet", columns=["rid", "fold", "deg"])
    deg = refs["deg"].to_numpy()
    valid_partition = src.get("validation_partition", {"modulus": 2, "remainder": 0})
    anchors = (refs["fold"].to_numpy() == 0) & (post.partition(refs["rid"].to_numpy(), valid_partition["modulus"]) == valid_partition["remainder"])
    if not anchors.any():
        raise ValueError("no development anchors")
    recipe = {"neural_weight": .6, "unseen_gate": False, "countries": sorted(src["fit_reference_counts"]),
              "edges": np.linspace(post.logit([.02])[0], post.logit([.999])[0], 25).tolist()}
    recipe["partition"] = src.get("calibration_partition", {"modulus": 2, "remainder": 1})
    recipe["known_score"] = "stack_prob" if src.get("stack_model") else "blend"
    train = post.histograms(prepared, recipe, fit=True)
    orphan = post.histograms(prepared, recipe, fit=True, orphans=True)
    counts = src["fit_reference_counts"]
    empirical = {**recipe, "curves": post.curves(train, train, counts, counts, np.asarray(recipe["edges"]), transfer=False)}
    shifted_counts = {k: [v[0] + orphan[k][0], v[1]] for k, v in train.items()}
    shifted = {**recipe, "curves": post.curves(train, shifted_counts, counts, counts, np.asarray(recipe["edges"]))}
    parts = {k: [] for k in ("qid", "tid", "y", "orphan", "control", "raw", "empirical", "shifted")}
    columns = ["qid", "tid", "y", "own", "prob", "gate_prob", "neural_prob", "co", "seg"]
    if recipe["known_score"] == "stack_prob":
        columns.append("stack_prob")
    for batch in pq.ParquetFile(prepared).iter_batches(batch_size=262144, columns=columns):
        d = batch.to_pydict()
        co, seg = np.asarray(d["co"]), np.asarray(d["seg"])
        raw = post.score_rows(d, recipe)
        for key in ("qid", "tid"):
            parts[key].append(np.asarray(d[key], np.uint32))
        parts["y"].append(np.asarray(d["y"], np.uint8))
        parts["orphan"].append(np.asarray(d["own"]) < 0)
        parts["control"].append(np.asarray(d["prob"], np.float32))
        parts["raw"].append(np.asarray(d["stack_prob"] if recipe["known_score"] == "stack_prob" else d["prob"], np.float32))
        parts["empirical"].append(post.adjust(raw, co, seg, empirical))
        parts["shifted"].append(post.adjust(raw, co, seg, shifted))
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    del parts
    q, t, y = arrays["qid"], arrays["tid"], arrays["y"]
    rs = {}
    for label, name, cut in (("base_threshold", "control", .8), ("raw_expected", "raw", None),
                                   ("empirical_expected", "empirical", None), ("shift_corrected_expected", "shifted", None)):
        p = arrays[name]
        top = decode.winners(q, t, p, arrays["raw"])
        top = top[anchors[q[top]]]
        for stress in (False, True):
            ids = np.r_[top, top[arrays["orphan"][top]]] if stress else top
            qi, ti, yi, pi, raw = q[ids], t[ids].copy(), y[ids], p[ids], arrays["raw"][ids]
            if stress:
                ti[len(top):] += int(t.max()) + 1
            if cut is None:
                keep, details = decode.choose(qi, ti, pi, raw, threads=threads)
            else:
                keep, details = pi >= cut, {}
            rs[f"{label}_{'double_orphans' if stress else 'original'}"] = {**metric(qi, yi, keep, deg, anchors), **details}
    res = {"scope": "fold0 held-out reference development, full-target competitors; transductive target reuse; no fold1 audit",
              "control": {"score_column": "prob", "threshold": .8, "scope": "current base-score configuration"},
              "data_meta_sha256": src["data_meta_sha256"],
              "anchors_sha256": sha256(refs["rid"].to_numpy()[anchors].astype("<u4").tobytes()).hexdigest(),
              "anchors": int(anchors.sum()), "source_score_sha256": src["score_sha256"], "validation_partition": valid_partition,
              "known_score": recipe["known_score"],
              "stress": "duplicate every orphan target once with a distinct synthetic id; true links unchanged; not real test labels",
              "results": rs, "public_score": None}
    infer._write(out, res)
    print(json.dumps(res, indent=2))
    return res


def check():
    q, y = np.array([0, 0, 1]), np.array([1, 0, 0])
    deg, mask = np.array([1, 0]), np.array([True, True])
    assert metric(q, y, np.array([True, False, False]), deg, mask)["macro_f05"] == 1.
    assert metric(q, y, np.array([True, False, True]), deg, mask)["macro_f05"] == .5
    print("postprocessor development checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=path, default=path("artifacts/post-train.parquet"))
    p.add_argument("--out", type=path, default=path("reports/post-development.json"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        check()
    else:
        run(a.scores, a.out, a.threads)
