"""versioned score calibration, country routing and source1 set decoding"""
import argparse as ap
import hashlib as hh
import json
import tempfile as tf
from pathlib import Path as path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression as isotonic

import decode
import infer


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def partition(ids, modulus=2):
    x = np.asarray(ids, dtype=np.uint64)
    return ((x * np.uint64(2654435761)) >> np.uint64(13)) % modulus


def held(ids):
    return partition(ids) == 0


def inputs(data, roots, split):
    runs = []
    for root in roots:
        root = path(root).resolve()
        if root.is_dir() and (root / "manifest.json").is_file():
            runs.append(root)
            continue
        p = root if root.is_file() else root / "runs.json"
        index = infer._json(p)
        if index.get("split", index.get("owner", {}).get("split")) != split:
            raise ValueError("score index split mismatch")
        if "complete" in index and not index["complete"]:
            raise ValueError("incomplete score index")
        for row in index["runs"]:
            rel = path(row if isinstance(row, str) else row["runpath"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("unsafe score index path")
            runs.append(p.parent / rel)
    return infer._runs(data, runs, split, infer._ids(data, split))


def prepare(data, roots, split, out):
    data, out = path(data).resolve(), path(out).resolve()
    if out.exists() or out.with_suffix(".json").exists():
        raise ValueError("prepared score output already exists")
    runs = inputs(data, roots, split)
    cols = ["qid", "tid", "prob", "gate_prob", "neural_prob", "sr"] + (["y"] if split == "train" else [])
    scores = pl.scan_parquet([p for r in runs for p in r["parts"]], parallel="row_groups", low_memory=True).select(cols)
    ref = pl.scan_parquet(data / split / "ref.parquet").select(
        pl.col("rid").alias("qid"), "co", "fold",
        pl.col("ad").fill_null("").str.extract(r"(\d+)", 1)
          .str.strip_chars_start("0").replace("", "0").fill_null("").alias("hs"))
    target = pl.concat([pl.scan_parquet(data / split / f"s{i}.parquet").select(
        pl.col("rid").alias("tid"), pl.col("sr").alias("actual_sr"), "own",
        pl.col("ad").fill_null("").str.extract_all(r"\d+").list.eval(
            pl.element().str.strip_chars_start("0").replace("", "0")).alias("nums")) for i in (2, 3)])
    d = scores.join(ref, on="qid", how="left").join(target, on="tid", how="left")
    equal = (pl.col("hs") != "") & pl.col("nums").list.contains(pl.col("hs"))
    d = d.with_columns(pl.when(equal).then(0).when((pl.col("hs") != "") & (pl.col("nums").list.len() > 0))
                      .then(1).otherwise(2).cast(pl.UInt8).alias("seg"))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    d.drop("hs", "nums").sort("tid", "qid").sink_parquet(tmp, compression="zstd", row_group_size=262144)
    check = pl.scan_parquet(tmp)
    invalid = pl.col("co").is_null() | pl.col("own").is_null() | (pl.col("sr") != pl.col("actual_sr"))
    for col in ("prob", "gate_prob", "neural_prob"):
        invalid |= pl.col(col).is_null() | ~pl.col(col).is_finite() | ~pl.col(col).is_between(0, 1)
    if split == "train":
        invalid |= pl.col("y") != (pl.col("qid").cast(pl.Int64) == pl.col("own")).cast(pl.UInt8)
    if check.select(invalid.any()).collect().item():
        tmp.unlink()
        raise ValueError("invalid score ownership, probabilities or source ids")
    pairs = check.select(pl.len()).collect().item()
    tmp.replace(out)
    refs = pl.read_parquet(data / split / "ref.parquet", columns=["rid", "co", "fold"])
    source = {"version": 1, "split": split, "data": str(data), "data_meta_sha256": infer._sha(data / "meta.json"),
              "config": runs[0]["manifest"]["config"], "config_sha256": runs[0]["manifest"]["config_sha256"], "score_sha256": infer._sha(out),
              "runs": [{"path": str(r["dir"]), "sha256": infer._sha(r["dir"] / "manifest.json")} for r in runs],
              "targets": sum(len(r["coverage"]) for r in runs), "pairs": pairs,
              "reference_counts": dict(refs.group_by("co").len().iter_rows())}
    if split == "train":
        fit_ref = refs.filter((pl.col("fold") == 0) & pl.Series(~held(refs["rid"].to_numpy())))
        source["fit_reference_counts"] = dict(fit_ref.group_by("co").len().iter_rows())
    infer._write(out.with_suffix(".json"), source)
    return source


def verified(p, split=None, runs=None):
    p = path(p)
    meta = infer._json(p.with_suffix(".json"))
    if meta.get("version") != 1 or (split and meta.get("split") != split) or infer._sha(p) != meta["score_sha256"]:
        raise ValueError("prepared scores changed or use the wrong split")
    if infer._sha(path(meta["data"]) / "meta.json") != meta["data_meta_sha256"]:
        raise ValueError("prepared data changed")
    if runs is not None:
        expected = {str(path(r).resolve()): infer._sha(path(r) / "manifest.json") for r in runs}
        recorded = {str(path(r["path"]).resolve()): r["sha256"] for r in meta["runs"]}
        if expected != recorded:
            raise ValueError("base score manifests changed since preparation")
    return meta


def score(gate, neural, country, recipe):
    weight = recipe["neural_weight"]
    if not 0 <= weight <= 1:
        raise ValueError("invalid neural weight")
    p = 1 / (1 + np.exp(-((1 - weight) * logit(gate) + weight * logit(neural))))
    if recipe["unseen_gate"]:
        p = np.where(np.isin(country, recipe["countries"]), p, gate)
    return p


def score_rows(rows, recipe):
    country = np.asarray(rows["co"])
    if recipe.get("known_score", "blend") not in ("blend", "stack_prob"):
        raise ValueError("unknown score policy")
    if not isinstance(recipe.get("unseen_gate"), bool):
        raise ValueError("invalid unseen-country policy")
    if recipe.get("known_score", "blend") == "stack_prob":
        p = np.asarray(rows["stack_prob"], dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("invalid stacked probabilities")
        return np.where(np.isin(country, recipe["countries"]), p, rows["gate_prob"]) if recipe["unseen_gate"] else p
    return score(np.asarray(rows["gate_prob"]), np.asarray(rows["neural_prob"]), country, recipe)


def histograms(p, recipe, fit=False, force_gate=False, orphans=False):
    result = {}
    cols = ["qid", "co", "seg", "gate_prob", "neural_prob"] + (["fold", "y"] if fit else [])
    if orphans:
        cols.append("own")
    if recipe.get("known_score") == "stack_prob" and not force_gate:
        cols.append("stack_prob")
    edges = np.asarray(recipe["edges"])
    for batch in pq.ParquetFile(p).iter_batches(batch_size=262144, columns=cols):
        d = batch.to_pydict()
        co, seg = np.asarray(d["co"]), np.asarray(d["seg"])
        part = recipe.get("partition", {"modulus": 2, "remainder": 1})
        use = ((np.asarray(d["fold"]) == 0) & (partition(d["qid"], part["modulus"]) == part["remainder"])) if fit else np.ones(len(co), bool)
        if orphans:
            use &= np.asarray(d["own"]) < 0
        gate = np.asarray(d["gate_prob"])
        p1 = gate if force_gate else score_rows(d, recipe)
        bins = np.searchsorted(edges, logit(p1), side="right")
        for country in np.unique(co[use]):
            for segment in range(3):
                take = use & (co == country) & (seg == segment)
                n = np.bincount(bins[take], minlength=len(edges) + 1).astype(float)
                y = np.bincount(bins[take], weights=np.asarray(d["y"])[take], minlength=len(edges) + 1) if fit else np.zeros_like(n)
                key = f"{country}|{segment}"
                if key not in result:
                    result[key] = [np.zeros_like(n), np.zeros_like(n)]
                result[key][0] += n
                result[key][1] += y
    return result


def curves(train, target, ntrain, ntarget, edges, minimum=50, transfer=True):
    """positive-density transfer with sparse-bin pooling and weighted isotonic fit"""
    centres = np.r_[edges[0] - 1, (edges[:-1] + edges[1:]) / 2, edges[-1] + 1]
    top = np.r_[-np.inf, edges] >= logit([.99])[0]
    countries = {k.rsplit("|", 1)[0] for k in target}
    result = {}
    for country in sorted(countries):
        pool = [country] if country in ntrain else sorted(ntrain)
        nh, nt = sum(ntrain[c] for c in pool), ntarget[country]
        if not nh or not nt:
            raise ValueError("empty reference calibration denominator")
        tn = [sum((train[f"{c}|{s}"][0] for c in pool), np.zeros(len(edges) + 1)) for s in range(3)]
        ty = [sum((train[f"{c}|{s}"][1] for c in pool), np.zeros(len(edges) + 1)) for s in range(3)]
        tests = [target.get(f"{country}|{s}", [np.zeros(len(edges) + 1), None])[0] for s in range(3)]
        top_pos = sum(y[top].sum() for y in ty)
        top_test = sum(n[top].sum() for n in tests)
        scale = (top_test / nt) / (top_pos / nh) if top_pos > 0 and top_test > 0 and min(top_pos, top_test) >= minimum else 1.
        if not transfer:
            scale = 1.
        for segment in range(3):
            count, positive, test = tn[segment], ty[segment], tests[segment]
            base = np.divide(positive, count, out=np.zeros_like(count), where=count > 0)
            target_positive = scale * positive / nh * nt
            posterior = np.divide(target_positive + minimum * base, test + minimum, out=base.copy(), where=test + minimum > 0)
            active = (count > 0) | (test > 0)
            if not active.any():
                values = 1 / (1 + np.exp(-centres))
            else:
                model = isotonic(out_of_bounds="clip").fit(centres[active], np.clip(posterior[active], 0, 1),
                                                           sample_weight=(test + minimum)[active])
                values = model.predict(centres)
            result[f"{country}|{segment}"] = {"centres": centres.tolist(), "posterior": values.tolist(),
                                              "positive_scale": float(scale), "pooled": country not in ntrain,
                                              "train_pairs": int(count.sum()), "test_pairs": int(test.sum())}
    return result


def fit(train, test, out, weight=.6, unseen=True):
    train, test, out = path(train), path(test), path(out)
    if out.exists():
        raise ValueError("calibration output already exists")
    tr, te = verified(train, "train"), verified(test, "test")
    if tr["config_sha256"] != te["config_sha256"] or tr["data_meta_sha256"] != te["data_meta_sha256"]:
        raise ValueError("train/test score configurations differ")
    recipe = {"version": 1, "kind": "segmented-postprocessor", "neural_weight": weight, "unseen_gate": unseen,
              "countries": sorted(tr["fit_reference_counts"]), "edges": np.linspace(logit([.02])[0], logit([.999])[0], 25).tolist(),
              "floor": .05, "exact_limit": 64, "config_sha256": tr["config_sha256"], "data_meta_sha256": tr["data_meta_sha256"],
              "sources": {"train": tr["score_sha256"], "test": te["score_sha256"]},
              "fit_scope": "fold0 fit-side anchors; full-target negatives; fold1 unused",
              "assumption": "positive score density transfers after scaling by high-score mass; unknown countries borrow labelled countries"}
    recipe["partition"] = tr.get("calibration_partition", {"modulus": 2, "remainder": 1})
    recipe["known_score"] = "stack_prob" if tr.get("stack_model") else "blend"
    if tr.get("stack_model"):
        if tr["stack_model"] != te.get("stack_model"):
            raise ValueError("train/test stack model differs")
        recipe["stack_model"] = tr["stack_model"]
    config = tr.get("config") or infer._json(path(tr["runs"][0]["path"]) / "manifest.json")["config"]
    if hh.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() != tr["config_sha256"]:
        raise ValueError("base scoring configuration changed")
    recipe.update({"config": config, "model_sha256": config["model"]["sha256"],
                   "train_coverage": {"complete": True, "targets": tr["targets"]}})
    target = histograms(test, recipe)
    training = histograms(train, recipe, fit=True)
    gate_train = histograms(train, recipe, fit=True, force_gate=True)
    result = curves(training, target, tr["fit_reference_counts"], te["reference_counts"], np.asarray(recipe["edges"]))
    unknown = {k: v for k, v in target.items() if k.rsplit("|", 1)[0] not in recipe["countries"]}
    if unseen and unknown:
        result.update(curves(gate_train, unknown, tr["fit_reference_counts"], te["reference_counts"], np.asarray(recipe["edges"])))
    recipe["curves"] = result
    validate_recipe(recipe)
    infer._write(out, recipe)
    return recipe


def validate_recipe(recipe):
    if recipe.get("version") != 1 or recipe.get("kind") != "segmented-postprocessor":
        raise ValueError("invalid postprocessor version or kind")
    if not isinstance(recipe.get("unseen_gate"), bool):
        raise ValueError("invalid unseen-country policy")
    for key in ("floor", "neural_weight"):
        value = recipe.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid postprocessor probability parameter")
    if isinstance(recipe.get("exact_limit"), bool) or not isinstance(recipe.get("exact_limit"), int) or not 1 <= recipe["exact_limit"] <= 256:
        raise ValueError("invalid exact decoding limit")
    if recipe.get("known_score", "blend") not in ("blend", "stack_prob"):
        raise ValueError("unknown score policy")
    if not isinstance(recipe.get("countries"), list) or not all(isinstance(c, str) for c in recipe["countries"]):
        raise ValueError("invalid labelled countries")
    if not isinstance(recipe.get("curves"), dict) or not recipe["curves"]:
        raise ValueError("missing posterior curves")
    part = recipe.get("partition", {"modulus": 2, "remainder": 1})
    if part not in ({"modulus": 2, "remainder": 1}, {"modulus": 3, "remainder": 1}):
        raise ValueError("invalid calibration partition")
    if recipe.get("known_score") == "stack_prob" and part != {"modulus": 3, "remainder": 1}:
        raise ValueError("stack calibration must exclude fitting and validation partitions")
    for curve in recipe["curves"].values():
        if not isinstance(curve, dict):
            raise ValueError("invalid posterior curve")
        x, y = np.asarray(curve.get("centres", []), dtype=float), np.asarray(curve.get("posterior", []), dtype=float)
        if (x.ndim != 1 or y.shape != x.shape or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all() or
                (np.diff(x) <= 0).any() or (np.diff(y) < 0).any() or ((y < 0) | (y > 1)).any()):
            raise ValueError("invalid monotone posterior curve")


def adjust(p, country, segment, recipe):
    p = np.asarray(p, dtype=float)
    out = p.copy()
    for co in np.unique(country):
        for seg in range(3):
            ix = (country == co) & (segment == seg)
            if not ix.any():
                continue
            curve = recipe["curves"].get(f"{co}|{seg}")
            if curve is None:
                raise ValueError("missing country/house calibration cell")
            out[ix] = np.interp(logit(p[ix]), curve["centres"], curve["posterior"])
    out = np.where(p < .02, np.minimum(p, out), out)
    return np.clip(out, 0, 1).astype(np.float32)


def export(prepared, recipe_path, out, threads=12):
    prepared, recipe_path, out = path(prepared), path(recipe_path), path(out)
    meta, recipe = verified(prepared, "test"), infer._json(recipe_path)
    validate_recipe(recipe)
    if (recipe.get("version") != 1 or recipe.get("kind") != "segmented-postprocessor" or
            recipe["config_sha256"] != meta["config_sha256"] or recipe["sources"]["test"] != meta["score_sha256"] or
            recipe["data_meta_sha256"] != meta["data_meta_sha256"]):
        raise ValueError("calibration does not match test scores")
    if out.exists():
        raise ValueError("export directory already exists")
    data = path(meta["data"])
    pieces = []
    for batch in pq.ParquetFile(prepared).iter_batches(batch_size=262144):
        d = batch.to_pydict()
        co, seg = np.asarray(d["co"]), np.asarray(d["seg"])
        raw = score_rows(d, recipe)
        pieces.append(pl.DataFrame({"qid": np.asarray(d["qid"], np.uint32), "tid": np.asarray(d["tid"], np.uint32),
                                   "prob": adjust(raw, co, seg, recipe), "raw": raw.astype(np.float32)}))
    frame = pl.concat(pieces)
    del pieces
    keep, stats = decode.choose(frame["qid"].to_numpy(), frame["tid"].to_numpy(), frame["prob"].to_numpy(),
                                frame["raw"].to_numpy(), recipe["floor"], recipe["exact_limit"], threads)
    matches = frame.filter(pl.Series(keep)).select("qid", "tid")
    del frame
    refs = pl.scan_parquet(data / "test/ref.parquet").select(pl.col("rid").alias("qid"), pl.col("eid").alias("source1_entity_id"))
    targets = pl.concat([pl.scan_parquet(data / "test" / f"s{i}.parquet").select(
        pl.col("rid").alias("tid"), pl.col("eid").alias("target_entity_id")) for i in (2, 3)])
    def names(d):
        return d.join(refs, on="qid", how="left").join(targets, on="tid", how="left").select(
            "source1_entity_id", "target_entity_id").sort("source1_entity_id", "target_entity_id")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tf.TemporaryDirectory(dir=out.parent) as tmp:
        tmp = path(tmp)
        infer._sink(refs.select("source1_entity_id").sort("source1_entity_id"), tmp / "refs.parquet")
        infer._sink(names(pl.scan_parquet(prepared).select("qid", "tid")), tmp / "candidates.parquet")
        infer._sink(names(matches.lazy()), tmp / "matches.parquet")
        result = infer._tsv(tmp / "refs.parquet", tmp / "candidates.parquet", tmp / "matches.parquet", out)
    result.update({"postprocessor_sha256": infer._sha(recipe_path), "scores_sha256": meta["score_sha256"],
                   "decoder": stats, "public_score": None})
    infer._write(out / "export.json", result)
    return result


def check():
    edges = np.linspace(-4, 4, 5)
    counts = np.array([100., 100., 100., 100., 100., 100.])
    pos = np.array([0., 10., 30., 60., 90., 100.])
    train = {f"us|{s}": [counts, pos] for s in range(3)}
    target = {f"us|{s}": [pos + 2 * (counts - pos), None] for s in range(3)}
    fitted = curves(train, target, {"us": 1000}, {"us": 1000}, edges, minimum=0)
    for value in fitted.values():
        np.testing.assert_allclose(value["posterior"], pos / (pos + 2 * (counts - pos)))
        assert np.all(np.diff(value["posterior"]) >= 0)
    recipe = {"neural_weight": .6, "unseen_gate": True, "countries": ["us"]}
    p = score(np.array([.1, .1]), np.array([.99, .99]), np.array(["us", "france"]), recipe)
    assert p[0] > .5 and p[1] == .1
    recipe["curves"] = fitted
    p = adjust(np.array([0., .5, 1.]), np.array(["us"] * 3), np.array([0, 1, 2]), recipe)
    assert np.isfinite(p).all() and p[0] == 0 and np.all(np.diff(p) >= 0)
    with tf.TemporaryDirectory() as tmp:
        root = path(tmp)
        data = infer._check_data(root)
        cfg = {"data_meta_sha256": infer._sha(data / "meta.json"), "model": {"sha256": "check"}}
        cfg_hash = hh.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        for split in ("train", "test"):
            run = root / split
            (run / "parts").mkdir(parents=True)
            rows = [(0, 10, .95, .9, .98, 2), (1, 10, .01, .01, .01, 2), (2, 11, .99, .3, .999, 2),
                    (0, 20, .98, .95, .99, 3), (0, 21, .3, .2, .5, 3)]
            frame = pl.DataFrame(rows, schema=["qid", "tid", "prob", "gate_prob", "neural_prob", "sr"], orient="row")
            frame = frame.with_columns(pl.col("qid", "tid").cast(pl.UInt32), pl.col("sr").cast(pl.UInt8))
            if split == "train":
                frame = frame.with_columns(pl.Series("y", [1, 0, 1, 1, 0], dtype=pl.UInt8))
            pair, cov = run / "parts/p.parquet", run / "parts/p.npy"
            frame.write_parquet(pair)
            np.save(cov, infer._ids(data, split))
            infer._write(run / "manifest.json", {"version": infer.ver, "split": split, "config": cfg,
                         "config_sha256": cfg_hash, "parts": [{"name": "parts/p.parquet", "coverage": "parts/p.npy",
                         "queries": 4, "pairs": len(frame), "pair_sha256": infer._sha(pair), "coverage_sha256": infer._sha(cov)}]})
            prepare(data, [run], split, root / f"{split}.parquet")
            prepare(data, [run], split, root / f"{split}-repeat.parquet")
            assert infer._sha(root / f"{split}.parquet") == infer._sha(root / f"{split}-repeat.parquet")
        recipe_path = root / "recipe.json"
        model = fit(root / "train.parquet", root / "test.parquet", recipe_path)
        assert model["countries"] == ["us"] and model["curves"]["france|0"]["pooled"]
        result = export(root / "test.parquet", recipe_path, root / "out", threads=2)
        assert result["source1"] == 4 and result["candidate_pairs"] == 5
        manifest = infer._json(root / "test/manifest.json")
        infer._write(root / "test/manifest.json", {**manifest, "changed": True})
        try:
            verified(root / "test.parquet", "test", [root / "test"])
        except ValueError as error:
            assert "manifests changed" in str(error)
        else:
            raise AssertionError("stale prepared scores accepted")
        infer._write(root / "test/manifest.json", manifest)
        bad = {**model, "sources": {"test": "wrong"}}
        infer._write(root / "bad.json", bad)
        try:
            export(root / "test.parquet", root / "bad.json", root / "bad-out")
        except ValueError as error:
            assert "does not match" in str(error)
        else:
            raise AssertionError("mismatched calibration accepted")
    decode.check()
    print("postprocessor checks passed")


def main():
    p = ap.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    z = sub.add_parser("prepare")
    z.add_argument("--data", type=path, default=path("cache/data"))
    z.add_argument("--split", choices=("train", "test"), required=True)
    z.add_argument("--runs", type=path, nargs="+", required=True)
    z.add_argument("--out", type=path, required=True)
    z = sub.add_parser("fit")
    z.add_argument("--train", type=path, required=True)
    z.add_argument("--test", type=path, required=True)
    z.add_argument("--out", type=path, required=True)
    z.add_argument("--weight", type=float, default=.6)
    z.add_argument("--blend-unseen", action="store_true")
    z = sub.add_parser("export")
    z.add_argument("--scores", type=path, required=True)
    z.add_argument("--recipe", type=path, required=True)
    z.add_argument("--out", type=path, required=True)
    z.add_argument("--threads", type=int, default=12)
    sub.add_parser("check")
    a = p.parse_args()
    if a.command == "check":
        check()
    elif a.command == "prepare":
        print(json.dumps(prepare(a.data, a.runs, a.split, a.out), indent=2))
    elif a.command == "fit":
        print(json.dumps(fit(a.train, a.test, a.out, a.weight, not a.blend_unseen), indent=2))
    else:
        print(json.dumps(export(a.scores, a.recipe, a.out, a.threads), indent=2))


if __name__ == "__main__":
    main()
