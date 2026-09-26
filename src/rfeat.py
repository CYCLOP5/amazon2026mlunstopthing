"""dense-aware gate features with full-corpus raw-text ambiguity statistics"""
import argparse as ap
import hashlib as hh
import json
import math
from pathlib import Path as path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

import feat
import infer
import norm2
import tfeat


backend = "hybrid-v2"
extra = ["ref_name_dup", "target_name_dup", "ref_name_in_targets", "name_idf_cos", "name_idf_ref_cover",
         "name_idf_target_cover", "name_idf_max_shared", "name_idf_max_missing", "addr_idf_cos",
         "addr_idf_ref_cover", "addr_idf_target_cover", "house_min_gap", "house_digit_diffs", "house_first_diff",
         "house_last_diff", "house_same_length", "target_house_in_ref"]
dense_allowed = ("ds_e5", "ds_qwen3", "ds_e5_large", "ds_e5_small", "ds_bge_m3")


def names(dense):
    if not isinstance(dense, (list, tuple)) or len(set(dense)) != len(dense) or set(dense) - set(dense_allowed):
        raise ValueError("invalid rich dense feature contract")
    return list(tfeat.ff) + extra + [name for d in dense for name in (d, d + "_rank", d + "_gap")]


def contract(meta):
    ds = meta.get("dense_features", [])
    if meta.get("feature_backend") != backend or meta.get("feature_names") != names(ds):
        raise ValueError("rich feature contract mismatch")
    return names(ds), list(ds)


def corpus(data, split, normalizer, cache):
    data, normalizer, cache = path(data), path(normalizer), path(cache)
    source = {"version": 1, "data": infer._sha(data / "meta.json"), "normalizer": infer._sha(normalizer), "split": split,
              "code": {n: infer._sha(path(__file__).parent / n) for n in ("norm2.py", "tm_prep.py", "tm_rules.py")}}
    key = hh.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    out = cache / "rich" / key
    if (out / "metadata.json").exists():
        meta = infer._json(out / "metadata.json")
        if meta["source"] != source or any(infer._sha(out / n) != sha for n, sha in meta["files"].items()):
            raise ValueError("rich corpus cache changed")
        return out
    mapping = norm2.load(normalizer, data)
    records = out / "records"
    records.mkdir(parents=True, exist_ok=True)
    paths = []
    for sr, name in ((1, "ref"), (2, "s2"), (3, "s3")):
        for i, batch in enumerate(pq.ParquetFile(data / split / f"{name}.parquet").iter_batches(
                batch_size=100000, columns=["rid", "nm", "ad", "co"])):
            p = records / f"s{sr}-{i:04d}.parquet"
            frame = norm2.apply(pl.from_arrow(batch), mapping)
            d = tfeat.prep(frame).select(pl.col("entity_id").alias("rid"), pl.col("country").alias("co"),
                                         "name_core", "addr_can", pl.lit(sr, pl.UInt8).alias("sr"))
            infer._pq(d, p)
            paths.append(p)
    rows = pl.scan_parquet(paths)
    counts = rows.group_by("co", "name_core").agg((pl.col("sr") == 1).sum().alias("nr"), (pl.col("sr") != 1).sum().alias("nt"))
    infer._sink(counts, out / "names.parquet")
    counted = rows.join(pl.scan_parquet(out / "names.parquet"), on=["co", "name_core"], how="left").select("rid", "sr", "nr", "nt")
    for side, condition, cols in (("ref", pl.col("sr") == 1, ["nr", "nt"]), ("target", pl.col("sr") != 1, ["nt"])):
        frame = counted.filter(condition).sort("rid").collect(engine="streaming")
        if not np.array_equal(frame["rid"].to_numpy(), np.arange(len(frame), dtype=np.uint32)):
            raise ValueError("rich corpus ids are not contiguous")
        infer._npy(out / f"{side}.npy", frame.select(cols).to_numpy().astype(np.uint32))
    totals = rows.group_by("co").len().rename({"len": "total"})
    words = []
    for field, col in (("n", "name_core"), ("a", "addr_can")):
        tokens = rows.select("co", pl.col(col).str.split(" ").list.unique().alias("word")).explode("word", empty_as_null=True).filter(pl.col("word") != "")
        words.append(tokens.group_by("co", "word").len().join(totals, on="co").select(
            "co", "word", pl.lit(field).alias("field"), ((pl.col("total") + 1) / (pl.col("len") + 1)).log().add(1).cast(pl.Float32).alias("idf")))
    infer._sink(pl.concat(words), out / "words.parquet")
    files = {n: infer._sha(out / n) for n in ("ref.npy", "target.npy", "words.parquet")}
    infer._write(out / "metadata.json", {"source": source, "files": files})
    return out


def prep(ref, data, split, normalizer, cache):
    mapping = norm2.load(normalizer, data)
    out = corpus(data, split, normalizer, cache)
    words = {}
    for co, word, field, idf in pl.read_parquet(out / "words.parquet").iter_rows():
        words.setdefault((co, field), {})[word] = idf
    return {"base": tfeat.prep(norm2.apply(ref, mapping)), "mapping": mapping, "words": words,
            "ref_counts": np.load(out / "ref.npy", mmap_mode="r"), "target_counts": np.load(out / "target.npy", mmap_mode="r")}


def overlap(a, b, weights):
    a, b = set(a.split()), set(b.split())
    shared = a & b
    value = lambda token: weights.get(token, 1.)
    sa, sb = sum(value(t) for t in a), sum(value(t) for t in b)
    aa, bb = sum(value(t) ** 2 for t in a), sum(value(t) ** 2 for t in b)
    both = sum(value(t) for t in shared)
    return [sum(value(t) ** 2 for t in shared) / math.sqrt(aa * bb) if aa and bb else 0.,
            both / sa if sa else 0., both / sb if sb else 0., max((value(t) for t in shared), default=0.),
            max((value(t) for t in a - b), default=0.)]


def house(a, b):
    a, b = a.split(), b.split()
    if not a or not b:
        return [-1., -1., -1., -1., -1., 0.]
    if len(a[0]) > 12:
        return [-1., -1., -1., -1., -1., float(b[0] in a)]
    possible = [v for v in b if len(v) <= 12]
    if not possible:
        return [-1., -1., -1., -1., -1., float(b[0] in a)]
    best = min(possible, key=lambda t: (abs(int(a[0]) - int(t)), t))
    x = a[0]
    same = len(x) == len(best)
    diffs = [i for i in range(len(x)) if x[i] != best[i]] if same else []
    return [math.log1p(abs(int(x) - int(best))), float(len(diffs)) if same else -1.,
            diffs[0] / len(x) if diffs else 99. if same else -1.,
            float(len(x) - 1 - diffs[-1]) if diffs else 99. if same else -1., float(same), float(b[0] in a)]


def make(st, queries, pairs, threads=1, dense=()):
    fs = names(dense)
    if not len(pairs):
        return np.empty((0, len(fs)), np.float32), fs
    q = norm2.apply(queries.filter(pl.col("rid").is_in(pairs["tid"].unique().implode())), st["mapping"])
    pool = st["base"].filter(pl.col("entity_id").is_in(pairs["qid"].unique().implode()))
    base, _ = tfeat.make(pool, q, pairs, threads)
    qt = tfeat.prep(q)
    joined = pairs.select("qid", "tid").join(pool.select(
        pl.col("entity_id").alias("qid"), pl.col("country").alias("co"), pl.col("name_core").alias("rn"),
        pl.col("addr_can").alias("ra"), pl.col("addr_nums").alias("ru")), on="qid", how="left", maintain_order="left", validate="m:1")
    joined = joined.join(qt.select(pl.col("entity_id").alias("tid"), pl.col("name_core").alias("tn"),
                                  pl.col("addr_can").alias("ta"), pl.col("addr_nums").alias("tu")),
                         on="tid", how="left", maintain_order="left", validate="m:1")
    if joined.null_count().sum_horizontal().sum():
        raise ValueError("rich feature join lost a record")
    qid, tid = pairs["qid"].to_numpy(), pairs["tid"].to_numpy()
    if qid.max() >= len(st["ref_counts"]) or tid.max() >= len(st["target_counts"]):
        raise ValueError("rich corpus scope does not contain pair ids")
    x = np.empty((len(pairs), len(extra)), dtype=np.float32)
    x[:, :3] = np.column_stack((np.log1p(st["ref_counts"][qid, 0]), np.log1p(st["target_counts"][tid, 0]),
                                np.log1p(st["ref_counts"][qid, 1])))
    for i, (co, rn, ra, ru, tn, ta, tu) in enumerate(joined.select("co", "rn", "ra", "ru", "tn", "ta", "tu").iter_rows()):
        x[i, 3:8] = overlap(rn, tn, st["words"].get((co, "n"), {}))
        x[i, 8:11] = overlap(ra, ta, st["words"].get((co, "a"), {}))[:3]
        x[i, 11:] = house(ru, tu)
    extra_dense = []
    for col in dense:
        score = feat._num(pairs, col)
        if ((score < -1) | (score > 1)).any():
            raise ValueError("invalid dense cosine")
        rank, gap, _ = feat._rk(tid, score)
        extra_dense.extend((score, rank, gap))
    result = np.column_stack([base, x, *extra_dense]).astype(np.float32, copy=False)
    if np.isinf(result).any():
        raise ValueError("infinite rich feature")
    return np.nan_to_num(result, nan=-1.), fs


def check():
    import tempfile as tf
    import train
    weights = {"common": 1., "rare": 5., "other": 5.}
    a = overlap("common rare", "common", weights)
    b = overlap("common rare", "rare", weights)
    assert b[0] > a[0] and a[4] == 5
    assert house("1958", "834 1958")[0] == 0
    assert house("2212", "2213")[1:5] == [1., .75, 0., 1.]
    ref = pl.DataFrame({"rid": [0, 1], "nm": ["ciel ecole", "ciel ecole"], "ad": ["27 rue duc", "29 rue duc"], "co": ["france"] * 2})
    query = pl.DataFrame({"rid": [0], "nm": ["ciel ecole & fils"], "ad": ["27 rue duc"], "co": ["france"]})
    pairs = pl.DataFrame({"qid": [0, 1], "tid": [0, 0], "ns": [.5, .5], "ads": [.9, .9], "ds_e5": [.9, .1]})
    state = {"base": tfeat.prep(ref), "mapping": {}, "words": {}, "ref_counts": np.array([[2, 4], [2, 4]]),
             "target_counts": np.array([[1]])}
    x, fs = make(state, query, pairs, dense=["ds_e5"])
    assert x.shape == (2, len(fs)) and np.isfinite(x).all()
    assert x[0, fs.index("ds_e5_rank")] == 1 and x[1, fs.index("ds_e5_rank")] == 2
    assert x[0, fs.index("house_min_gap")] == 0 < x[1, fs.index("house_min_gap")]
    with tf.TemporaryDirectory() as tmp:
        root = path(tmp)
        data = infer._check_data(root)
        for split in ("train", "test"):
            for sr in (2, 3):
                p = data / split / f"s{sr}.parquet"
                d = pl.read_parquet(p).with_columns(pl.Series("rid", np.arange((sr - 2) * 2, (sr - 1) * 2, dtype=np.uint32)))
                d.write_parquet(p)
        normalizer = root / "normalizer.json"
        norm2.fit(data, normalizer)
        directory = corpus(data, "train", normalizer, root / "cache")
        assert np.load(directory / "target.npy").shape == (4, 1)
        assert corpus(data, "train", normalizer, root / "cache") == directory
        xx = np.tile(x, (80, 1))
        yy = np.tile(np.array([1, 0]), 80)
        fitted = train._fit_lgb(xx, yy, xx, yy, 5, 1, fs)
        assert fitted.get_params()["subsample_freq"] == 1
        fitted.booster_.save_model(root / "lgb.txt")
        metadata = {"feature_backend": backend, "feature_names": fs, "dense_features": ["ds_e5"],
                    "models": ["lgb"], "model_files": {"lgb": "lgb.txt"}}
        infer._write(root / "metadata.json", metadata)
        loaded, _ = train.load_models(root)
        np.testing.assert_allclose(train.predict(loaded["lgb"], x), fitted.predict_proba(x)[:, 1])
    print("rich feature checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=path, default=path("cache/data"))
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--normalizer", type=path, default=path("artifacts/norm2.json"))
    p.add_argument("--cache", type=path, default=path("cache"))
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        check()
    else:
        print(corpus(a.data, a.split, a.normalizer, a.cache))
