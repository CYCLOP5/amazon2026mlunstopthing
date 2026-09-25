import argparse as ap
import hashlib as hh
import json
import os
import shutil as sh
import tempfile as tf
import time
from collections import defaultdict as dd
from pathlib import Path as path

import joblib as jl
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sparse_dot_topn import sp_matmul_topn as topn


schema = {"tid": pl.UInt32, "qid": pl.UInt32, "ns": pl.Float32,
          "ads": pl.Float32, "en": pl.UInt8, "ea": pl.UInt8}
sv = 2


def indexes(ref, fit, dest):
    ref = ref.sort("rid")
    dest.mkdir(parents=True, exist_ok=True)
    idx = {}
    for fld in ("nn", "an"):
        p = dest / f"{fld}.joblib"
        if not p.exists():
            t = time.monotonic()
            v = tfidf(analyzer="char_wb", ngram_range=(3, 4), lowercase=False, dtype=np.float32,
                      min_df=2 if len(fit) >= 200 else 1, max_df=0.03 if len(fit) >= 200 else 1.0,
                      max_features=600000, sublinear_tf=True)
            v.fit(fit[fld].to_list())
            b = v.transform(ref[fld].to_list()).T.tocsr()
            tmp = p.with_suffix(f".{os.getpid()}.tmp")
            jl.dump((v, b, ref["rid"].to_numpy()), tmp, compress=0)
            tmp.replace(p)
            print("index", fld, b.shape, b.nnz, "seconds", round(time.monotonic() - t, 1), flush=True)
            del v, b
        obj = jl.load(p, mmap_mode="c")
        dp = dest / f"{fld}_rows.joblib"
        if not dp.exists():
            tmp = dp.with_suffix(f".{os.getpid()}.tmp")
            jl.dump(obj[1].T.tocsr(), tmp, compress=0)
            tmp.replace(dp)
        idx[fld] = (*obj, jl.load(dp, mmap_mode="c"))
        if not np.array_equal(idx[fld][2], ref["rid"].to_numpy()):
            raise ValueError("stale reference index")
    eq = {fld: dd(list) for fld in ("nn", "an")}
    for fld in eq:
        for rid, s in zip(ref["rid"].to_list(), ref[fld].to_list()):
            if s:
                eq[fld][s].append(rid)
    return idx, eq


def rescore(q, p, idx):
    if p.is_empty():
        return p
    ids = q["rid"].to_numpy()
    order = np.argsort(ids)
    qi = np.searchsorted(ids[order], p["tid"].to_numpy())
    if (qi >= len(ids)).any() or not np.array_equal(ids[order][qi], p["tid"].to_numpy()):
        raise ValueError("unknown query id during scoring")
    qi = order[qi]
    for fld, col in (("nn", "ns"), ("an", "ads")):
        v, _, ids, d = idx[fld]
        ri = np.searchsorted(ids, p["qid"].to_numpy())
        if (ri >= len(ids)).any() or not np.array_equal(ids[ri], p["qid"].to_numpy()):
            raise ValueError("unknown reference id during scoring")
        a = v.transform(q[fld].to_list())
        vals = np.asarray(a[qi].multiply(d[ri]).sum(axis=1)).ravel()
        p = p.with_columns(pl.Series(col, np.clip(vals, 0, 1).astype(np.float32)))
    return p


def search(q, idx, eq, k=10, threads=4):
    if k < 1 or threads < 1:
        raise ValueError("positive k and thread count required")
    fs = []
    tids = q["rid"].to_numpy()
    for fld, sc in (("nn", "ns"), ("an", "ads")):
        v, b, ids, _ = idx[fld]
        a = v.transform(q[fld].to_list())
        c = topn(a, b, top_n=min(k, b.shape[1]), threshold=0.01, sort=True, n_threads=threads)
        rr = np.repeat(np.arange(len(q)), np.diff(c.indptr))
        z = np.zeros(c.nnz)
        fs.append(pl.DataFrame({"tid": tids[rr], "qid": ids[c.indices],
                                "ns": c.data if sc == "ns" else z,
                                "ads": c.data if sc == "ads" else z,
                                "en": z, "ea": z}, schema=schema))
    ts, rs, ns, ads = [], [], [], []
    for fld in ("nn", "an"):
        for tid, s in zip(tids, q[fld].to_list()):
            for rid in eq[fld].get(s, ()):
                ts.append(int(tid))
                rs.append(rid)
                ns.append(int(fld == "nn"))
                ads.append(int(fld == "an"))
    if ts:
        fs.append(pl.DataFrame({"tid": ts, "qid": rs, "ns": [0.] * len(ts), "ads": [0.] * len(ts),
                                "en": ns, "ea": ads}, schema=schema))
    p = pl.concat(fs).group_by("tid", "qid").agg(pl.col("ns", "ads", "en", "ea").max())
    p = p.join(q.select(pl.col("rid").alias("tid"), "own", "sr"), on="tid", how="left", validate="m:1")
    p = p.with_columns((pl.col("qid").cast(pl.Int64) == pl.col("own")).cast(pl.UInt8).alias("y")).sort("tid", "qid")
    return rescore(q, p, idx)


def global_scope(ref, co):
    return not co or ref.filter(pl.col("co") == co).is_empty()


def setup(data, cache, co, sp="train", fold=0):
    allref = pl.read_parquet(data / sp / "ref.parquet")
    fallback = global_scope(allref, co)
    ref = allref if fallback else allref.filter(pl.col("co") == co)
    tr = pl.read_parquet(data / "train/ref.parquet")
    if sp == "train":
        if fallback:
            fit = tr.filter(pl.col("fold") == 2)
        else:
            fit = tr.filter((pl.col("co") == co) & (pl.col("fold") == 2))
            if fit.is_empty():
                fit = tr.filter(pl.col("fold") == 2)
        if fold == 2:
            ref = ref.filter(pl.col("fold") == 2)
    else:
        if fallback:
            fit = tr
        else:
            fit = tr.filter(pl.col("co") == co)
            if fit.is_empty():
                fit = tr
    if ref.is_empty() or fit.is_empty():
        raise ValueError(f"no references or fit rows for {co}")
    cfg = {"data": json.loads((data / "meta.json").read_text()), "country": None if fallback else co,
           "scope": "global" if fallback else "country", "split": sp,
           "pool": "fit" if sp == "train" and fold == 2 else "all", "v": 1}
    key = hh.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:20]
    idx, eq = indexes(ref, fit, cache / key)
    return ref, idx, eq


def sample(data, co, fold, n, neg, seed=42):
    r = pl.read_parquet(data / "train/ref.parquet").filter((pl.col("co") == co) & (pl.col("fold") == fold))
    r = r.sample(n=min(n, len(r)), seed=seed).sort("rid")
    if r.is_empty():
        raise ValueError("empty anchor sample")
    ids = r["rid"].cast(pl.Int32).to_list()
    qs = []
    for sr in (2, 3):
        d = pl.scan_parquet(data / "train" / f"s{sr}.parquet").filter(pl.col("co") == co).collect(engine="streaming")
        pos = d.filter(pl.col("own").is_in(ids))
        no = d.filter(pl.col("own") < 0).with_columns(
            (pl.concat_str("co", "nn", "an", separator="\t").hash(seed=seed) % 10).alias("h"))
        no = no.filter(pl.col("h") >= 2) if fold == 2 else no.filter(pl.col("h") == fold)
        no = no.sample(n=min((neg + 1) // 2, len(no)), seed=seed + sr).drop("h")
        qs.extend([pos, no])
        del d
    q = pl.concat(qs).sort("rid")
    if q.filter(pl.col("own") >= 0).height != int(r["deg"].sum()):
        raise ValueError("sample is missing labeled aliases")
    return r, q


def probe(data, cache, dest, co, fold=0, n=1000, neg=1000, k=10, threads=4, batch=1024):
    t = time.monotonic()
    r, q = sample(data, co, fold, n, neg)
    ref, idx, eq = setup(data, cache, co, fold=fold)
    dest.mkdir(parents=True, exist_ok=True)
    r.write_parquet(dest / "anchors.parquet")
    q.write_parquet(dest / "queries.parquet")
    fs = []
    parts = []
    for j, qb in enumerate(q.iter_slices(batch)):
        p = search(qb, idx, eq, k, threads)
        part = f"pairs_{j:05d}.parquet"
        p.write_parquet(dest / part, compression="zstd")
        parts.append(part)
        fs.append(p.select("tid", "qid", "y", "sr"))
        if j % 10 == 0:
            print("block", co, fold, min((j + 1) * batch, len(q)), "of", len(q), flush=True)
    p = pl.concat(fs)
    hits = p.filter(pl.col("y") == 1)
    tp = dict(hits.group_by("qid").len().iter_rows())
    vals = [1.0 if d == 0 else 1.25 * tp.get(i, 0) / (tp.get(i, 0) + 0.25 * d)
            for i, d in r.select("rid", "deg").iter_rows()]
    cn = p.group_by("tid").len()["len"].to_numpy()
    out = {"country": co, "fold": fold, "anchors": len(r), "queries": len(q), "reference_pool": len(ref),
           "positive_queries": int(r["deg"].sum()), "pairs": len(p), "retrieved_links": len(hits),
           "link_recall": len(hits) / int(r["deg"].sum()) if r["deg"].sum() else 1.0,
           "oracle_macro_f05": float(np.mean(vals)), "k": k,
           "empty_queries": len(q) - len(cn), "mean_candidates": len(p) / len(q),
           "p99_candidates": float(np.quantile(cn, 0.99)) if len(cn) else 0,
           "seconds": time.monotonic() - t, "score_version": sv, "parts": parts, "slices": {}}
    for sr in (2, 3):
        z = q.filter((pl.col("sr") == sr) & (pl.col("own") >= 0))
        hs = set(hits.filter(pl.col("sr") == sr)["tid"].to_list())
        for fld, mask in (("all", pl.lit(True)), ("non_ascii_name", pl.col("nm").str.contains(r"[^\x00-\x7f]")),
                          ("blank_addr", pl.col("ad").str.strip_chars() == "")):
            v = z.filter(mask)
            out["slices"][f"s{sr}_{fld}"] = {"n": len(v), "hits": sum(i in hs for i in v["rid"])}
    (dest / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    return out


def check():
    with tf.TemporaryDirectory() as tmp:
        r = pl.DataFrame({"rid": [3, 9, 12], "nn": ["ram market", "acme inc", "acme inc"],
                          "an": ["10 lake road delhi", "20 main street", "80 distant road"]},
                         schema_overrides={"rid": pl.UInt32})
        q = pl.DataFrame({"rid": [4, 20], "nn": ["ram market", "acme inc"],
                          "an": ["10 lake rd delhi", ""], "own": [3, 9], "sr": [2, 3]},
                         schema_overrides={"rid": pl.UInt32, "own": pl.Int32, "sr": pl.UInt8})
        idx, eq = indexes(r, r, path(tmp))
        p = search(q, idx, eq, k=1, threads=1)
        assert p.filter(pl.col("y") == 1).height == 2
        assert {9, 12} <= set(p.filter(pl.col("tid") == 20)["qid"])
        assert np.allclose(p.filter((pl.col("tid") == 20) & pl.col("qid").is_in([9, 12]))["ns"], 1)
        assert p.select("tid", "qid").n_unique() == len(p)
        idx2, eq2 = indexes(r, r, path(tmp))
        assert search(q, idx2, eq2, 1, 1).equals(p)
    print("checks passed")


def revise(data, cache, src, dest):
    if src.resolve() == dest.resolve():
        raise ValueError("rescored runs need a new output directory")
    met = json.loads((src / "metrics.json").read_text())
    ref, idx, eq = setup(data, cache, met["country"], fold=met["fold"])
    del ref, eq
    q = pl.read_parquet(src / "queries.parquet")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").unlink(missing_ok=True)
    for name in ("anchors.parquet", "queries.parquet"):
        sh.copyfile(src / name, dest / name)
    for name in met["parts"]:
        if path(name).name != name or not name.endswith(".parquet"):
            raise ValueError("invalid candidate part path")
        p = pl.read_parquet(src / name)
        qb = q.filter(pl.col("rid").is_in(p["tid"].unique().implode()))
        p = rescore(qb, p, idx)
        p.write_parquet(dest / name, compression="zstd")
    met.update({"score_version": sv, "derived_from": str(src)})
    (dest / "metrics.json").write_text(json.dumps(met, indent=2) + "\n")
    print("rescored", src, "to", dest, flush=True)


def main():
    root = path(__file__).resolve().parents[1]
    pa = ap.ArgumentParser()
    pa.add_argument("--data", type=path, default=root / "cache/data")
    pa.add_argument("--cache", type=path, default=root / "cache/block")
    pa.add_argument("--out", type=path)
    pa.add_argument("--country", default="us")
    pa.add_argument("--fold", type=int, choices=[0, 1, 2], default=0)
    pa.add_argument("--n", type=int, default=1000)
    pa.add_argument("--neg", type=int, default=1000)
    pa.add_argument("--k", type=int, default=10)
    pa.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    pa.add_argument("--batch", type=int, default=1024)
    pa.add_argument("--check", action="store_true")
    pa.add_argument("--rescore", type=path)
    a = pa.parse_args()
    if a.check:
        check()
    elif a.rescore:
        if a.out is None:
            raise ValueError("rescore requires an output directory")
        revise(a.data, a.cache, a.rescore, a.out)
    else:
        if a.n < 1 or a.neg < 0 or a.batch < 1:
            raise ValueError("invalid sample or batch size")
        dest = a.out or root / "cache/runs" / f"block_{a.country}_{a.fold}"
        probe(a.data, a.cache, dest, a.country, a.fold, a.n, a.neg, a.k, a.threads, a.batch)


if __name__ == "__main__":
    main()
