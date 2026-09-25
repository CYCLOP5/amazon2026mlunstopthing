import argparse as ap
import gc
import hashlib as hh
import json
import tempfile as tf
import time
from pathlib import Path as path

import numpy as np
import polars as pl


mod0 = "intfloat/multilingual-e5-base"
rev0 = "d128750597153bb5987e10b1c3493a34e5a4502a"
src0 = path("/home/cyclops/Documents/GitHub/Amazon ML/reports/model_sources.json")
fmt = "{role}: name: {nm}\naddress: {ad}\ncountry: {co}"


def sha(b):
    return hh.sha256(b).hexdigest()



def atom(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    q.replace(p)


def serial(d, role):
    if role not in ("query", "passage"):
        raise ValueError("invalid e5 role")
    return [fmt.format(role=role, nm=str(n), ad=str(a), co=str(c))
            for n, a, c in d.select("nm", "ad", "co").iter_rows()]


def fprint(d, txt):
    h = hh.sha256()
    for rid, s in zip(d["rid"].to_list(), txt):
        h.update(str(rid).encode())
        h.update(b"\0")
        h.update(s.encode("utf-8"))
        h.update(b"\0")
    return {"rows": len(d), "sha256": h.hexdigest()}


def need(d, cols, what):
    miss = set(cols) - set(d.columns)
    if miss:
        raise ValueError(f"{what} missing {sorted(miss)}")


def load_run(data, run):
    a = pl.read_parquet(run / "anchors.parquet")
    q = pl.read_parquet(run / "queries.parquet")
    need(a, ("rid", "nm", "ad", "co", "deg", "fold"), "anchors")
    need(q, ("rid", "nm", "ad", "co", "own", "sr"), "queries")
    if a.is_empty() or q.is_empty() or a["rid"].n_unique() != len(a) or q["rid"].n_unique() != len(q):
        raise ValueError("empty or duplicate run ids")
    cos = a["co"].unique().to_list()
    fs = a["fold"].unique().to_list()
    if len(cos) != 1 or len(fs) != 1 or q.filter(pl.col("co") != cos[0]).height:
        raise ValueError("run must contain one country and fold")
    if int(fs[0]) not in (0, 1, 2):
        raise ValueError("invalid fold")
    ids = set(a["rid"].to_list())
    pos = q.filter(pl.col("own") >= 0)
    if set(pos["own"].to_list()) - ids:
        raise ValueError("query owns an unselected anchor")
    got = dict(pos.group_by("own").len().iter_rows())
    for rid, deg in a.select("rid", "deg").iter_rows():
        if got.get(rid, 0) != deg:
            raise ValueError("selected aliases do not match anchor degree")
    r = pl.read_parquet(data / "train/ref.parquet").filter(pl.col("co") == cos[0])
    if int(fs[0]) == 2:
        r = r.filter(pl.col("fold") == 2)
    need(r, ("rid", "nm", "ad", "co", "fold"), "references")
    if r.is_empty() or a.join(r.select("rid"), on="rid", how="anti").height:
        raise ValueError("reference pool excludes selected anchor")
    return a.sort("rid"), q.sort("rid"), r.sort("rid"), str(cos[0]), int(fs[0])


def source(p, model, rev):
    if not p.exists():
        raise FileNotFoundError(f"model source metadata not found: {p}")
    for x in json.loads(p.read_text(encoding="utf-8")):
        if x.get("model") == model and x.get("revision") == rev:
            if x.get("license", "").lower() != "mit":
                raise ValueError("model license is not mit")
            return x
    raise ValueError("model revision is absent from source metadata")


def model_load(model, rev, dev, maxlen):
    import torch
    from sentence_transformers import SentenceTransformer

    use = dev if dev == "cuda" and torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(model, revision=rev, trust_remote_code=False, device=use)
    m.max_seq_length = maxlen
    if use == "cuda":
        m.half()
    else:
        m.float()
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    return m, use, n


def trunc(m, rs, qs, maxlen):
    out = {}
    for k, x in (("passages", rs), ("queries", qs)):
        z = x[:min(512, len(x))]
        ids = m.tokenizer(z, add_special_tokens=True, truncation=False)["input_ids"]
        ns = np.array([len(v) for v in ids], dtype=np.int32)
        out[k] = {"sample": len(z), "truncated": int((ns > maxlen).sum()),
                  "max_tokens": int(ns.max()) if len(ns) else 0}
    return out


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero embedding")
    return x / n


def meta_core(data, model, rev, pars, maxlen, dim, co, fold, refs, qs, src):
    mp = data / "meta.json"
    return {"v": 1, "model": model, "revision": rev, "params": pars, "maxlen": maxlen,
            "dim": dim, "dtype": "float16", "prefixes": {"reference": "passage", "target": "query"},
            "format": fmt, "country": co, "fold": fold, "pool": "fold2" if fold == 2 else "all",
            "data_meta_sha256": sha(mp.read_bytes()) if mp.exists() else None,
            "inputs": {"references": fprint(refs[0], refs[1]), "queries": fprint(qs[0], qs[1])},
            "source": src}


def cache_meta(p, core, tr=None):
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old.get("core") != core:
            raise ValueError("cache metadata mismatch; use a new cache directory")
        return old
    out = {"core": core, "truncation": tr}
    atom(p, out)
    return out


def stage(base, kind, txt, dim, m, batch):
    p = base / f"{kind}.npy"
    pp = base / f"{kind}.progress.json"
    n = len(txt)
    if p.exists() != pp.exists():
        raise ValueError(f"orphaned {kind} cache artifact")
    if p.exists():
        a = np.load(p, mmap_mode="r+")
        if a.shape != (n, dim) or a.dtype != np.float16:
            raise ValueError(f"stale {kind} embedding shape")
    else:
        a = np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(n, dim))
        a.flush()
    if pp.exists():
        pr = json.loads(pp.read_text(encoding="utf-8"))
        done = int(pr.get("done", -1))
        if int(pr.get("rows", -1)) != n:
            raise ValueError(f"stale {kind} progress")
    else:
        done = 0
        atom(pp, {"done": done, "rows": n})
    if not 0 <= done <= n:
        raise ValueError(f"invalid {kind} progress")
    step = max(batch, min(4096, batch * 32))
    for lo in range(done, n, step):
        hi = min(lo + step, n)
        z = m.encode(txt[lo:hi], batch_size=batch, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=True, precision="float32")
        z = norm(z)
        if z.shape != (hi - lo, dim):
            raise ValueError("unexpected embedding shape")
        a[lo:hi] = z.astype(np.float16)
        a.flush()
        atom(pp, {"done": hi, "rows": n})
        print(kind, hi, "of", n, flush=True)
    return p


def topk_np(q, r, k):
    s = np.asarray(q, np.float32) @ np.asarray(r, np.float32).T
    ix = np.argsort(-s, axis=1, kind="stable")[:, :k]
    return ix, np.take_along_axis(s, ix, axis=1)


def gpu_search(qp, rp, k, batch):
    import torch

    free, _ = torch.cuda.mem_get_info()
    nr, dim = rp.shape
    room = free - nr * dim * 2 - 256 * 1024**2
    qb = min(batch, room // max(1, nr * 2))
    if qb < 1:
        return None
    r = torch.tensor(np.asarray(rp), device="cuda", dtype=torch.float16)
    out = []
    for lo in range(0, len(qp), qb):
        q = torch.tensor(np.asarray(qp[lo:lo + qb]), device="cuda", dtype=torch.float16)
        s, ix = torch.topk(q @ r.T, k, dim=1)
        out.append((ix.cpu().numpy(), s.float().cpu().numpy()))
    del r
    torch.cuda.empty_cache()
    return out, int(qb)


def cpu_search(qp, rp, k, batch):
    import faiss

    idx = faiss.IndexFlatIP(rp.shape[1])
    idx.add(np.asarray(rp, dtype=np.float32))
    return [(ix, s) for s, ix in (idx.search(np.asarray(qp[lo:lo + batch], dtype=np.float32), k)
                                  for lo in range(0, len(qp), batch))]


def write_candidates(out, q, refs, hits):
    import pyarrow.parquet as pq

    tmp = out.with_suffix(".tmp")
    wr = None
    off = 0
    try:
        for ix, ds in hits:
            n = len(ix)
            tids = np.repeat(q["rid"].to_numpy()[off:off + n], ix.shape[1])
            own = np.repeat(q["own"].to_numpy()[off:off + n], ix.shape[1])
            sr = np.repeat(q["sr"].to_numpy()[off:off + n], ix.shape[1])
            d = pl.DataFrame({"tid": tids, "qid": refs["rid"].to_numpy()[ix].reshape(-1),
                              "ds": ds.reshape(-1).astype(np.float32), "own": own, "sr": sr},
                             schema_overrides={"tid": pl.UInt32, "qid": pl.UInt32, "ds": pl.Float32,
                                               "own": pl.Int32, "sr": pl.UInt8})
            t = d.to_arrow()
            if wr is None:
                wr = pq.ParquetWriter(tmp, t.schema, compression="zstd")
            wr.write_table(t)
            off += n
    finally:
        if wr is not None:
            wr.close()
    if off != len(q):
        raise ValueError("retrieval output does not cover queries")
    tmp.replace(out)
def score(a, cand):
    hit = cand.filter((pl.col("own") >= 0) & (pl.col("qid").cast(pl.Int64) == pl.col("own")))
    hs = set(hit["tid"].to_list())
    pos = int(a["deg"].sum())
    vals = []
    all_hit = 0
    got = dict(hit.group_by("own").len().iter_rows())
    for rid, deg in a.select("rid", "deg").iter_rows():
        n = got.get(rid, 0)
        vals.append(1.0 if deg == 0 else 1.25 * n / (n + 0.25 * deg))
        all_hit += int(n == deg)
    return {"links": len(hs), "link_recall": len(hs) / pos if pos else 1.0,
            "oracle_macro_f05": float(np.mean(vals)), "all_links_recalled": all_hit / len(a) if len(a) else 1.0}, hs


def lexical_hits(run):
    p = run / "metrics.json"
    if not p.exists():
        return set(), []
    m = json.loads(p.read_text(encoding="utf-8"))
    out = set()
    parts = []
    for x in m.get("parts", []):
        q = run / x
        if not q.exists():
            continue
        d = pl.read_parquet(q)
        need(d, ("tid", "qid"), "lexical part")
        if "y" in d.columns:
            out.update(d.filter(pl.col("y") == 1)["tid"].to_list())
        elif "own" in d.columns:
            out.update(d.filter(pl.col("qid").cast(pl.Int64) == pl.col("own"))["tid"].to_list())
        parts.append(x)
    return out, parts


def run(data, dest, cache, model, rev, dev, batch, k, maxlen, sources):
    if batch < 1 or k < 1 or maxlen < 8:
        raise ValueError("invalid batch k or maxlen")
    a, q, r, co, fold = load_run(data, dest)
    if k > len(r):
        k = len(r)
    si = source(sources, model, rev)
    m, use, pars = model_load(model, rev, dev, maxlen)
    rt = serial(r, "passage")
    qt = serial(q, "query")
    dim = m.get_embedding_dimension()
    core = meta_core(data, model, rev, pars, maxlen, dim, co, fold, (r, rt), (q, qt), si)
    base = cache / f"e5_{co}_f{fold}"
    tr = trunc(m, rt, qt, maxlen)
    if not (base / "meta.json").exists() and any(base.glob("*.npy")):
        raise ValueError("cache artifacts lack metadata")
    cache_meta(base / "meta.json", core, tr)
    rp = stage(base, "references", rt, dim, m, batch)
    qp = stage(base, "queries", qt, dim, m, batch)
    del m
    gc.collect()
    if use == "cuda":
        import torch
        torch.cuda.empty_cache()
    rr = np.load(rp, mmap_mode="r")
    qq = np.load(qp, mmap_mode="r")
    t = time.monotonic()
    ans = gpu_search(qq, rr, k, batch) if use == "cuda" else None
    if ans is None:
        hits, rb = cpu_search(qq, rr, k, batch), batch
        mode = "exact_cpu_faiss_ip"
    else:
        hits, rb = ans
        mode = "exact_gpu_fp16"
    out = dest / "dense_candidates.parquet"
    write_candidates(out, q, r, hits)
    cand = pl.read_parquet(out)
    met, dh = score(a, cand)
    lh, parts = lexical_hits(dest)
    pos = int(a["deg"].sum())
    met.update({"country": co, "fold": fold, "anchors": len(a), "queries": len(q),
                "references": len(r), "k": k, "candidate_file": out.name, "cache": str(base),
                "search": mode, "search_batch": rb, "seconds": time.monotonic() - t,
                "lexical_parts": parts, "union_link_recall": len(dh | lh) / pos if pos else 1.0,
                "warning": "classifier/full-pool f0.5 is not evaluated; fp16 brute-force scores are exact only in half precision, not ann, and retrieval benchmarks do not guarantee entity-resolution quality."})
    atom(dest / "embed_metrics.json", met)
    print(json.dumps(met, indent=2), flush=True)
    return met


def check():
    with tf.TemporaryDirectory() as tmp:
        p = path(tmp)
        d = p / "data"
        z = p / "run"
        (d / "train").mkdir(parents=True)
        z.mkdir()
        a = pl.DataFrame({"rid": [4, 8], "nm": ["राम", "quiet"], "ad": ["1 गली", "2 rd"],
                          "co": ["india", "india"], "deg": [1, 0], "fold": [2, 2]})
        q = pl.DataFrame({"rid": [10, 11], "nm": ["राम", "orphan"], "ad": ["1 गली", "x"],
                          "co": ["india", "india"], "own": [4, -1], "sr": [2, 3]})
        r = pl.DataFrame({"rid": [4, 8, 9], "nm": ["राम", "quiet", "other"], "ad": ["1 गली", "2 rd", "z"],
                          "co": ["india"] * 3, "fold": [2, 2, 0]})
        a.write_parquet(z / "anchors.parquet")
        q.write_parquet(z / "queries.parquet")
        r.write_parquet(d / "train/ref.parquet")
        aa, qq, rr, co, fold = load_run(d, z)
        assert co == "india" and fold == 2 and rr["rid"].to_list() == [4, 8]
        s = serial(aa, "passage")[0]
        assert "राम" in s and s.startswith("passage:") and "rid" not in s and "fold" not in s and "deg" not in s
        v = norm([[3, 4], [0, 2]])
        assert np.allclose(np.linalg.norm(v, axis=1), 1)
        ix, ds = topk_np([[1, 0]], [[1, 0], [0, 1], [.5, .5]], 2)
        assert ix.tolist() == [[0, 2]] and np.allclose(ds, [[1, .5]])
        core = {"inputs": {"references": {"sha256": "a"}}}
        cache_meta(p / "meta.json", core)
        try:
            cache_meta(p / "meta.json", {"inputs": {"references": {"sha256": "b"}}})
            raise AssertionError("missing fingerprint mismatch")
        except ValueError:
            pass
        c = pl.DataFrame({"tid": [10], "qid": [4], "ds": [1.], "own": [4], "sr": [2]})
        met, hs = score(aa, c)
        assert hs == {10} and met["link_recall"] == 1 and met["oracle_macro_f05"] == 1
    print("checks passed")


def main():
    root = path(__file__).resolve().parents[1]
    pa = ap.ArgumentParser()
    pa.add_argument("--data", type=path, default=root / "cache/data")
    pa.add_argument("--run", type=path, default=root / "cache/runs/val_india")
    pa.add_argument("--cache", type=path, default=root / "cache/embed")
    pa.add_argument("--model", default=mod0)
    pa.add_argument("--revision", default=rev0)
    pa.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    pa.add_argument("--batch", type=int, default=64)
    pa.add_argument("--k", type=int, default=50)
    pa.add_argument("--maxlen", type=int, default=256)
    pa.add_argument("--sources", type=path, default=src0)
    pa.add_argument("--check", action="store_true")
    x = pa.parse_args()
    if x.check:
        check()
    else:
        run(x.data, x.run, x.cache, x.model, x.revision, x.device, x.batch, x.k, x.maxlen, x.sources)


if __name__ == "__main__":
    main()
