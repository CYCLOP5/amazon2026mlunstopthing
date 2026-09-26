import argparse as ap
import gc
import hashlib as hh
import json
import sys
import tempfile as tf
import time
import types
from pathlib import Path as path

import numpy as np
import polars as pl


mod0 = "intfloat/multilingual-e5-base"
rev0 = "d128750597153bb5987e10b1c3493a34e5a4502a"
src0 = path(__file__).resolve().parents[1] / "reports/model_sources.json"
fmt = "{role}: name: {nm}\naddress: {ad}\ncountry: {co}"
rawfmt = "name: {nm}\naddress: {ad}\ncountry: {co}"
qwen0 = "Qwen/Qwen3-Embedding-0.6B"
large0 = "intfloat/multilingual-e5-large-instruct"
bge0 = "BAAI/bge-m3"
inst0 = "Given a business record, retrieve records for the SAME BUSINESS identity using the business name and address."


def sha(b):
    return hh.sha256(b).hexdigest()



def atom(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    q.replace(p)


def family(model):
    if model == "intfloat/multilingual-e5-small":
        return "e5-small"
    if model == mod0:
        return "e5"
    if model == qwen0:
        return "qwen3"
    if model == large0:
        return "e5-large"
    if model == bge0:
        return "bge-m3"
    raise ValueError("unsupported embedding model family")


def serial(d, role, model=mod0):
    if role not in ("query", "passage"):
        raise ValueError("invalid e5 role" if model == mod0 else "invalid embedding role")
    f = family(model)
    if f == "e5-small":
        return [f"query: {str(n)} | {str(a)}" for n, a in d.select("nm", "ad").iter_rows()]
    if f == "e5":
        return [fmt.format(role=role, nm=str(n), ad=str(a), co=str(c))
                for n, a, c in d.select("nm", "ad", "co").iter_rows()]
    raw = [rawfmt.format(nm=str(n), ad=str(a), co=str(c))
           for n, a, c in d.select("nm", "ad", "co").iter_rows()]
    if f == "qwen3" and role == "query":
        return [f"Instruct: {inst0}\nQuery:{x}" for x in raw]
    if f == "e5-large" and role == "query":
        return [f"Instruct: {inst0}\nQuery: {x}" for x in raw]
    return raw


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
            if x.get("license", "").lower() not in {"mit", "apache-2.0"}:
                raise ValueError("ineligible model license")
            return x
    raise ValueError("model revision is absent from source metadata")


def check_params(n):
    if n > 8_000_000_000:
        raise ValueError("loaded model exceeds 8b parameters")


def hf_cache():
    p = path(__file__).resolve().parents[1] / "models/hf"
    return p if p.is_dir() else None


def model_load(model, rev, dev, maxlen, checkpoint=None):
    import torch
    from sentence_transformers import SentenceTransformer

    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    use = dev
    kw = {"model_kwargs": {"attn_implementation": "sdpa"}}
    if use == "cuda":
        kw["model_kwargs"]["torch_dtype"] = torch.float16
    if family(model) == "qwen3":
        kw["tokenizer_kwargs"] = {"padding_side": "left"}
    cache = hf_cache()
    if cache is not None:
        kw.update(cache_folder=str(cache), local_files_only=True)
    m = SentenceTransformer(str(checkpoint) if checkpoint else model, revision=None if checkpoint else rev,
                            trust_remote_code=False, device=use, **kw)
    if family(model) == "qwen3" and m.tokenizer.padding_side != "left":
        raise ValueError("qwen requires left tokenizer padding")
    m.max_seq_length = maxlen
    if use == "cuda":
        m.half()
    else:
        m.float()
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    check_params(n)
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


def meta_core(data, model, rev, pars, maxlen, dim, co, fold, refs, qs, src, batch):
    mp = data / "meta.json"
    out = {"v": 1, "model": model, "revision": rev, "params": pars, "maxlen": maxlen,
           "dim": dim, "dtype": "float16", "prefixes": {"reference": "passage", "target": "query"},
           "format": fmt, "country": co, "fold": fold, "pool": "fold2" if fold == 2 else "all",
           "data_meta_sha256": sha(mp.read_bytes()) if mp.exists() else None,
           "inputs": {"references": fprint(refs[0], refs[1]), "queries": fprint(qs[0], qs[1])},
           "source": src}
    if family(model) != "e5":
        out.update({"serialization": {"family": family(model), "reference": rawfmt,
                                       "query": f"Instruct: {inst0}\nQuery:{rawfmt}" if family(model) == "qwen3" else
                                                f"Instruct: {inst0}\nQuery: {rawfmt}" if family(model) == "e5-large" else rawfmt},
                    "encoding": {"batch": batch, "normalize": True, "precision": "float32"},
                    "padding_side": "left" if family(model) == "qwen3" else None})
    return out


def cache_base(cache, model, rev, co, fold):
    if family(model) == "e5":
        return cache / f"e5_{co}_f{fold}"
    tag = "".join(x if x.isalnum() else "_" for x in model.lower()).strip("_")
    r = "".join(x if x.isalnum() else "_" for x in rev.lower()).strip("_")
    return cache / f"{family(model)}_{tag}_{r}_{co}_f{fold}"


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
    family(model)
    a, q, r, co, fold = load_run(data, dest)
    if k > len(r):
        k = len(r)
    si = source(sources, model, rev)
    m, use, pars = model_load(model, rev, dev, maxlen)
    rt = serial(r, "passage", model)
    qt = serial(q, "query", model)
    dim = m.get_embedding_dimension()
    core = meta_core(data, model, rev, pars, maxlen, dim, co, fold, (r, rt), (q, qt), si, batch)
    base = cache_base(cache, model, rev, co, fold)
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
        a = pl.DataFrame({"rid": [4, 8], "nm": ["राम Café", "quiet"], "ad": ["1 गली", "2 rd"],
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
        assert s == "passage: name: राम Café\naddress: 1 गली\ncountry: india"
        zq = serial(aa, "query", qwen0)[0]
        zd = serial(aa, "passage", qwen0)[0]
        assert zq == f"Instruct: {inst0}\nQuery:name: राम Café\naddress: 1 गली\ncountry: india"
        assert zd == "name: राम Café\naddress: 1 गली\ncountry: india" and "Instruct:" not in zd
        assert serial(aa, "query", large0)[0] == f"Instruct: {inst0}\nQuery: {zd}"
        assert serial(aa, "passage", large0)[0] == zd
        assert family(large0) == "e5-large"
        assert serial(aa, "query", bge0)[0] == zd
        assert "rid" not in zq and "fold" not in zq and "deg" not in zq and "own" not in zq
        ec = meta_core(d, mod0, rev0, 1, 32, 2, co, fold, (rr, serial(rr, "passage")),
                       (qq, serial(qq, "query")), {"license": "mit"}, 1)
        assert ec["format"] == fmt and ec["prefixes"] == {"reference": "passage", "target": "query"}
        assert "serialization" not in ec and "encoding" not in ec and "padding_side" not in ec
        qc = meta_core(d, qwen0, "one", 1, 32, 2, co, fold, (rr, [zd] * len(rr)),
                       (qq, serial(qq, "query", qwen0)), {"license": "apache-2.0"}, 1)
        assert qc["serialization"]["query"] == f"Instruct: {inst0}\nQuery:{rawfmt}" and qc["padding_side"] == "left"
        lc = meta_core(d, large0, "large", 1, 512, 2, co, fold, (rr, [zd] * len(rr)),
                       (qq, serial(qq, "query", large0)), {"license": "mit"}, 1)
        assert lc["serialization"]["query"] == f"Instruct: {inst0}\nQuery: {rawfmt}" and lc["maxlen"] == 512
        assert cache_base(p, mod0, rev0, "india", 0) == p / "e5_india_f0"
        assert cache_base(p, qwen0, "one", "india", 0) != cache_base(p, qwen0, "two", "india", 0)
        assert cache_base(p, qwen0, "one", "india", 0) != cache_base(p, bge0, "one", "india", 0)
        check_params(8_000_000_000)
        try:
            check_params(8_000_000_001)
            raise AssertionError("missing parameter limit")
        except ValueError:
            pass
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
        from importlib import util
        moved = p / "moved/src/embed.py"
        moved.parent.mkdir(parents=True)
        moved.write_text(path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
        atom(p / "moved/reports/model_sources.json", [{"model": "test", "revision": "local", "license": "apache-2.0"}])
        spec = util.spec_from_file_location("moved_embed", moved)
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.source(mod.src0, "test", "local")["license"] == "apache-2.0"
        calls = []

        class fake_model:
            def __init__(self, *args, **kw):
                calls.append((args, kw))
                self.tokenizer = types.SimpleNamespace(padding_side="left")

            def parameters(self):
                return ()

            def float(self):
                return self

            def half(self):
                return self

            def eval(self):
                return self

        old = {n: sys.modules.get(n) for n in ("torch", "sentence_transformers")}
        sys.modules["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=fake_model)
        try:
            packed = moved.parents[1] / "models/hf"
            packed.mkdir(parents=True)
            mod.model_load(mod.mod0, mod.rev0, "cpu", 12)
            plain = p / "plain/src/embed.py"
            plain.parent.mkdir(parents=True)
            plain.write_text(path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
            spec = util.spec_from_file_location("plain_embed", plain)
            plainmod = util.module_from_spec(spec)
            spec.loader.exec_module(plainmod)
            plainmod.model_load(plainmod.mod0, plainmod.rev0, "cpu", 12)
        finally:
            for n, v in old.items():
                if v is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = v
        assert calls[0][0] == (mod.mod0,) and calls[0][1]["revision"] == mod.rev0
        assert calls[0][1]["cache_folder"] == str(packed) and calls[0][1]["local_files_only"] is True
        assert "cache_folder" not in calls[1][1] and "local_files_only" not in calls[1][1]
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
