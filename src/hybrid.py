import argparse as ap
import hashlib as hh
import json
import shutil as sh
import tempfile as tf
from pathlib import Path as path

import numpy as np
import polars as pl

import block
import embed


mx = 8_000_000_000
ln = 256
pc = 20_000
qc = 512


def atom(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    q.replace(p)


def tag(s):
    return "".join(x if x.isalnum() else "_" for x in s.lower()).strip("_")


def specs(xs=None):
    if xs is None:
        src = json.loads(embed.src0.read_text(encoding="utf-8"))
        xs = [(x["model"], x["revision"]) for x in src
              if x["model"] in (embed.mod0, embed.qwen0)]
    out = []
    for x in xs:
        if isinstance(x, dict):
            d = dict(x)
        else:
            d = {"model": x[0], "revision": x[1]}
        embed.family(d["model"])
        if not d.get("revision"):
            raise ValueError("model revision required")
        out.append(d)
    if not 1 <= len(out) <= 2:
        raise ValueError("one or two retrieval models required")
    if len({(x["model"], x["revision"]) for x in out}) != len(out):
        raise ValueError("duplicate retrieval model")
    return out


def feat(d):
    return "ds_" + embed.family(d["model"]).replace("-", "_")


def load(d, dev):
    src = embed.source(embed.src0, d["model"], d["revision"])
    if "encoder" in d:
        m = d["encoder"]
        n = int(d.get("params", 0))
        z = d.get("device", dev)
    else:
        m, z, n = embed.model_load(d["model"], d["revision"], dev, ln)
    if n < 0:
        raise ValueError("invalid model parameter count")
    return m, z, n, src


def enc(m, txt, batch):
    z = m.encode(txt, batch_size=batch, show_progress_bar=False, convert_to_numpy=True,
                 normalize_embeddings=True, precision="float32")
    return embed.norm(z)


def core(data, d, refs, txt, dim, co, pars, fallback=False):
    mp = data / "meta.json"
    f = embed.family(d["model"])
    out = {"v": 1, "model": d["model"], "revision": d["revision"], "params": pars,
           "role": "passage", "format": embed.fmt, "maxlen": ln, "dtype": "float16",
           "normalize": True, "dim": dim, "country": co,
           "inputs": {"references": embed.fprint(refs, txt)}, "prefixes": {"reference": "passage"},
           "data_meta_sha256": hh.sha256(mp.read_bytes()).hexdigest() if mp.exists() else None}
    if fallback:
        out["scope"] = "global"
    if f != "e5":
        out["serialization"] = {"family": f, "reference": embed.rawfmt}
        out["encoding"] = {"normalize": True, "precision": "float32"}
        out["padding_side"] = "left" if f == "qwen3" else None
    return out


def oldok(p, c):
    mp = p / "meta.json"
    rp = p / "references.npy"
    pp = p / "references.progress.json"
    if not (mp.exists() or rp.exists() or pp.exists()):
        return None
    if not (mp.exists() and rp.exists() and pp.exists()):
        raise ValueError("orphaned embedding reference cache")
    try:
        o = json.loads(mp.read_text(encoding="utf-8")).get("core", {})
        pr = json.loads(pp.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        raise ValueError("invalid embedding reference cache")
    ks = ("model", "revision", "params", "maxlen", "dtype", "dim", "country", "format",
          "data_meta_sha256")
    if (any(o.get(k) != c.get(k) for k in ks) or
            o.get("prefixes", {}).get("reference") != c["prefixes"]["reference"] or
            o.get("inputs", {}).get("references") != c["inputs"]["references"]):
        raise ValueError("stale embedding reference cache")
    if c.get("scope") and o.get("scope") != c["scope"]:
        raise ValueError("stale embedding reference cache")
    if c.get("serialization") and any(o.get("serialization", {}).get(k) != v
                                        for k, v in c["serialization"].items()):
        raise ValueError("stale embedding serialization metadata")
    if c.get("padding_side") != o.get("padding_side"):
        raise ValueError("stale embedding padding metadata")
    if c.get("encoding") and any(o.get("encoding", {}).get(k) != v for k, v in c["encoding"].items()):
        raise ValueError("stale embedding encoding metadata")
    done, rows = int(pr.get("done", -1)), int(pr.get("rows", -1))
    if rows != c["inputs"]["references"]["rows"] or not 0 <= done <= rows:
        raise ValueError("invalid embedding reference progress")
    if done != rows:
        return None
    try:
        a = np.load(rp, mmap_mode="r")
    except (OSError, ValueError):
        raise ValueError("invalid embedding reference array")
    if a.shape != (rows, c["dim"]) or a.dtype != np.float16:
        raise ValueError("stale embedding reference shape")
    return a


def dirs(cache, d, co, sp, fallback=False):
    x = f"{embed.family(d['model'])}_{tag(d['revision'])}_{sp}_{'global_' if fallback else 'all_'}{tag(co)}"
    return cache / "hybrid" / x, [cache / "embed-local", cache]


def embp(root, d, co):
    if embed.family(d["model"]) == "e5":
        return root / f"e5_{co}_f0"
    return embed.cache_base(root, d["model"], d["revision"], co, 0)


def refs(data, cache, d, m, batch, r, co, sp, pars, fallback=False):
    txt = embed.serial(r, "passage", d["model"])
    dim = int(m.get_embedding_dimension())
    c = core(data, d, r, txt, dim, co, pars, fallback)
    dst, roots = dirs(cache, d, co, sp, fallback)
    mp = dst / "references.meta.json"
    rp = dst / "references.npy"
    pp = dst / "references.progress.json"
    if mp.exists():
        o = json.loads(mp.read_text(encoding="utf-8"))
        if o.get("core") != c:
            raise ValueError("stale hybrid reference cache")
        if not (rp.exists() and pp.exists()):
            raise ValueError("orphaned hybrid reference cache")
        a = np.load(rp, mmap_mode="r+")
        done = int(json.loads(pp.read_text(encoding="utf-8")).get("done", -1))
        if a.shape != (len(r), dim) or a.dtype != np.float16 or not 0 <= done <= len(r):
            raise ValueError("invalid hybrid reference cache")
    else:
        if rp.exists() or pp.exists():
            raise ValueError("orphaned hybrid reference cache")
        for root in roots:
            old = oldok(embp(root, d, co), c) if sp == "train" and not fallback else None
            if old is not None:
                return old, embp(root, d, co), "embed", c
        dst.mkdir(parents=True, exist_ok=True)
        atom(mp, {"core": c})
        a = np.lib.format.open_memmap(rp, mode="w+", dtype=np.float16, shape=(len(r), dim))
        a.flush()
        done = 0
        atom(pp, {"done": done, "rows": len(r)})
    for lo in range(done, len(r), batch):
        hi = min(lo + batch, len(r))
        a[lo:hi] = enc(m, txt[lo:hi], batch).astype(np.float16)
        a.flush()
        atom(pp, {"done": hi, "rows": len(r)})
    return np.load(rp, mmap_mode="r"), dst, "hybrid", c


def ridfp(r):
    h = hh.sha256()
    for i in r["rid"].to_list():
        h.update(str(i).encode())
        h.update(b"\0")
    return h.hexdigest()


def setup(data, cache, country, split="train", fold=0, models=None, device="cuda", batch=64):
    data, cache = path(data), path(cache)
    if batch < 1 or fold not in (0, 1, 2):
        raise ValueError("invalid batch or fold")
    r, ix, eq = block.setup(data, cache / "block", country, split, fold=fold)
    r = r.sort("rid")
    allr = pl.read_parquet(data / split / "ref.parquet").sort("rid")
    fallback = block.global_scope(allr, country)
    if not fallback:
        allr = allr.filter(pl.col("co") == country)
    if allr["rid"].n_unique() != len(allr):
        raise ValueError("duplicate full reference ids")
    ri = pos(allr["rid"].to_numpy(), r["rid"].to_numpy(), "reference")
    cache_country = "all" if fallback else country
    ms = specs(models)
    es, total = [], 0
    for d in ms:
        m, dev, n, src = load(d, device)
        total += n
        if total > mx:
            raise ValueError("active retrieval models exceed 8b parameters")
        a, cp, kind, c = refs(data, cache, d, m, batch, allr, cache_country, split, n, fallback)
        fi = None
        if dev == "cpu":
            import faiss
            fi = faiss.IndexFlatIP(a.shape[1])
            z = np.empty((len(ri), a.shape[1]), dtype=np.float32)
            for lo in range(0, len(ri), pc):
                hi = min(lo + pc, len(ri))
                z[lo:hi] = a[ri[lo:hi]]
            fi.add(z)
        es.append({"spec": d, "model": m, "device": dev, "params": n, "source": src, "refs": a, "rows": ri,
                   "index": fi, "cache": str(cp), "cache_kind": kind, "core": c, "feature": feat(d)})
    fs = [x["feature"] for x in es]
    if len(set(fs)) != len(fs):
        raise ValueError("dense feature names collide")
    return {"data": data, "cache": cache, "country": country, "fallback": fallback, "split": split, "fold": fold,
            "refs": r, "idx": ix, "eq": eq, "encoders": es, "batch": batch,
            "dense_features": fs, "config": {"models": [{"model": x["spec"]["model"],
            "revision": x["spec"]["revision"], "params": x["params"], "source": x["source"], "feature": x["feature"], "device": x["device"],
            "reference_cache": x["cache"], "cache_kind": x["cache_kind"],
            "encoding_corpus": x["core"]["inputs"]["references"],
            "search_pool": {"rows": len(r), "rid_sha256": ridfp(r)}} for x in es],
            "total_params": total, "reference_rows": len(r), "encoding_reference_rows": len(allr), "fallback": fallback,
            "search_reference_rows": len(r), "maxlen": ln, "pair_chunk": pc}}


def near(q, r, k, dev, ix=None, rows=None):
    nr = len(r) if rows is None else len(rows)
    if dev != "cuda":
        if ix is None:
            import faiss
            ix = faiss.IndexFlatIP(r.shape[1])
            ix.add(np.asarray(r if rows is None else r[rows], dtype=np.float32))
        s, i = ix.search(np.asarray(q, dtype=np.float32), k)
        return i, s
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    free, _ = torch.cuda.mem_get_info()
    q = np.asarray(q, dtype=np.float32)
    dim = r.shape[1]
    room = free - 512 * 1024**2 - len(q) * dim * 2
    rb = room // max(1, dim * 2 + len(q) * 2)
    if rb < k:
        raise RuntimeError("gpu cannot hold a bounded reference block; use cpu or reduce batch")
    qb = torch.tensor(q, device="cuda", dtype=torch.float16)
    bs = bi = None
    for lo in range(0, nr, int(rb)):
        z = torch.tensor(np.asarray(r[lo:lo + rb] if rows is None else r[rows[lo:lo + rb]]),
                         device="cuda", dtype=torch.float16)
        s, i = torch.topk(qb @ z.T, min(k, len(z)), dim=1)
        if bs is None:
            bs, bi = s, i + lo
        else:
            s, j = torch.topk(torch.cat((bs, s), dim=1), k, dim=1)
            bs = s
            bi = torch.gather(torch.cat((bi, i + lo), dim=1), 1, j)
        del z
    out = bi.cpu().numpy(), bs.float().cpu().numpy()
    del qb, bs, bi
    torch.cuda.empty_cache()
    return out


def pos(ids, want, what):
    o = np.argsort(ids)
    j = np.searchsorted(ids[o], want)
    if (j >= len(ids)).any() or not np.array_equal(ids[o][j], want):
        raise ValueError(f"unknown {what} id")
    return o[j]


def cos(p, ids, qv, refs, r, rows, fld):
    qi = pos(ids, p["tid"].to_numpy(), "query")
    ri = pos(refs["rid"].to_numpy(), p["qid"].to_numpy(), "reference")
    out = np.empty(len(p), dtype=np.float32)
    for lo in range(0, len(p), pc):
        hi = min(lo + pc, len(p))
        a = qv[qi[lo:hi]]
        b = np.asarray(r[ri[lo:hi] if rows is None else rows[ri[lo:hi]]], dtype=np.float32)
        out[lo:hi] = np.sum(a * b, axis=1) / np.linalg.norm(b, axis=1)
    return p.with_columns(pl.Series(fld, np.clip(out, -1, 1)))


def search(state, queries, k_lex=10, k_dense=50, threads=8):
    if k_lex < 1 or k_dense < 1 or threads < 1:
        raise ValueError("positive candidate counts required")
    q = queries.sort("rid")
    need = {"rid", "nm", "ad", "nn", "an", "co", "own", "sr"}
    if need - set(q.columns) or q.filter(pl.col("co") != state["country"]).height:
        raise ValueError("invalid query records")
    if q["rid"].n_unique() != len(q):
        raise ValueError("duplicate query ids")
    fs = []
    for qb in q.iter_slices(qc):
        lx = block.search(qb, state["idx"], state["eq"], k_lex, threads)
        xs = [lx.select("tid", "qid", "ns", "ads", "en", "ea")]
        qv = {}
        for e in state["encoders"]:
            z = enc(e["model"], embed.serial(qb, "query", e["spec"]["model"]), state["batch"])
            qv[e["feature"]] = z
            ii, _ = near(z, e["refs"], min(k_dense, len(state["refs"])), e["device"], e["index"], e["rows"])
            xs.append(pl.DataFrame({"tid": np.repeat(qb["rid"].to_numpy(), ii.shape[1]),
                                    "qid": state["refs"]["rid"].to_numpy()[ii].reshape(-1),
                                    "ns": np.zeros(ii.size, np.float32), "ads": np.zeros(ii.size, np.float32),
                                    "en": np.zeros(ii.size, np.uint8), "ea": np.zeros(ii.size, np.uint8)},
                                   schema_overrides=block.schema))
        p = pl.concat(xs).group_by("tid", "qid").agg(pl.col("ns", "ads", "en", "ea").max()).sort("tid", "qid")
        p = p.join(qb.select(pl.col("rid").alias("tid"), "own", "sr"), on="tid", how="left", validate="m:1")
        p = p.with_columns((pl.col("qid").cast(pl.Int64) == pl.col("own")).cast(pl.UInt8).alias("y"))
        p = block.rescore(qb, p, state["idx"])
        for e in state["encoders"]:
            p = cos(p, qb["rid"].to_numpy(), qv[e["feature"]], state["refs"], e["refs"], e["rows"], e["feature"])
        fs.append(p.sort("tid", "qid"))
    if not fs:
        return pl.DataFrame({"tid": [], "qid": [], "ns": [], "ads": [], "en": [], "ea": [],
                             "own": [], "sr": [], "y": [], **{x: [] for x in state["dense_features"]}})
    return pl.concat(fs).sort("tid", "qid")


def score(a, p):
    hit = p.filter(pl.col("y") == 1)
    got = dict(hit.group_by("own").len().iter_rows())
    vals = [1.0 if d == 0 else 1.25 * got.get(i, 0) / (got.get(i, 0) + .25 * d)
            for i, d in a.select("rid", "deg").iter_rows()]
    n = int(a["deg"].sum())
    return {"retrieved_links": len(hit), "link_recall": len(hit) / n if n else 1.0,
            "oracle_macro_f05": float(np.mean(vals)) if vals else 1.0}


def probe(data, cache, src, out, models=None, device="cuda", batch=64, k_lex=10, k_dense=50, threads=8):
    src, out = path(src), path(out)
    if out.exists() or src.resolve() == out.resolve():
        raise ValueError("output run directory must be new")
    met = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
    a, q, _, co, fold = embed.load_run(path(data), src)
    if met.get("country") != co or int(met.get("fold", -1)) != fold or met.get("score_version") != block.sv:
        raise ValueError("run metadata mismatch")
    s = setup(data, cache, co, "train", fold, models, device, batch)
    out.mkdir(parents=True)
    for n in ("anchors.parquet", "queries.parquet"):
        sh.copyfile(src / n, out / n)
    parts, allp = [], []
    for i, qb in enumerate(q.iter_slices(qc)):
        p = search(s, qb, k_lex, k_dense, threads)
        n = f"pairs_{i:05d}.parquet"
        p.write_parquet(out / n, compression="zstd")
        parts.append(n)
        allp.append(p)
    p = pl.concat(allp) if allp else pl.DataFrame()
    z = dict(met)
    z.update(score(a, p), country=co, fold=fold, score_version=block.sv, parts=parts,
             derived_from=str(src), dense_features=s["dense_features"], retrieval=s["config"],
             k_lex=k_lex, k_dense=k_dense, reference_pool=s["config"]["reference_rows"],
             queries=len(q), anchors=len(a), pairs=len(p))
    atom(out / "metrics.json", z)
    return z


class fake:
    def __init__(self, v):
        self.v = v
        self.n = 0

    def get_embedding_dimension(self):
        return 2

    def encode(self, txt, **kw):
        self.n += len(txt)
        return np.asarray([[.6, .8] if "query:" in s.lower() else self.v[s.split("name: ", 1)[1].split("\n", 1)[0]]
                           for s in txt], np.float32)


def check():
    with tf.TemporaryDirectory() as tmp:
        p = path(tmp)
        d = p / "data"
        (d / "train").mkdir(parents=True)
        (d / "meta.json").write_text("{}\n")
        r = pl.DataFrame({"rid": [1, 2, 3, 4], "nm": ["lex coffee", "coffee shop", "gold hidden", "lex coffee"],
                          "ad": ["one road", "two road", "zzzzz", "four road"], "co": ["us"] * 4,
                          "nn": ["lex coffee", "coffee shop", "gold hidden", "lex coffee"],
                          "an": ["one road", "two road", "zzzzz", "four road"], "fold": [2, 0, 0, 0], "deg": [0, 0, 1, 0]})
        q = pl.DataFrame({"rid": [10], "nm": ["lex coffee"], "ad": ["road"], "co": ["us"],
                          "nn": ["lex coffee"], "an": ["road"], "own": [3], "sr": [2]})
        r.write_parquet(d / "train/ref.parquet")
        v = {"lex coffee": [1, 0], "coffee shop": [0, 1], "gold hidden": [-1, 0]}
        qrev = next(x["revision"] for x in json.loads(embed.src0.read_text(encoding="utf-8")) if x["model"] == embed.qwen0)
        reuse = p / "reuse"
        legacy = reuse / "e5_us_f0"
        qcache = embed.cache_base(reuse, embed.qwen0, qrev, "us", 0)
        for base, model, rev, pars in ((legacy, embed.mod0, embed.rev0, 2),
                                       (qcache, embed.qwen0, qrev, 3)):
            base.mkdir(parents=True)
            rt = embed.serial(r, "passage", model)
            mc = embed.meta_core(d, model, rev, pars, ln, 2, "us", 0, (r, rt),
                                 (q, embed.serial(q, "query", model)), embed.source(embed.src0, model, rev), 1)
            atom(base / "meta.json", {"core": mc})
            np.save(base / "references.npy", np.asarray([[1, 0], [0, 1], [-1, 0], [1, 0]], np.float16))
            atom(base / "references.progress.json", {"done": len(r), "rows": len(r)})
        e5, qw = fake(v), fake(v)
        ms = [{"model": embed.mod0, "revision": embed.rev0, "encoder": e5, "params": 2},
              {"model": embed.qwen0, "revision": qrev, "encoder": qw, "params": 3}]
        s = setup(d, reuse, "us", fold=2, models=ms, device="cpu", batch=1)
        assert e5.n == qw.n == 0
        x = search(s, q, 1, 1)
        assert s["refs"]["rid"].to_list() == [1]
        assert all(set(e["rows"].tolist()) <= {0} for e in s["encoders"])
        assert len({e["cache"] for e in s["encoders"]}) == 2
        assert {e["cache_kind"] for e in s["encoders"]} == {"embed"}
        assert x["qid"].to_list() == [1]
        assert x.filter(pl.col("qid") == 1)["ds_e5"][0] > 0
        assert x.filter(pl.col("qid") == 1)["ns"][0] > 0
        assert x.filter(pl.col("y") == 1).is_empty()
        s2 = setup(d, reuse, "us", fold=2, models=ms, device="cpu", batch=2)
        assert x.equals(search(s2, q, 1, 1)) and x.select("tid", "qid").n_unique() == len(x)
        bad = p / "bad"
        qbad = embed.cache_base(bad, embed.qwen0, qrev, "us", 0)
        qbad.mkdir(parents=True)
        sh.copyfile(qcache / "meta.json", qbad / "meta.json")
        np.save(qbad / "references.npy", np.zeros((len(r), 2), np.float16))
        atom(qbad / "references.progress.json", {"done": len(r) - 1, "rows": len(r)})
        badq = fake(v)
        bads = setup(d, bad, "us", fold=2, models=[{"model": embed.qwen0, "revision": qrev,
                      "encoder": badq, "params": 3}], device="cpu", batch=1)
        assert bads["encoders"][0]["cache_kind"] == "hybrid" and badq.n == len(r)
        assert pos(np.array([1, 3]), np.array([3]), "reference").tolist() == [1]
        src, out = p / "run", p / "out"
        src.mkdir()
        r.filter(pl.col("rid") == 3).write_parquet(src / "anchors.parquet")
        q.write_parquet(src / "queries.parquet")
        atom(src / "metrics.json", {"country": "us", "fold": 0, "score_version": block.sv})
        z = probe(d, p / "cache", src, out, ms, "cpu", 1, 1, 1)
        assert z["dense_features"] == ["ds_e5", "ds_qwen3"] and (out / "queries.parquet").exists()
        assert z["retrieval"]["models"][0]["source"]["license"] == "mit"
    print("checks passed")


def main():
    root = path(__file__).resolve().parents[1]
    a = ap.ArgumentParser()
    a.add_argument("--data", type=path, default=root / "cache/data")
    a.add_argument("--cache", type=path, default=root / "cache")
    a.add_argument("--run", type=path)
    a.add_argument("--out", type=path)
    a.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a.add_argument("--batch", type=int, default=64)
    a.add_argument("--k-lex", type=int, default=10)
    a.add_argument("--k-dense", type=int, default=50)
    a.add_argument("--threads", type=int, default=8)
    a.add_argument("--check", action="store_true")
    x = a.parse_args()
    if x.check:
        check()
    elif x.run and x.out:
        print(json.dumps(probe(x.data, x.cache, x.run, x.out, device=x.device, batch=x.batch,
                               k_lex=x.k_lex, k_dense=x.k_dense, threads=x.threads), indent=2))
    else:
        raise ValueError("use --check or --run and --out")


if __name__ == "__main__":
    main()
