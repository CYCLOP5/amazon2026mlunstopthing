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


def atom(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    q.replace(p)


def tag(s):
    return "".join(x if x.isalnum() else "_" for x in s.lower()).strip("_")


def specs(xs):
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
    if "encoder" in d:
        m = d["encoder"]
        n = int(d.get("params", 0))
        z = d.get("device", dev)
    else:
        embed.source(embed.src0, d["model"], d["revision"])
        m, z, n = embed.model_load(d["model"], d["revision"], dev, ln)
    if n < 0:
        raise ValueError("invalid model parameter count")
    return m, z, n


def enc(m, txt, batch):
    z = m.encode(txt, batch_size=batch, show_progress_bar=False, convert_to_numpy=True,
                 normalize_embeddings=True, precision="float32")
    return embed.norm(z)


def core(data, d, refs, txt, dim, co, sp):
    mp = data / "meta.json"
    return {"v": 1, "model": d["model"], "revision": d["revision"], "role": "passage",
            "format": embed.fmt if embed.family(d["model"]) == "e5" else embed.rawfmt,
            "maxlen": ln, "dtype": "float16", "normalize": True, "dim": dim,
            "country": co, "split": sp, "inputs": embed.fprint(refs, txt),
            "prefixes": {"reference": "passage"},
            "data_meta_sha256": hh.sha256(mp.read_bytes()).hexdigest() if mp.exists() else None}


def oldok(p, c):
    mp = p / "meta.json"
    rp = p / "references.npy"
    pp = p / "references.progress.json"
    if not (mp.exists() and rp.exists() and pp.exists()):
        return None
    try:
        o = json.loads(mp.read_text(encoding="utf-8")).get("core", {})
        pr = json.loads(pp.read_text(encoding="utf-8"))
        a = np.load(rp, mmap_mode="r")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    ks = ("model", "revision", "maxlen", "dtype", "dim", "country", "format",
          "data_meta_sha256")
    if (any(o.get(k) != c.get(k) for k in ks) or
            o.get("prefixes", {}).get("reference") != c["prefixes"]["reference"] or
            o.get("inputs", {}).get("references") != c["inputs"]):
        return None
    if int(pr.get("done", -1)) != len(a) or a.shape != (c["inputs"]["rows"], c["dim"]) or a.dtype != np.float16:
        return None
    return a


def dirs(cache, d, co, sp):
    x = f"{embed.family(d['model'])}_{tag(d['revision'])}_{sp}_{tag(co)}"
    return cache / "hybrid" / x, [cache / "embed-local", cache]


def refs(data, cache, d, m, batch, r, co, sp):
    txt = embed.serial(r, "passage", d["model"])
    dim = int(m.get_embedding_dimension())
    c = core(data, d, r, txt, dim, co, sp)
    dst, roots = dirs(cache, d, co, sp)
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
            old = oldok(root / f"e5_{co}_f0", c) if embed.family(d["model"]) == "e5" and sp == "train" else None
            if old is not None:
                return old, root / f"e5_{co}_f0", "embed"
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
    return np.load(rp, mmap_mode="r"), dst, "hybrid"


def setup(data, cache, country, split="train", fold=0, models=None, device="cuda", batch=64):
    data, cache = path(data), path(cache)
    if batch < 1 or fold not in (0, 1, 2):
        raise ValueError("invalid batch or fold")
    r = pl.read_parquet(data / split / "ref.parquet").filter(pl.col("co") == country).sort("rid")
    if r.is_empty():
        raise ValueError(f"no references for {country}")
    br, ix, eq = block.setup(data, cache / "block", country, split, fold=0)
    if not np.array_equal(r["rid"].to_numpy(), br["rid"].to_numpy()):
        raise ValueError("lexical and dense reference mapping differs")
    ms = specs(models)
    es, total = [], 0
    for d in ms:
        m, dev, n = load(d, device)
        total += n
        if total > mx:
            raise ValueError("active retrieval models exceed 8b parameters")
        a, cp, kind = refs(data, cache, d, m, batch, r, country, split)
        fi = None
        if dev == "cpu":
            import faiss
            fi = faiss.IndexFlatIP(a.shape[1])
            fi.add(np.asarray(a, dtype=np.float32))
        es.append({"spec": d, "model": m, "device": dev, "params": n, "refs": a, "index": fi,
                   "cache": str(cp), "cache_kind": kind, "feature": feat(d)})
    fs = [x["feature"] for x in es]
    if len(set(fs)) != len(fs):
        raise ValueError("dense feature names collide")
    return {"data": data, "cache": cache, "country": country, "split": split, "fold": fold,
            "refs": r, "idx": ix, "eq": eq, "encoders": es, "batch": batch,
            "dense_features": fs, "config": {"models": [{"model": x["spec"]["model"],
            "revision": x["spec"]["revision"], "params": x["params"], "feature": x["feature"], "device": x["device"],
            "reference_cache": x["cache"], "cache_kind": x["cache_kind"]} for x in es],
            "total_params": total, "reference_rows": len(r), "maxlen": ln, "pair_chunk": pc}}


def near(q, r, k, dev, ix=None):
    if dev != "cuda":
        if ix is None:
            import faiss
            ix = faiss.IndexFlatIP(r.shape[1])
            ix.add(np.asarray(r, dtype=np.float32))
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
    bs, bi = [], []
    for lo in range(0, len(r), int(rb)):
        z = torch.tensor(np.asarray(r[lo:lo + rb]), device="cuda", dtype=torch.float16)
        s, i = torch.topk(qb @ z.T, min(k, len(z)), dim=1)
        bs.append(s)
        bi.append(i + lo)
        del z
    s, j = torch.topk(torch.cat(bs, dim=1), k, dim=1)
    i = torch.gather(torch.cat(bi, dim=1), 1, j)
    out = i.cpu().numpy(), s.float().cpu().numpy()
    del qb, bs, bi, s, i
    torch.cuda.empty_cache()
    return out


def pos(ids, want, what):
    o = np.argsort(ids)
    j = np.searchsorted(ids[o], want)
    if (j >= len(ids)).any() or not np.array_equal(ids[o][j], want):
        raise ValueError(f"unknown {what} id")
    return o[j]


def cos(p, ids, qv, refs, r, fld):
    qi = pos(ids, p["tid"].to_numpy(), "query")
    ri = pos(refs["rid"].to_numpy(), p["qid"].to_numpy(), "reference")
    out = np.empty(len(p), dtype=np.float32)
    for lo in range(0, len(p), pc):
        hi = min(lo + pc, len(p))
        a = qv[qi[lo:hi]]
        b = np.asarray(r[ri[lo:hi]], dtype=np.float32)
        out[lo:hi] = np.sum(a * b, axis=1) / np.linalg.norm(b, axis=1)
    return p.with_columns(pl.Series(fld, np.clip(out, -1, 1)))


def search(state, queries, k_lex=10, k_dense=50):
    if k_lex < 1 or k_dense < 1:
        raise ValueError("positive candidate counts required")
    q = queries.sort("rid")
    need = {"rid", "nm", "ad", "nn", "an", "co", "own", "sr"}
    if need - set(q.columns) or q.filter(pl.col("co") != state["country"]).height:
        raise ValueError("invalid query records")
    if q["rid"].n_unique() != len(q):
        raise ValueError("duplicate query ids")
    fs = []
    for qb in q.iter_slices(state["batch"]):
        lx = block.search(qb, state["idx"], state["eq"], k_lex, 1)
        xs = [lx.select("tid", "qid", "ns", "ads", "en", "ea")]
        qv = {}
        for e in state["encoders"]:
            z = enc(e["model"], embed.serial(qb, "query", e["spec"]["model"]), state["batch"])
            qv[e["feature"]] = z
            ii, _ = near(z, e["refs"], min(k_dense, len(state["refs"])), e["device"], e["index"])
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
            p = cos(p, qb["rid"].to_numpy(), qv[e["feature"]], state["refs"], e["refs"], e["feature"])
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


def probe(data, cache, src, out, models=None, device="cuda", batch=64, k_lex=10, k_dense=50):
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
    for i, qb in enumerate(q.iter_slices(batch)):
        p = search(s, qb, k_lex, k_dense)
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

    def get_embedding_dimension(self):
        return 2

    def encode(self, txt, **kw):
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
        ms = [{"model": embed.mod0, "revision": embed.rev0, "encoder": fake(v), "params": 2},
              {"model": embed.qwen0, "revision": "fake", "encoder": fake(v), "params": 3}]
        s = setup(d, p / "cache", "us", fold=2, models=ms, device="cpu", batch=1)
        x = search(s, q, 1, 1)
        y = search(s, q, 1, 1)
        assert s["refs"]["rid"].to_list() == [1, 2, 3, 4]
        assert len({e["cache"] for e in s["encoders"]}) == 2
        assert {1, 2, 4} <= set(x["qid"].to_list()) and 3 not in set(x["qid"].to_list())
        assert x.filter(pl.col("qid") == 1)["ds_e5"][0] > 0
        assert x.filter(pl.col("qid") == 2)["ns"][0] > 0
        assert x.filter(pl.col("y") == 1).is_empty()
        assert x.equals(y) and x.select("tid", "qid").n_unique() == len(x)
        src, out = p / "run", p / "out"
        src.mkdir()
        r.filter(pl.col("rid") == 3).write_parquet(src / "anchors.parquet")
        q.write_parquet(src / "queries.parquet")
        atom(src / "metrics.json", {"country": "us", "fold": 0, "score_version": block.sv})
        z = probe(d, p / "cache", src, out, ms, "cpu", 1, 1, 1)
        assert z["dense_features"] == ["ds_e5", "ds_qwen3"] and (out / "queries.parquet").exists()
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
    a.add_argument("--check", action="store_true")
    x = a.parse_args()
    if x.check:
        check()
    elif x.run and x.out:
        print(json.dumps(probe(x.data, x.cache, x.run, x.out, device=x.device, batch=x.batch,
                               k_lex=x.k_lex, k_dense=x.k_dense), indent=2))
    else:
        raise ValueError("use --check or --run and --out")


if __name__ == "__main__":
    main()
