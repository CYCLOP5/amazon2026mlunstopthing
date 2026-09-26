import argparse as ap
import hashlib as hh
import json
import os
import subprocess as sp
import tempfile as tf
from pathlib import Path as path
import time

import numpy as np
import polars as pl

try:
    import block
    import embed
    import feat
    import hybrid
    import infer
    import neural
    import train
    import tfeat
except ImportError:
    from src import block, embed, feat, hybrid, infer, neural, train, tfeat


_sha = infer._sha
_json = infer._json
_write = infer._write
_pq = infer._pq
_npy = infer._npy


ver = 1


def _pairs(d):
    h = hh.sha256()
    for t, q in d.select("tid", "qid").sort("tid", "qid").iter_rows():
        h.update(int(t).to_bytes(4, "little", signed=False))
        h.update(int(q).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def _files(d, names):
    z = {}
    for n in names:
        p = d / n
        if not p.is_file():
            raise ValueError(f"missing model file {p}")
        z[n] = _sha(p)
    return z


def _gate(d):
    d = path(d).resolve()
    m = _json(d / "metadata.json")
    names, dense = train._contract(m)
    if m.get("score_version") != block.sv:
        raise ValueError("gate candidate score version mismatch")
    fs = m.get("model_files", {})
    if not isinstance(m.get("models"), list) or not m["models"]:
        raise ValueError("gate metadata has no models")
    if any(not isinstance(fs.get(n), str) or path(fs[n]).name != fs[n] for n in m["models"]):
        raise ValueError("invalid gate model file name")
    files = [fs[n] for n in m["models"]]
    if m.get("feature_backend") == "hybrid-v2":
        normalizer = m.get("normalizer", {})
        name = normalizer.get("file")
        if not isinstance(name, str) or path(name).name != name or _sha(d / name) != normalizer.get("sha256"):
            raise ValueError("rich gate normalization model changed")
        files.append(name)
    f = _files(d, files)
    z = {"metadata_sha256": _sha(d / "metadata.json"), "files": f}
    if m.get("feature_backend") in (tfeat.backend, "hybrid-v2"):
        if m["models"] != ["lgb"]:
            raise ValueError("teammate gate requires its validated lightgbm booster")
        z["feature_backend"] = m["feature_backend"]
    z["sha256"] = hh.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()
    return m, names, dense, z


def _neural(d):
    d = path(d).resolve()
    m = _json(d / "neural_metadata.json")
    n = m.get("parameters")
    if not isinstance(n, int) or n < 0 or n > hybrid.mx:
        raise ValueError("invalid neural parameter count")
    if m.get("problem_type") != "multi_label_classification":
        raise ValueError("invalid neural model metadata")
    if str(m.get("source", {}).get("license", "")).lower() not in {"mit", "apache-2.0", "apache2", "apache 2.0"}:
        raise ValueError("neural model license is not mit or apache-2.0")
    ws = sorted(p.name for p in d.glob("*.safetensors"))
    ws += sorted(p.name for p in d.glob("pytorch_model*.bin"))
    if not ws:
        raise ValueError("neural model has no weights")
    cs = [x for x in ("config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json",
                      "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                      "sentencepiece.bpe.model", "spiece.model", "vocab.txt", "vocab.json", "merges.txt") if (d / x).is_file()]
    z = {"metadata_sha256": _sha(d / "neural_metadata.json"), "weights": _files(d, ws),
         "tokenization": _files(d, cs), "parameters": n}
    z["sha256"] = hh.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()
    return m, z


def _retrievers(xs):
    if xs is None:
        allx = {embed.family(x["model"]): x for x in hybrid.specs()}
        xs = [allx["e5"]]
    else:
        xs = hybrid.specs(xs)
    fs = [embed.family(x["model"]) for x in xs]
    if fs not in (["e5"], ["e5", "qwen3"], ["e5", "qwen3", "e5-large"], ["e5-small"], ["e5-small", "qwen3"]):
        raise ValueError("retrievers need e5 or e5-small, optionally followed by supported complements")
    return xs


def _cfg(data, gate, nn, rs, klex, kdense, kgate, neural_weight, device="cpu", neural_floor=None):
    dm = _json(data / "meta.json")
    ret = []
    for x in rs:
        f = embed.family(x["model"])
        ret.append({"model": x["model"], "revision": x["revision"], "feature": hybrid.feat(x),
                     "format": embed.fmt if f == "e5" else embed.rawfmt, "family": f})
        if f == "e5-small":
            ret[-1]["format"] = "query: {nm} | {ad}"
        if x.get("checkpoint_sha256"):
            ret[-1]["checkpoint_sha256"] = x["checkpoint_sha256"]
    import torch
    precision = "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "fp16" if device == "cuda" else "fp32"
    src = path(__file__).resolve().parent
    srcs = ["data.py", "block.py", "embed.py", "hybrid.py", "feat.py", "train.py", "neural.py", "match.py"]
    if gate.get("feature_backend") in (tfeat.backend, "hybrid-v2"):
        srcs += ["tfeat.py", "tm_prep.py", "tm_rules.py"]
    if gate.get("feature_backend") == "hybrid-v2":
        srcs += ["rfeat.py", "norm2.py"]
    if any(x.get("checkpoint_sha256") for x in rs):
        srcs += ["retr.py"]
    z = {"version": ver, "data_meta_sha256": _sha(data / "meta.json"), "data_version": dm.get("version"),
         "normalization": dm.get("normalization", "data.norm"), "retriever_sources_sha256": _sha(embed.src0),
         "gate": gate, "neural": nn, "retrievers": ret,
          "feature_contract": gate["feature_names"], "block_score_version": block.sv,
           "source_files": {x: _sha(src / x) for x in srcs},
         "text_format": neural.fmt, "k": {"lexical": klex, "dense": kdense, "gate": kgate},
          "numeric": {"gate": "mean_probability", "neural": "sigmoid_probability", "topk_ties": "qid_asc",
                      "device": device, "neural_precision": precision, "retrieval_precision": "fp16" if device == "cuda" else "fp32",
                      "blend": {"method": "weighted_logit", "neural_weight": neural_weight}}}
    if neural_floor is not None:
        z["numeric"]["neural_selection"] = {"gate_floor": neural_floor, "unscored": "gate_probability"}
    z["model"] = {"sha256": hh.sha256(json.dumps(
        {"gate": gate["sha256"], "neural": nn["sha256"], "neural_weight": neural_weight}, sort_keys=True).encode()).hexdigest()}
    return z, hh.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()


def _top(d, k):
    return d.sort(["tid", "gate", "qid"], descending=[False, True, False]).group_by(
        "tid", maintain_order=True).head(k).sort("tid", "qid")


def _text(ref, q, d, fallback=False):
    x = d.select("tid", "qid").join(ref.select(pl.col("rid").alias("qid"), pl.col("nm").alias("rnm"),
        pl.col("ad").alias("rad"), pl.col("co").alias("rco")), on="qid", how="left", validate="m:1").join(
        q.select(pl.col("rid").alias("tid"), pl.col("nm").alias("tnm"), pl.col("ad").alias("tad"),
                 pl.col("co").alias("tco")), on="tid", how="left", validate="m:1")
    if x["rnm"].null_count() or x["tnm"].null_count() or (not fallback and x.filter(pl.col("rco") != pl.col("tco")).height):
        raise ValueError("unknown or cross-country neural pair")
    return pl.DataFrame({"text_a": neural.text(x.select(pl.col("rnm").alias("nm"), pl.col("rad").alias("ad"),
                                                       pl.col("rco").alias("co"))),
                         "text_b": neural.text(x.select(pl.col("tnm").alias("nm"), pl.col("tad").alias("ad"),
                                                       pl.col("tco").alias("co")))})


def _prob(x, what):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError(f"invalid {what} probabilities")
    return x


def _weight(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise ValueError("neural weight must be finite and in [0, 1]") from None
    if not np.isfinite(x) or not 0 <= x <= 1:
        raise ValueError("neural weight must be finite and in [0, 1]")
    return x


def _blend(neural_prob, gate_prob, neural_weight):
    neural_prob, gate_prob = _prob(neural_prob, "neural"), _prob(gate_prob, "gate")
    if neural_prob.shape != gate_prob.shape:
        raise ValueError("neural and gate probabilities have different shapes")
    if neural_weight == 1:
        return neural_prob
    eps = np.finfo(np.float32).eps
    neural_prob, gate_prob = (np.clip(x, eps, 1 - eps) for x in (neural_prob, gate_prob))
    logits = np.float32(neural_weight) * (np.log(neural_prob) - np.log1p(-neural_prob))
    logits += np.float32(1 - neural_weight) * (np.log(gate_prob) - np.log1p(-gate_prob))
    return (np.float32(1) / (np.float32(1) + np.exp(-logits))).astype(np.float32, copy=False)


def _empty(split, sr):
    return infer._empty(split).with_columns(pl.lit(sr, dtype=pl.UInt8).alias("sr"),
                                            pl.lit(None, dtype=pl.Float32).alias("gate_prob"),
                                            pl.lit(None, dtype=pl.Float32).alias("neural_prob"))


def _valid(d, q, refs, split, sr):
    need = {"qid", "tid", "prob", "sr"} | ({"y"} if split == "train" else set())
    if need - set(d.columns) or d.select("qid", "tid").n_unique() != len(d):
        raise ValueError("invalid matcher part")
    if len(d) and (d.filter(~pl.col("tid").is_in(q["rid"].implode())).height or
                   d.filter(~pl.col("qid").is_in(refs["rid"].implode())).height or
                   d.filter(pl.col("sr") != sr).height or not np.isfinite(d["prob"].to_numpy()).all()):
        raise ValueError("unknown or invalid matcher pair")
    if split == "train":
        y = d.select("tid", "qid", "y").join(q.select(pl.col("rid").alias("tid"), "own"), on="tid", how="left")
        if y["own"].null_count() or y.filter(pl.col("y") != (pl.col("qid").cast(pl.Int64) == pl.col("own")).cast(pl.UInt8)).height:
            raise ValueError("matcher diagnostic labels disagree")


def _old(out, x, ids, split, q, refs):
    _, got = infer._part(out, x, split)
    if not np.array_equal(got, ids):
        raise ValueError("resume part mismatch")
    d = pl.read_parquet(out / x["name"])
    if "sr" not in d.columns:
        raise ValueError("resume part has no target source")
    _valid(d, q, refs, split, x["source"])
    scored = d.filter(pl.col("neural_scored")) if "neural_scored" in d.columns else d
    if x.get("neural_pairs") != len(scored) or x.get("neural_input_sha256") != _pairs(scored):
        raise ValueError("resume neural candidate mismatch")


def match(data, cache, gate_dir, neural_dir, out, split="test", country=None, rid_start=None, rid_stop=None,
          k_lex=10, k_dense=50, k_gate=20, device="auto", threads=4, encoder_batch=64, neural_batch=32,
           query_batch=512, retrievers=None, gate_models=None, neural_bundle=None, neural_predictor=None, neural_weight=1.0,
           neural_floor=None):
    t0 = time.perf_counter()
    tm = {"setup": 0., "retrieval": 0., "features": 0., "neural": 0., "new_queries": 0}
    if split not in {"train", "test"} or min(k_lex, k_dense, k_gate, threads, encoder_batch, neural_batch, query_batch) < 1:
        raise ValueError("invalid matching options")
    if rid_start is not None and rid_stop is not None and rid_start >= rid_stop:
        raise ValueError("rid start must be below rid stop")
    data, cache, out = (path(x).resolve() for x in (data, cache, out))
    neural_weight = _weight(neural_weight)
    if neural_floor is not None and (not np.isfinite(neural_floor) or not 0 <= neural_floor <= 1):
        raise ValueError("neural floor must be in [0,1]")
    device = neural._device(device)
    gm, names, dense, gi = _gate(gate_dir)
    nm, ni = _neural(neural_dir)
    rs = _retrievers(retrievers)
    cfg, ch = _cfg(data, {**gi, "feature_names": names, "dense_features": dense, "score_version": block.sv}, ni,
                   rs, k_lex, k_dense, k_gate, neural_weight, device, neural_floor)
    scope = infer._scope(data, split, country, rid_start, rid_stop)
    mp = out / "manifest.json"
    if mp.exists():
        man = _json(mp)
        if man.get("version") != ver or man.get("split") != split or man.get("config") != cfg or man.get("scope") != scope:
            raise ValueError("existing matching manifest does not match this run")
    else:
        if (out / "parts").exists() and any((out / "parts").iterdir()):
            raise ValueError("unlisted inference artifacts exist")
        man = {"version": ver, "split": split, "config": cfg, "config_sha256": ch, "scope": scope,
               "parts": [], "provenance": {"countries": {}}}
        _write(mp, man)
    done = {x["name"]: x for x in man["parts"]}
    known = set(done) | {x.get("coverage") for x in done.values()}
    if (out / "parts").exists():
        extra = {str(p.relative_to(out)) for p in (out / "parts").rglob("*") if p.is_file()} - known
        if extra:
            raise ValueError(f"unlisted inference artifacts {sorted(extra)}")
    for x in done.values():
        infer._part(out, x, split)
    if gate_models is None:
        gate_models, loaded = train.load_models(gate_dir)
        if loaded != gm:
            raise ValueError("gate metadata changed while loading")
    if not gate_models:
        raise ValueError("no loaded gate models")
    if gm.get("feature_backend") == "hybrid-v2":
        import rfeat
        ft = rfeat
    else:
        ft = tfeat if gm.get("feature_backend") == tfeat.backend else feat
    if neural_bundle is None and neural_predictor is None:
        neural_bundle = neural.load(neural_dir, device)
    n, nq, npair, seen = 0, 0, 0, set()
    refs = pl.read_parquet(data / split / "ref.parquet", columns=["rid", "co"])
    for co in infer._countries(data, split, country):
        cr = refs.filter(pl.col("co") == co)
        state = None
        allowed = cr
        if len(refs):
            t1 = time.perf_counter()
            state = hybrid.setup(data, cache, co, split, 0, rs, device, encoder_batch)
            if state["config"]["total_params"] + ni["parameters"] > hybrid.mx:
                raise ValueError("active retrieval and neural models exceed 8b parameters")
            if set(dense) - set(state["dense_features"]):
                raise ValueError("gate needs unavailable dense features")
            man["provenance"]["countries"][co] = {"status": "ready", "retrieval": state["config"],
                                                       "references": state["config"]["reference_rows"]}
            fst = (ft.prep(state["refs"], data, split, path(gate_dir) / gm["normalizer"]["file"], cache)
                   if gm.get("feature_backend") == "hybrid-v2" else ft.prep(state["refs"]))
            allowed = state["refs"]
            tm["setup"] += time.perf_counter() - t1
        else:
            man["provenance"]["countries"][co] = {"status": "no_references"}
            fst = None
        for sr in (2, 3):
            for q in infer._queries(data / split / f"s{sr}.parquet", co, rid_start, rid_stop, query_batch):
                name, cov = f"parts/part_{n:07d}.parquet", f"parts/part_{n:07d}.npy"
                n += 1
                ids = q["rid"].to_numpy().astype(np.uint32, copy=False)
                old = done.get(name)
                if old is not None:
                    _old(out, old, ids, split, q, allowed)
                    nq, npair = nq + len(ids), npair + old["pairs"]
                    seen.add(name)
                    continue
                pp, cp = out / name, out / cov
                if pp.exists() or cp.exists():
                    raise ValueError(f"unlisted inference artifact {name}")
                retrieved_true = kept_true = neural_count = 0
                if state is None:
                    scored, before = _empty(split, sr), 0
                else:
                    t1 = time.perf_counter()
                    p = hybrid.search(state, q, k_lex, k_dense, threads)
                    tm["retrieval"] += time.perf_counter() - t1
                    before = len(p)
                    if split == "train":
                        retrieved_true = int(p["y"].sum())
                    if p.is_empty():
                        scored = _empty(split, sr)
                    else:
                        t1 = time.perf_counter()
                        x, got = ft.make(fst, q, p, threads, dense)
                        if got != names:
                            raise ValueError("gate feature contract mismatch")
                        gp = _prob(np.mean([train.predict(m, x, threads) for m in gate_models.values()], axis=0, dtype=np.float64), "gate")
                        post = _top(p.with_columns(pl.Series("gate", gp)), k_gate)
                        selected = np.ones(len(post), bool) if neural_floor is None else post["gate"].to_numpy() >= neural_floor
                        tx = _text(state["refs"], q, post.filter(pl.Series(selected)), state["fallback"])
                        neural_count = int(selected.sum())
                        if split == "train":
                            kept_true = int(post["y"].sum())
                        tm["features"] += time.perf_counter() - t1
                        t1 = time.perf_counter()
                        raw = post["gate"].to_numpy().copy()
                        if len(tx):
                            npb = neural_predictor(tx) if neural_predictor is not None else neural.predict(neural_bundle, tx, neural_batch)
                            raw[selected] = _prob(npb, "neural")
                        tm["neural"] += time.perf_counter() - t1
                        raw = _prob(raw, "neural")
                        npb = _blend(raw, post["gate"].to_numpy(), neural_weight)
                        scored = post.select("qid", "tid", "sr", *(["y"] if split == "train" else [])).with_columns(
                            pl.Series("prob", npb), pl.Series("gate_prob", post["gate"].to_numpy()),
                            pl.Series("neural_prob", raw)).select("qid", "tid", "prob", "sr", "gate_prob", "neural_prob",
                                                                  *(["y"] if split == "train" else [])).sort("tid", "qid")
                        if neural_floor is not None:
                            flags = post.select("qid", "tid").with_columns(pl.Series("neural_scored", selected))
                            scored = scored.join(flags, on=["qid", "tid"], how="left", maintain_order="left", validate="1:1")
                if neural_floor is not None and "neural_scored" not in scored.columns:
                    scored = scored.with_columns(pl.lit(False).alias("neural_scored"))
                _valid(scored, q, allowed, split, sr)
                _pq(scored, pp)
                _npy(cp, ids)
                z = {"name": name, "coverage": cov, "queries": len(ids), "pairs": len(scored),
                     "pair_sha256": _sha(pp), "coverage_sha256": _sha(cp), "country": co, "source": sr,
                      "retrieved_pairs": before, "neural_pairs": neural_count,
                      "neural_input_sha256": _pairs(scored.filter(pl.col("neural_scored"))) if neural_floor is not None else _pairs(scored)}
                if split == "train":
                    eligible = int(q.filter(pl.col("own") >= 0).height)
                    z["recall_stages"] = {"true_queries": eligible, "retrieved_true": retrieved_true, "kept_true": kept_true,
                                          "retrieval_misses": eligible - retrieved_true, "gate_pruned_true": retrieved_true - kept_true}
                man["parts"].append(z)
                _write(mp, man)
                tm["new_queries"] += len(ids)
                nq, npair = nq + len(ids), npair + len(scored)
                seen.add(name)
    if set(done) - seen:
        raise ValueError("unvisited resumed part")
    r = infer._run(data, out, split, infer._ids(data, split, country, rid_start, rid_stop))
    z = {"manifest": str(mp), "parts": len(r["parts"]), "queries": nq, "pairs": npair,
          "coverage": len(r["coverage"]), "config_sha256": ch}
    tm["total"] = time.perf_counter() - t0
    _write(out / "timing.json", tm)
    z["timing"] = tm
    print(json.dumps(z, indent=2))
    return z


def _data(root):
    d = root / "data"
    for x in ("train", "test"):
        (d / x).mkdir(parents=True)
    _write(d / "meta.json", {"version": 1, "normalization": "check"})
    ref = pl.DataFrame({"rid": [1, 2, 3, 4], "eid": ["S1-us", "S1-empty", "S1-fr", "S1-fit"],
                        "nm": ["alpha", "empty", "ecole", "fit"], "ad": ["1 main", "2 main", "3 rue", "4 road"],
                        "nn": ["alpha", "empty", "ecole", "fit"], "an": ["1 main", "2 main", "3 rue", "4 road"],
                        "co": ["us", "us", "france", "us"], "sr": [1, 1, 1, 1], "deg": [2, 2, 1, 0],
                        "uni": [0, 0, 0, 0], "blank": [0, 0, 0, 0], "fold": [0, 0, 1, 2]}).with_columns(
        pl.col("rid").cast(pl.UInt32), pl.col("sr").cast(pl.UInt8), pl.col("deg").cast(pl.UInt32), pl.col("fold").cast(pl.UInt8))
    rows = {2: [[10, "S2-us", "alpha", "1 main", "alpha", "1 main", "us", 2, 1],
                [11, "S2-fr", "ecole", "3 rue", "ecole", "3 rue", "france", 2, 3],
                [12, "S2-empty", "empty", "2 main", "empty", "2 main", "", 2, 2]],
            3: [[20, "S3-us", "alpha", "1 main", "alpha", "1 main", "us", 3, 1],
                [21, "S3-none", "none", "9 lane", "none", "9 lane", "void", 3, -1],
                [22, "S3-empty", "empty", "2 main", "empty", "2 main", "unknown", 3, 2]]}
    for x in ("train", "test"):
        ref.write_parquet(d / x / "ref.parquet")
        for sr, rs in rows.items():
            pl.DataFrame(rs, schema=["rid", "eid", "nm", "ad", "nn", "an", "co", "sr", "own"], orient="row").with_columns(
                pl.col("rid").cast(pl.UInt32), pl.col("sr").cast(pl.UInt8), pl.col("own").cast(pl.Int32)).write_parquet(d / x / f"s{sr}.parquet")
    return d


def check():
    class gate:
        def __init__(self, constant=None):
            self.constant = constant

        def predict_proba(self, x):
            p = np.clip(x[:, feat.ff.index("ns")], 0, 1) if self.constant is None else np.full(len(x), self.constant)
            return np.column_stack([1 - p, p])

    class enc:
        def get_embedding_dimension(self):
            return 2

        def encode(self, xs, **kw):
            return np.asarray([[1, 0] if "alpha" in x else [0, 1] for x in xs], np.float32)

    with tf.TemporaryDirectory() as tmp:
        root = path(tmp)
        data = _data(root)
        gd, nd = root / "gate", root / "neural"
        gd.mkdir(); nd.mkdir()
        (gd / "fake.bin").write_bytes(b"gate")
        _write(gd / "metadata.json", {"score_version": block.sv, "feature_names": feat.ff + ["ds_e5"],
                                         "dense_features": ["ds_e5"], "models": ["fake"], "model_files": {"fake": "fake.bin"}})
        (nd / "model.safetensors").write_bytes(b"neural")
        _write(nd / "neural_metadata.json", {"parameters": 2, "problem_type": "multi_label_classification",
                                                "source": {"license": "mit"}, "configuration": {"maxlen": 8}})
        ident = _neural(nd)[1]
        (nd / "checkpoint-1").mkdir()
        (nd / "checkpoint-1/model.safetensors").write_bytes(b"training checkpoint")
        assert _neural(nd)[1] == ident
        _write(nd / "tokenizer_config.json", {"padding_side": "left"})
        assert _neural(nd)[1]["sha256"] != ident["sha256"]
        import sys
        from unittest.mock import patch
        argv = ["match.py", "--data", str(data), "--cache", str(root / "cache"), "--gate", str(gd),
                "--neural", str(nd), "--out", str(root / "cli"), "--retrievers", "e5", "qwen3"]
        with patch.object(sys, "argv", argv), patch(__name__ + ".match") as cli:
            main()
            assert cli.call_count == 1
        calls = []

        def pred(x):
            calls.extend(x.iter_rows())
            return np.full(len(x), .9, np.float32)

        nn, gp = np.array([.8], np.float32), np.array([.2], np.float32)
        manual = 1 / (1 + np.exp(-(.6 * (np.log(nn) - np.log1p(-nn)) + .4 * (np.log(gp) - np.log1p(-gp)))))
        np.testing.assert_allclose(_blend(nn, gp, .6), manual.astype(np.float32), rtol=1e-6)
        assert _blend(nn, gp, 1).tobytes() == nn.tobytes()

        rs = [{"model": embed.mod0, "revision": embed.rev0, "encoder": enc(), "params": 2}]
        allref = pl.read_parquet(data / "test/ref.parquet")
        normal = hybrid.setup(data, root / "normal", "france", "test", models=rs, device="cpu", batch=2)
        fallback = hybrid.setup(data, root / "fallback", "", "test", models=rs, device="cpu", batch=2)
        assert not normal["fallback"] and normal["refs"]["rid"].to_list() == [3]
        assert fallback["fallback"] and fallback["refs"]["rid"].to_list() == [1, 2, 3, 4]
        assert fallback["encoders"][0]["core"]["scope"] == "global"
        assert any("country: us" in x for x in embed.serial(fallback["refs"], "passage", embed.mod0))
        _, fit, _ = block.setup(data, root / "lex", "unknown", "train")
        fold2, _, _ = block.setup(data, root / "lex", "unknown", "train", fold=2)
        assert fit["nn"][0].transform(["ecole"]).nnz == 0 and fold2["rid"].to_list() == [4]
        q_us = pl.read_parquet(data / "test/s2.parquet").filter(pl.col("rid") == 10)
        cross = pl.DataFrame({"tid": [10], "qid": [3]})
        try:
            _text(allref, q_us, cross)
        except ValueError:
            pass
        else:
            raise AssertionError("normal cross-country candidate accepted")
        assert len(_text(allref, q_us, cross, True)) == 1
        tr, te = root / "run-tr", root / "run-te"
        match(data, root / "cache", gd, nd, tr, "train", k_lex=2, k_dense=2, k_gate=1, device="cpu", threads=1,
              encoder_batch=2, neural_batch=2, query_batch=1, retrievers=rs, gate_models={"fake": gate()}, neural_predictor=pred)
        train_calls = len(calls)
        match(data, root / "cache", gd, nd, te, "test", k_lex=2, k_dense=2, k_gate=1, device="cpu", threads=1,
              encoder_batch=2, neural_batch=2, query_batch=1, retrievers=rs, gate_models={"fake": gate()}, neural_predictor=pred)
        ps = [pl.read_parquet(te / x["name"]) for x in _json(te / "manifest.json")["parts"]]
        outp = pl.concat(ps)
        assert len(calls) == train_calls + len(outp) and set(outp["tid"]) >= {20} and set(outp["sr"]) == {2, 3} and all(
            x.sort("tid", "qid").equals(x) for x in ps)
        assert {(11, 3), (12, 2), (22, 2)} <= set(outp.select("tid", "qid").iter_rows())
        man = _json(te / "manifest.json")
        assert any(x["retrieved_pairs"] > x["neural_pairs"] for x in man["parts"]) and all(x["pairs"] == x["neural_pairs"] for x in man["parts"])
        assert np.array_equal(infer._run(data, te, "test")["coverage"], infer._ids(data, "test"))
        cal = infer.calibrate(data, [tr], root / "cal.json")
        ex = infer.export(data, [te], root / "cal.json", root / "out")
        assert "S1-empty\t" in (root / "out/matching_results.tsv").read_text() and ex["source1"] == 4
        old_calls = len(calls)
        for _ in range(2):
            match(data, root / "cache", gd, nd, root / "selective", "train", k_lex=2, k_dense=2, k_gate=1,
                  device="cpu", threads=1, encoder_batch=2, neural_batch=2, query_batch=1, retrievers=rs,
                  gate_models={"fake": gate(.1)}, neural_predictor=pred, neural_floor=.5)
        selective = _json(root / "selective/manifest.json")
        assert len(calls) == old_calls and sum(p["pairs"] for p in selective["parts"]) > 0
        for part in selective["parts"]:
            assert part["neural_pairs"] == 0
            stages = part["recall_stages"]
            assert stages["true_queries"] == stages["retrieved_true"] + stages["retrieval_misses"]
            assert stages["retrieved_true"] == stages["kept_true"] + stages["gate_pruned_true"]
        import norm2
        import retr
        import rfeat
        rich_data = infer._check_data(root / "rich-data")
        for split in ("train", "test"):
            for sr in (2, 3):
                p = rich_data / split / f"s{sr}.parquet"
                d = pl.read_parquet(p).with_columns(pl.Series("rid", np.arange((sr - 2) * 2, (sr - 1) * 2, dtype=np.uint32)))
                d.write_parquet(p)
        rg = root / "rich-gate"
        rg.mkdir()
        norm2.fit(rich_data, rg / "normalizer.json")
        rr = pl.read_parquet(rich_data / "train/ref.parquet")
        qq = pl.read_parquet(rich_data / "train/s2.parquet")
        state = rfeat.prep(rr, rich_data, "train", rg / "normalizer.json", root / "rich-cache")
        pp = pl.DataFrame({"qid": [0, 1], "tid": [0, 0], "ns": [1., 0.], "ads": [1., 0.], "ds_e5_small": [1., 0.]})
        x, fs = rfeat.make(state, qq, pp, dense=["ds_e5_small"])
        tx, ty = np.tile(x, (60, 1)), np.tile(np.array([1, 0]), 60)
        model = train._fit_lgb(tx, ty, tx.copy(), ty.copy(), 5, 1, fs)
        model.booster_.save_model(rg / "lgb.txt")
        _write(rg / "metadata.json", {"feature_backend": rfeat.backend, "feature_names": fs,
               "dense_features": ["ds_e5_small"], "score_version": block.sv, "models": ["lgb"], "model_files": {"lgb": "lgb.txt"},
               "normalizer": {"file": "normalizer.json", "sha256": _sha(rg / "normalizer.json")}})
        small = [{"model": retr.model_id, "revision": retr.revision, "encoder": enc(), "params": 2}]
        result = match(rich_data, root / "rich-cache", rg, nd, root / "rich-match", "train", k_lex=2, k_dense=2,
                       k_gate=1, device="cpu", threads=1, encoder_batch=2, neural_batch=2, query_batch=1,
                       retrievers=small, neural_predictor=pred)
        assert result["queries"] == 4 and result["pairs"] > 0
        try:
            match(data, root / "cache", gd, nd, te, "test", k_lex=2, k_dense=2, k_gate=1, device="cpu", threads=1,
                  retrievers=rs, gate_models={"fake": gate()}, neural_predictor=pred, neural_weight=.6)
        except ValueError:
            pass
        else:
            raise AssertionError("weight mismatch resume accepted")
        raw = root / "raw"; raw.mkdir()
        for n, rows in (("test_source1.tsv", ["S1-us", "S1-empty", "S1-fr", "S1-fit"]), ("test_source2.tsv", ["S2-us", "S2-fr", "S2-empty"]),
                        ("test_source3.tsv", ["S3-us", "S3-none", "S3-empty"])):
            (raw / n).write_text("entity_id\n" + "\n".join(rows) + "\n", encoding="utf-8")
        v = path(__file__).resolve().parents[1] / "student_resource/utils/validate_submission.py"
        assert sp.run([os.sys.executable, v, "--matching", root / "out/matching_results.tsv", "--candidate",
                       root / "out/candidate_pairs.tsv", "--test-dir", raw, "--check-ids"], capture_output=True, text=True).returncode == 0
        ids = infer._ids(data, "test")
        for bad, msg in ((ids[:-1], "extra coverage accepted"), (np.r_[ids, np.uint32(99)], "missing coverage accepted")):
            try:
                infer._run(data, te, "test", bad)
            except ValueError:
                pass
            else:
                raise AssertionError(msg)
        old = _json(te / "manifest.json")
        old["parts"].append(dict(old["parts"][0], name="parts/part_9999999.parquet", coverage="parts/part_9999999.npy"))
        (te / "parts/part_9999999.parquet").write_bytes((te / old["parts"][0]["name"]).read_bytes())
        (te / "parts/part_9999999.npy").write_bytes((te / old["parts"][0]["coverage"]).read_bytes())
        old["parts"][-1]["pair_sha256"] = _sha(te / old["parts"][-1]["name"])
        old["parts"][-1]["coverage_sha256"] = _sha(te / old["parts"][-1]["coverage"])
        _write(te / "manifest.json", old)
        try:
            infer._run(data, te, "test", infer._ids(data, "test"))
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate coverage accepted")
        _write(te / "manifest.json", man)
        (te / "parts/part_9999999.parquet").unlink(); (te / "parts/part_9999999.npy").unlink()
        (gd / "fake.bin").write_bytes(b"changed")
        try:
            match(data, root / "cache", gd, nd, te, "test", k_lex=2, k_dense=2, k_gate=1, device="cpu", threads=1,
                  retrievers=rs, gate_models={"fake": gate()}, neural_predictor=pred)
        except ValueError:
            pass
        else:
            raise AssertionError("model mismatch resume accepted")
        from unittest.mock import patch
        with patch.object(hybrid, "setup", side_effect=RuntimeError("broken index")):
            try:
                match(data, root / "cache", gd, nd, root / "broken", "test", country="us", k_lex=2, k_dense=2,
                      k_gate=1, device="cpu", threads=1, retrievers=rs, gate_models={"fake": gate()}, neural_predictor=pred)
            except RuntimeError as e:
                assert str(e) == "broken index"
            else:
                raise AssertionError("index failure was hidden")
    print("checks passed")


def main():
    root = path(__file__).resolve().parents[1]
    p = ap.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--data", type=path, default=root / "cache/data")
    p.add_argument("--cache", type=path, default=root / "cache")
    p.add_argument("--gate", type=path)
    p.add_argument("--neural", type=path)
    p.add_argument("--out", type=path)
    p.add_argument("--split", choices=("train", "test"), default="test")
    p.add_argument("--country")
    p.add_argument("--rid-start", type=int); p.add_argument("--rid-stop", type=int)
    p.add_argument("--k-lex", type=int, default=10); p.add_argument("--k-dense", type=int, default=50); p.add_argument("--k-gate", type=int, default=20)
    p.add_argument("--retrievers", choices=("e5", "qwen3", "e5-large", "e5-small"), nargs="+", default=["e5"])
    p.add_argument("--retrievers-file", type=path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    p.add_argument("--encoder-batch", type=int, default=64); p.add_argument("--neural-batch", type=int, default=32); p.add_argument("--query-batch", type=int, default=512)
    p.add_argument("--neural-weight", type=float, default=1.0)
    p.add_argument("--neural-floor", type=float)
    a = p.parse_args()
    if a.check:
        check()
    elif a.gate and a.neural and a.out:
        retrievers = hybrid.configs(a.retrievers, a.retrievers_file)
        match(a.data, a.cache, a.gate, a.neural, a.out, a.split, a.country, a.rid_start, a.rid_stop, a.k_lex, a.k_dense,
              a.k_gate, a.device, a.threads, a.encoder_batch, a.neural_batch, a.query_batch, retrievers,
              neural_weight=a.neural_weight, neural_floor=a.neural_floor)
    else:
        p.error("use --check or --gate --neural --out")


if __name__ == "__main__":
    main()
