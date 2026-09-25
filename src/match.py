import argparse as ap
import hashlib as hh
import json
import os
import subprocess as sp
import tempfile as tf
from pathlib import Path as path

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
except ImportError:
    from src import block, embed, feat, hybrid, infer, neural, train


ver = 1


def _sha(p):
    return infer._sha(p)


def _json(p):
    return infer._json(p)


def _write(p, z):
    infer._write(p, z)


def _pq(d, p):
    infer._pq(d, p)


def _npy(p, x):
    infer._npy(p, x)


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
    if names != list(feat.ff) + dense:
        raise ValueError("invalid gate feature contract")
    fs = m.get("model_files", {})
    if not isinstance(m.get("models"), list) or not m["models"]:
        raise ValueError("gate metadata has no models")
    if any(not isinstance(fs.get(n), str) or path(fs[n]).name != fs[n] for n in m["models"]):
        raise ValueError("invalid gate model file name")
    f = _files(d, [fs[n] for n in m["models"]])
    z = {"metadata_sha256": _sha(d / "metadata.json"), "files": f}
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
    ws = sorted(p.relative_to(d).as_posix() for p in d.rglob("*.safetensors"))
    ws += sorted(p.relative_to(d).as_posix() for p in d.rglob("pytorch_model*.bin"))
    if not ws:
        raise ValueError("neural model has no weights")
    z = {"metadata_sha256": _sha(d / "neural_metadata.json"), "weights": _files(d, ws), "parameters": n}
    z["sha256"] = hh.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()
    return m, z


def _retrievers(xs):
    if xs is None:
        allx = {embed.family(x["model"]): x for x in hybrid.specs()}
        xs = [allx["e5"]]
    else:
        xs = hybrid.specs(xs)
    fs = [embed.family(x["model"]) for x in xs]
    if fs not in (["e5"], ["e5", "qwen3"]):
        raise ValueError("retrievers must be e5 or e5 plus qwen3")
    return xs


def _cfg(data, gate, nn, rs, klex, kdense, kgate):
    dm = _json(data / "meta.json")
    ret = []
    for x in rs:
        f = embed.family(x["model"])
        ret.append({"model": x["model"], "revision": x["revision"], "feature": hybrid.feat(x),
                    "format": embed.fmt if f == "e5" else embed.rawfmt, "family": f})
    z = {"version": ver, "data_meta_sha256": _sha(data / "meta.json"), "data_version": dm.get("version"),
         "normalization": dm.get("normalization", "data.norm"), "retriever_sources_sha256": _sha(embed.src0),
         "gate": gate, "neural": nn, "retrievers": ret,
         "feature_contract": gate["feature_names"], "block_score_version": block.sv,
         "text_format": neural.fmt, "k": {"lexical": klex, "dense": kdense, "gate": kgate},
         "numeric": {"gate": "mean_probability", "neural": "sigmoid_probability", "topk_ties": "qid_asc"}}
    z["model"] = {"sha256": hh.sha256(json.dumps({"gate": gate["sha256"], "neural": nn["sha256"]}, sort_keys=True).encode()).hexdigest()}
    return z, hh.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()


def _top(d, k):
    return d.sort(["tid", "gate", "qid"], descending=[False, True, False]).group_by(
        "tid", maintain_order=True).head(k).sort("tid", "qid")


def _text(ref, q, d):
    x = d.select("tid", "qid").join(ref.select(pl.col("rid").alias("qid"), pl.col("nm").alias("rnm"),
        pl.col("ad").alias("rad"), pl.col("co").alias("rco")), on="qid", how="left", validate="m:1").join(
        q.select(pl.col("rid").alias("tid"), pl.col("nm").alias("tnm"), pl.col("ad").alias("tad"),
                 pl.col("co").alias("tco")), on="tid", how="left", validate="m:1")
    if x["rnm"].null_count() or x["tnm"].null_count() or x.filter(pl.col("rco") != pl.col("tco")).height:
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


def _empty(split, sr):
    return infer._empty(split).with_columns(pl.lit(sr, dtype=pl.UInt8).alias("sr"))


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
    if not np.array_equal(got, ids) or x.get("neural_pairs") != x.get("pairs"):
        raise ValueError("resume part mismatch")
    d = pl.read_parquet(out / x["name"])
    if "sr" not in d.columns:
        raise ValueError("resume part has no target source")
    _valid(d, q, refs, split, x["source"])
    if x.get("neural_input_sha256") != _pairs(d):
        raise ValueError("resume neural candidate mismatch")


def match(data, cache, gate_dir, neural_dir, out, split="test", country=None, rid_start=None, rid_stop=None,
          k_lex=10, k_dense=50, k_gate=20, device="auto", threads=4, encoder_batch=64, neural_batch=32,
          query_batch=512, retrievers=None, gate_models=None, neural_bundle=None, neural_predictor=None):
    if split not in {"train", "test"} or min(k_lex, k_dense, k_gate, threads, encoder_batch, neural_batch, query_batch) < 1:
        raise ValueError("invalid matching options")
    if rid_start is not None and rid_stop is not None and rid_start >= rid_stop:
        raise ValueError("rid start must be below rid stop")
    data, cache, out = (path(x).resolve() for x in (data, cache, out))
    gm, names, dense, gi = _gate(gate_dir)
    nm, ni = _neural(neural_dir)
    rs = _retrievers(retrievers)
    cfg, ch = _cfg(data, {**gi, "feature_names": names, "dense_features": dense, "score_version": block.sv}, ni,
                   rs, k_lex, k_dense, k_gate)
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
    if neural_bundle is None and neural_predictor is None:
        neural_bundle = neural.load(neural_dir, device)
    n, nq, npair, seen = 0, 0, 0, set()
    refs = pl.read_parquet(data / split / "ref.parquet", columns=["rid", "co"])
    for co in infer._countries(data, split, country):
        cr = refs.filter(pl.col("co") == co)
        state = None
        if len(cr):
            state = hybrid.setup(data, cache, co, split, 0, rs, "cpu" if device == "auto" else device, encoder_batch)
            if state["config"]["total_params"] + ni["parameters"] > hybrid.mx:
                raise ValueError("active retrieval and neural models exceed 8b parameters")
            if set(dense) - set(state["dense_features"]):
                raise ValueError("gate needs unavailable dense features")
            man["provenance"]["countries"][co] = {"status": "ready", "retrieval": state["config"],
                                                      "references": state["config"]["reference_rows"]}
            fst = feat.prep(state["refs"])
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
                    _old(out, old, ids, split, q, cr)
                    nq, npair = nq + len(ids), npair + old["pairs"]
                    seen.add(name)
                    continue
                pp, cp = out / name, out / cov
                if pp.exists() or cp.exists():
                    raise ValueError(f"unlisted inference artifact {name}")
                if state is None:
                    scored, before = _empty(split, sr), 0
                else:
                    p = hybrid.search(state, q, k_lex, k_dense, threads)
                    before = len(p)
                    if p.is_empty():
                        scored = _empty(split, sr)
                    else:
                        x, got = feat.make(fst, q, p, threads, dense)
                        if got != names:
                            raise ValueError("gate feature contract mismatch")
                        gp = _prob(np.mean([train.predict(m, x) for m in gate_models.values()], axis=0, dtype=np.float64), "gate")
                        post = _top(p.with_columns(pl.Series("gate", gp)), k_gate)
                        tx = _text(state["refs"], q, post)
                        npb = neural_predictor(tx) if neural_predictor is not None else neural.predict(neural_bundle, tx, neural_batch)
                        npb = _prob(npb, "neural")
                        scored = post.select("qid", "tid", "sr", *(["y"] if split == "train" else [])).with_columns(
                            pl.Series("prob", npb)).select("qid", "tid", "prob", "sr", *(["y"] if split == "train" else [])).sort("tid", "qid")
                _valid(scored, q, cr, split, sr)
                _pq(scored, pp)
                _npy(cp, ids)
                z = {"name": name, "coverage": cov, "queries": len(ids), "pairs": len(scored),
                     "pair_sha256": _sha(pp), "coverage_sha256": _sha(cp), "country": co, "source": sr,
                     "retrieved_pairs": before, "neural_pairs": len(scored), "neural_input_sha256": _pairs(scored)}
                man["parts"].append(z)
                _write(mp, man)
                nq, npair = nq + len(ids), npair + len(scored)
                seen.add(name)
    if set(done) - seen:
        raise ValueError("unvisited resumed part")
    r = infer._run(data, out, split, infer._ids(data, split, country, rid_start, rid_stop))
    z = {"manifest": str(mp), "parts": len(r["parts"]), "queries": nq, "pairs": npair,
         "coverage": len(r["coverage"]), "config_sha256": ch}
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
                        "co": ["us", "us", "france", "us"], "sr": [1, 1, 1, 1], "deg": [2, 0, 1, 0],
                        "uni": [0, 0, 0, 0], "blank": [0, 0, 0, 0], "fold": [0, 0, 1, 2]}).with_columns(
        pl.col("rid").cast(pl.UInt32), pl.col("sr").cast(pl.UInt8), pl.col("deg").cast(pl.UInt32), pl.col("fold").cast(pl.UInt8))
    rows = {2: [[10, "S2-us", "alpha", "1 main", "alpha", "1 main", "us", 2, 1],
                [11, "S2-fr", "ecole", "3 rue", "ecole", "3 rue", "france", 2, 3]],
            3: [[20, "S3-us", "alpha", "1 main", "alpha", "1 main", "us", 3, 1],
                [21, "S3-none", "none", "9 lane", "none", "9 lane", "void", 3, -1]]}
    for x in ("train", "test"):
        ref.write_parquet(d / x / "ref.parquet")
        for sr, rs in rows.items():
            pl.DataFrame(rs, schema=["rid", "eid", "nm", "ad", "nn", "an", "co", "sr", "own"], orient="row").with_columns(
                pl.col("rid").cast(pl.UInt32), pl.col("sr").cast(pl.UInt8), pl.col("own").cast(pl.Int32)).write_parquet(d / x / f"s{sr}.parquet")
    return d


def check():
    class gate:
        def predict_proba(self, x):
            p = np.clip(x[:, feat.ff.index("ns")], 0, 1)
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
        calls = []

        def pred(x):
            calls.extend(x.iter_rows())
            return np.full(len(x), .9, np.float32)

        rs = [{"model": embed.mod0, "revision": embed.rev0, "encoder": enc(), "params": 2}]
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
        man = _json(te / "manifest.json")
        assert any(x["retrieved_pairs"] > x["neural_pairs"] for x in man["parts"]) and all(x["pairs"] == x["neural_pairs"] for x in man["parts"])
        cal = infer.calibrate(data, [tr], root / "cal.json")
        ex = infer.export(data, [te], root / "cal.json", root / "out")
        assert "S1-empty\t" in (root / "out/matching_results.tsv").read_text() and ex["source1"] == 4
        raw = root / "raw"; raw.mkdir()
        for n, rows in (("test_source1.tsv", ["S1-us", "S1-empty", "S1-fr", "S1-fit"]), ("test_source2.tsv", ["S2-us", "S2-fr"]),
                        ("test_source3.tsv", ["S3-us", "S3-none"])):
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
    p.add_argument("--retrievers", choices=("e5", "qwen3"), nargs="+", default=["e5"])
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    p.add_argument("--encoder-batch", type=int, default=64); p.add_argument("--neural-batch", type=int, default=32); p.add_argument("--query-batch", type=int, default=512)
    a = p.parse_args()
    if a.check:
        check()
    elif a.gate and a.neural and a.out:
        allx = {embed.family(x["model"]): x for x in hybrid.specs()}
        match(a.data, a.cache, a.gate, a.neural, a.out, a.split, a.country, a.rid_start, a.rid_stop, a.k_lex, a.k_dense,
              a.k_gate, a.device, a.threads, a.encoder_batch, a.neural_batch, a.query_batch, [allx[x] for x in a.retrievers])
    else:
        p.error("use --check or --gate --neural --out")


if __name__ == "__main__":
    main()
