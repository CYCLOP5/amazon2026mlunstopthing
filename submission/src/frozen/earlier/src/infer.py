import argparse as ap
import csv
import hashlib as hh
import json
import os
import tempfile as tf
from pathlib import Path as path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

try:
    import block
    import feat
    import train
except ImportError:
    from src import block, feat, train


ver = 1


def _sha(p):
    h = hh.sha256()
    with path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _json(p):
    try:
        return json.loads(path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"invalid json {p}") from e


def _write(p, z):
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(z, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    t.replace(p)


def _pq(d, p):
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    d.write_parquet(t, compression="zstd")
    t.replace(p)


def _npy(p, x):
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    with t.open("wb") as f:
        np.save(f, x.astype(np.uint32, copy=False), allow_pickle=False)
    t.replace(p)


def _ids(data, sp, co=None, lo=None, hi=None):
    fs = []
    for sr in (2, 3):
        q = pl.scan_parquet(data / sp / f"s{sr}.parquet")
        if co is not None:
            q = q.filter(pl.col("co") == co)
        if lo is not None:
            q = q.filter(pl.col("rid") >= lo)
        if hi is not None:
            q = q.filter(pl.col("rid") < hi)
        fs.append(q.select("rid").collect(engine="streaming")["rid"].to_numpy().astype(np.uint32, copy=False))
    x = np.concatenate(fs) if fs else np.empty(0, np.uint32)
    if len(x) != len(np.unique(x)):
        raise ValueError(f"duplicate {sp} target rid")
    return np.sort(x)


def _countries(data, sp, only):
    q = pl.concat([pl.scan_parquet(data / sp / f"s{sr}.parquet").select("co") for sr in (2, 3)])
    z = q.unique().sort("co").collect(engine="streaming")["co"].to_list()
    if only is not None:
        z = [only] if only in z else []
    return z


def _scope(data, sp, co, lo, hi):
    x = _ids(data, sp, co, lo, hi)
    return {"split": sp, "country": co, "rid_start": lo, "rid_stop": hi,
            "queries": len(x), "query_sha256": hh.sha256(x.tobytes()).hexdigest()}


def _model(models):
    p = path(models)
    m = _json(p / "metadata.json")
    if not isinstance(m.get("models"), list) or not m["models"]:
        raise ValueError("model metadata has no models")
    fs = m.get("model_files", {})
    z = {"metadata": _sha(p / "metadata.json"), "files": {}}
    for n in m["models"]:
        if not isinstance(fs.get(n), str) or path(fs[n]).name != fs[n]:
            raise ValueError("invalid model file name")
        f = p / fs[n]
        if not f.is_file():
            raise ValueError(f"missing model file {f}")
        z["files"][n] = _sha(f)
    z["sha256"] = hh.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()
    return m, z


def _cfg(data, models, k):
    m, mi = _model(models)
    if m.get("feature_names") != feat.ff:
        raise ValueError("model feature order mismatch")
    if m.get("score_version") != block.sv:
        raise ValueError("model candidate score version mismatch")
    dm = _json(data / "meta.json")
    z = {"version": ver, "data_meta_sha256": _sha(data / "meta.json"), "source_files": dm.get("files", {}), "model": mi,
         "features": feat.ff, "block_score_version": block.sv, "k": k}
    return z, hh.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()


def _queries(p, co, lo, hi, batch):
    f = pq.ParquetFile(p)
    cs = [x for x in ("rid", "eid", "nm", "ad", "nn", "an", "co", "sr", "own") if x in f.schema.names]
    lf = pl.scan_parquet(p).select(cs).filter(pl.col("co") == co)
    if lo is not None:
        lf = lf.filter(pl.col("rid") >= lo)
    if hi is not None:
        lf = lf.filter(pl.col("rid") < hi)
    for d in lf.collect_batches(chunk_size=batch, maintain_order=True):
        if not d.is_empty():
            yield d.sort("rid")


def _empty(sp):
    s = {"qid": pl.Series([], dtype=pl.UInt32), "tid": pl.Series([], dtype=pl.UInt32),
         "prob": pl.Series([], dtype=pl.Float32)}
    if sp == "train":
        s["y"] = pl.Series([], dtype=pl.UInt8)
    return pl.DataFrame(s)


def _part(run, x, sp):
    n, v = path(x["name"]), path(x.get("coverage", ""))
    if (n.is_absolute() or v.is_absolute() or ".." in n.parts or ".." in v.parts or
            n.parent != path("parts") or v.parent != path("parts")):
        raise ValueError("unsafe part path")
    p = run / n
    c = run / v
    if p.suffix != ".parquet" or c.suffix != ".npy" or not p.is_file() or not c.is_file():
        raise ValueError(f"missing listed part {x.get('name')}")
    if _sha(p) != x.get("pair_sha256") or _sha(c) != x.get("coverage_sha256"):
        raise ValueError(f"changed listed part {p}")
    q = np.load(c, allow_pickle=False)
    if q.dtype != np.uint32 or q.ndim != 1 or len(q) != len(np.unique(q)):
        raise ValueError(f"invalid coverage {c}")
    if len(q) != x.get("queries"):
        raise ValueError(f"coverage count mismatch {c}")
    d = pl.read_parquet(p)
    need = {"qid", "tid", "prob"} | ({"y"} if sp == "train" else set())
    if not need.issubset(d.columns) or len(d) != x.get("pairs"):
        raise ValueError(f"invalid pair part {p}")
    if d.select("qid", "tid").n_unique() != len(d):
        raise ValueError(f"duplicate scored pair {p}")
    if len(d) and d.filter(~pl.col("tid").is_in(q)).height:
        raise ValueError(f"pair outside coverage {p}")
    return p, q


def _run(data, run, sp, expected=None):
    run = path(run).resolve()
    m = _json(run / "manifest.json")
    if m.get("version") != ver or m.get("split") != sp or not isinstance(m.get("parts"), list):
        raise ValueError(f"invalid inference manifest {run}")
    if m.get("config_sha256") != hh.sha256(json.dumps(m.get("config"), sort_keys=True).encode()).hexdigest():
        raise ValueError(f"invalid configuration hash {run}")
    ps, xs = [], []
    for x in m["parts"]:
        n = path(x.get("name", "")) if isinstance(x, dict) else path()
        if not isinstance(x, dict) or n.is_absolute() or ".." in n.parts or n.suffix != ".parquet":
            raise ValueError(f"invalid part name {run}")
        p, q = _part(run, x, sp)
        ps.append(p)
        xs.append(q)
    if len(set(ps)) != len(ps):
        raise ValueError(f"repeated listed part {run}")
    parts = run / "parts"
    listed = {x["name"] for x in m["parts"]} | {x["coverage"] for x in m["parts"]}
    actual = {str(p.relative_to(run)) for p in parts.rglob("*") if p.is_file()} if parts.is_dir() else set()
    if actual != listed:
        raise ValueError(f"unlisted inference artifacts {run}")
    got = np.sort(np.concatenate(xs)) if xs else np.empty(0, np.uint32)
    if len(got) != len(np.unique(got)):
        raise ValueError(f"overlapping target coverage {run}")
    if expected is not None and not np.array_equal(got, expected):
        raise ValueError(f"incomplete target coverage {run}")
    return {"dir": run, "manifest": m, "parts": ps, "coverage": got}


def _runs(data, runs, sp, expected):
    if not runs:
        raise ValueError("at least one inference manifest is required")
    rs = [_run(data, x, sp) for x in runs]
    cs = {r["manifest"]["config_sha256"] for r in rs}
    if len(cs) != 1:
        raise ValueError("inference manifests have different configurations")
    if any(r["manifest"]["config"].get("data_meta_sha256") != _sha(data / "meta.json") for r in rs):
        raise ValueError("inference manifests use stale prepared data")
    got = np.sort(np.concatenate([r["coverage"] for r in rs]))
    if len(got) != len(np.unique(got)) or not np.array_equal(got, expected):
        raise ValueError("inference manifests do not cover the target pool exactly once")
    return rs


def _lf(rs, cs=("qid", "tid", "prob", "y")):
    return pl.concat([pl.scan_parquet(p).select([x for x in cs if x in pq.ParquetFile(p).schema.names])
                      for r in rs for p in r["parts"]])


def infer(data, models, out, cache, sp="test", k=10, threads=4, batch=4096,
          country=None, rid_start=None, rid_stop=None, ms=None, mm=None):
    if sp not in {"train", "test"} or k < 1 or threads < 1 or batch < 1:
        raise ValueError("invalid inference options")
    data, out, cache, models = (path(x).resolve() for x in (data, out, cache, models))
    if rid_start is not None and rid_stop is not None and rid_start >= rid_stop:
        raise ValueError("rid start must be below rid stop")
    cfg, ch = _cfg(data, models, k)
    sc = _scope(data, sp, country, rid_start, rid_stop)
    mp = out / "manifest.json"
    if mp.exists():
        man = _json(mp)
        if man.get("config") != cfg or man.get("scope") != sc or man.get("split") != sp:
            raise ValueError("existing inference manifest does not match this run")
    else:
        if (out / "parts").exists() and any((out / "parts").iterdir()):
            raise ValueError("unlisted inference artifacts exist")
        man = {"version": ver, "split": sp, "config": cfg, "config_sha256": ch, "scope": sc, "parts": []}
        _write(mp, man)
    done = {x["name"]: x for x in man["parts"]}
    known = set(done) | {x.get("coverage") for x in done.values()}
    if (out / "parts").exists():
        extra = {str(p.relative_to(out)) for p in (out / "parts").iterdir() if p.is_file()} - known
        if extra:
            raise ValueError(f"unlisted inference artifacts {sorted(extra)}")
    for x in done.values():
        _part(out, x, sp)
    if ms is None:
        ms, mm = train.load_models(models)
    if not ms:
        raise ValueError("no loaded models")
    n, nq, npair, seen = 0, 0, 0, set()
    cos = set(pl.read_parquet(data / sp / "ref.parquet", columns=["co"])["co"])
    for co in _countries(data, sp, country):
        if co in cos:
            ref, ix, eq = block.setup(data, cache, co, sp=sp, fold=0)
            st = feat.prep(ref)
        else:
            ref = ix = eq = st = None
        for sr in (2, 3):
            for q in _queries(data / sp / f"s{sr}.parquet", co, rid_start, rid_stop, batch):
                name = f"parts/part_{n:07d}.parquet"
                cov = f"parts/part_{n:07d}.npy"
                n += 1
                ids = q["rid"].to_numpy().astype(np.uint32, copy=False)
                old = done.get(name)
                if old is not None:
                    _, got = _part(out, old, sp)
                    if not np.array_equal(got, ids):
                        raise ValueError(f"resume batch changed {name}")
                    nq += len(ids)
                    npair += old["pairs"]
                    seen.add(name)
                    continue
                pp, cp = out / name, out / cov
                if pp.exists() or cp.exists():
                    raise ValueError(f"unlisted inference artifact {name}")
                if ref is None:
                    d = _empty(sp)
                else:
                    p = block.search(q, ix, eq, k=k, threads=threads)
                    if p.is_empty():
                        d = _empty(sp)
                    else:
                        x, names = feat.make(st, q, p, threads=threads)
                        if names != feat.ff:
                            raise ValueError("unexpected feature names")
                        pr = np.mean([train.predict(m, x) for m in ms.values()], axis=0, dtype=np.float64)
                        d = p.select("qid", "tid", *( ["y"] if sp == "train" else [])).with_columns(
                            pl.Series("prob", pr.astype(np.float32))).select("qid", "tid", "prob", *( ["y"] if sp == "train" else [])).sort("tid", "qid")
                _pq(d, pp)
                _npy(cp, ids)
                x = {"name": name, "coverage": cov, "queries": len(ids), "pairs": len(d),
                     "pair_sha256": _sha(pp), "coverage_sha256": _sha(cp), "country": co, "source": sr}
                man["parts"].append(x)
                _write(mp, man)
                nq += len(ids)
                npair += len(d)
                seen.add(name)
    if set(done) - seen:
        raise ValueError("unvisited resumed part")
    r = _run(data, out, sp, _ids(data, sp, country, rid_start, rid_stop))
    z = {"manifest": str(mp), "parts": len(r["parts"]), "queries": nq, "pairs": npair,
         "coverage": len(r["coverage"]), "config_sha256": ch}
    print(json.dumps(z, indent=2))
    return z


def _groups(lf, anchors, fold, top):
    if top:
        lf = lf.sort(["tid", "prob", "qid"], descending=[False, True, False]).group_by("tid", maintain_order=True).first()
    a = anchors.filter(pl.col("fold") == fold).select(pl.col("rid").alias("qid"), "deg")
    return lf.join(a.lazy(), on="qid", how="inner").group_by("qid", "prob").agg(
        pl.len().alias("n"), pl.col("y").sum().alias("tp"), pl.first("deg").alias("deg")).sort(["qid", "prob"], descending=[False, True]).collect(engine="streaming")


def _curve(g, anchors):
    n = len(anchors)
    if not n:
        raise ValueError("empty calibration anchors")
    base = float((anchors["deg"] == 0).sum())
    rows, cur, pn, tn, deg = [], None, 0, 0, 0
    for qid, prob, add, hit, d in g.iter_rows():
        if qid != cur:
            cur, pn, tn, deg = qid, 0, 0, int(d)
        before = 1.0 if deg == 0 and pn == 0 else (1.25 * tn / (pn + .25 * deg) if deg else 0.0)
        pn += int(add)
        tn += int(hit)
        after = 1.25 * tn / (pn + .25 * deg) if deg else 0.0
        rows.append((float(prob), after - before, int(add), int(hit)))
    if not rows:
        return {"threshold": float(np.nextafter(1.0, np.inf)), "macro_f05": base / n, "pair_precision": 1.0,
                "pair_recall": 0.0, "pairs": 0, "true_pairs": int(anchors["deg"].sum())}
    d = pl.DataFrame(rows, schema=["prob", "delta", "pairs", "tp"], orient="row").group_by("prob").agg(
        pl.col("delta", "pairs", "tp").sum()).sort("prob", descending=True)
    score, pn, tn = base, 0, 0
    truth = int(anchors["deg"].sum())
    best = {"threshold": float(np.nextafter(float(d["prob"][0]), np.inf)), "macro_f05": score / n,
            "pair_precision": 1.0, "pair_recall": 0.0, "pairs": 0, "true_pairs": truth}
    for prob, delta, add, hit in d.iter_rows():
        score += float(delta)
        pn += int(add)
        tn += int(hit)
        z = {"threshold": float(prob), "macro_f05": score / n, "pair_precision": tn / pn if pn else 1.0,
             "pair_recall": tn / truth if truth else 1.0, "pairs": pn, "true_pairs": truth}
        if (z["macro_f05"], z["pair_precision"], z["threshold"]) > (best["macro_f05"], best["pair_precision"], best["threshold"]):
            best = z
    return best


def _fixed(lf, anchors, fold, top, th):
    if top:
        lf = lf.sort(["tid", "prob", "qid"], descending=[False, True, False]).group_by("tid", maintain_order=True).first()
    a = anchors.filter(pl.col("fold") == fold).select(pl.col("rid").alias("qid"), "deg")
    z = lf.filter(pl.col("prob") >= th).join(a.lazy(), on="qid", how="inner").group_by("qid").agg(
        pl.len().alias("predicted"), pl.col("y").sum().alias("true_positive")).collect(engine="streaming")
    z = a.lazy().join(z.lazy(), on="qid", how="left").with_columns(pl.col("predicted", "true_positive").fill_null(0)).collect()
    f = np.where(z["deg"].to_numpy() == 0, np.where(z["predicted"].to_numpy() == 0, 1.0, 0.0),
                 1.25 * z["true_positive"].to_numpy() / (z["predicted"].to_numpy() + .25 * z["deg"].to_numpy()))
    pn, tn, truth = (int(z[x].sum()) for x in ("predicted", "true_positive", "deg"))
    return {"macro_f05": float(f.mean()) if len(f) else 0.0, "pair_precision": tn / pn if pn else 1.0,
            "pair_recall": tn / truth if truth else 1.0, "pairs": pn, "true_pairs": truth}


def calibrate(data, runs, out, audit=False):
    data, out = path(data).resolve(), path(out).resolve()
    rs = _runs(data, runs, "train", _ids(data, "train"))
    man = rs[0]["manifest"]
    anchors = pl.read_parquet(data / "train/ref.parquet").select("rid", "deg", "fold")
    if not {0, 1}.issubset(set(anchors["fold"].to_list())):
        raise ValueError("prepared training folds do not include tune and audit")
    lf = _lf(rs)
    if "y" not in lf.collect_schema().names():
        raise ValueError("training inference has no labels")
    ds = {}
    for name, top in (("plain_threshold", False), ("target_top1_then_threshold", True)):
        ds[name] = _curve(_groups(lf, anchors, 0, top), anchors.filter(pl.col("fold") == 0))
    dec, sel = max(ds.items(), key=lambda x: (x[1]["macro_f05"], x[1]["pair_precision"], x[1]["threshold"]))
    z = {"version": ver, "kind": "full-corpus-calibration", "config": man["config"],
         "config_sha256": man["config_sha256"], "model_sha256": man["config"]["model"]["sha256"],
         "runs": [{"path": str(r["dir"]), "manifest_sha256": _sha(r["dir"] / "manifest.json")} for r in rs],
         "train_coverage": {"complete": True, "targets": len(_ids(data, "train"))},
         "tune": {"fold": 0, "decoders": ds, "selected": {"decoder": dec, **sel}}}
    if audit:
        z["audit"] = {"fold": 1, "decoder": dec, **_fixed(lf, anchors, 1, dec == "target_top1_then_threshold", sel["threshold"])}
    _write(out, z)
    print(json.dumps(z, indent=2))
    return z


def _sink(lf, p):
    try:
        lf.sink_parquet(p, compression="zstd")
    except TypeError:
        lf.sink_parquet(p)


def _groups_tsv(p):
    cur, xs = None, []
    for b in pq.ParquetFile(p).iter_batches(columns=["source1_entity_id", "target_entity_id"]):
        for qid, eid in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            if qid is None or eid is None:
                raise ValueError(f"unknown export id {p}")
            if cur is not None and qid < cur:
                raise ValueError(f"unsorted export pairs {p}")
            if qid != cur:
                if cur is not None:
                    if len(xs) != len(set(xs)):
                        raise ValueError(f"duplicate export pair {p}")
                    yield cur, xs
                cur, xs = qid, []
            xs.append(eid)
    if cur is not None:
        if len(xs) != len(set(xs)):
            raise ValueError(f"duplicate export pair {p}")
        yield cur, xs


def _tsv(ref, cand, match, out):
    cg, mg = _groups_tsv(cand), _groups_tsv(match)
    c = next(cg, None)
    m = next(mg, None)
    out.mkdir(parents=True, exist_ok=True)
    tm, tc = out / "matching_results.tsv.tmp", out / "candidate_pairs.tsv.tmp"
    with tm.open("w", encoding="utf-8", newline="") as fm, tc.open("w", encoding="utf-8", newline="") as fc:
        wm, wc = csv.writer(fm, delimiter="\t", lineterminator="\n"), csv.writer(fc, delimiter="\t", lineterminator="\n")
        wm.writerow(["source1_entity_id", "matched_entity_ids"])
        wc.writerow(["source1_entity_id", "candidate_entity_ids"])
        last, nr, nc, nm = None, 0, 0, 0
        for b in pq.ParquetFile(ref).iter_batches(columns=["source1_entity_id"]):
            for eid in b.column(0).to_pylist():
                if eid is None or (last is not None and eid <= last):
                    raise ValueError("test reference ids are not uniquely sorted")
                last, nr = eid, nr + 1
                while c is not None and c[0] < eid:
                    raise ValueError("candidate has unknown reference")
                while m is not None and m[0] < eid:
                    raise ValueError("match has unknown reference")
                cx = c[1] if c is not None and c[0] == eid else []
                mx = m[1] if m is not None and m[0] == eid else []
                if not set(mx).issubset(cx):
                    raise ValueError("match outside candidate set")
                wm.writerow([eid, ",".join(mx)])
                wc.writerow([eid, ",".join(cx)])
                nc += len(cx)
                nm += len(mx)
                if c is not None and c[0] == eid:
                    c = next(cg, None)
                if m is not None and m[0] == eid:
                    m = next(mg, None)
        if c is not None or m is not None:
            raise ValueError("export pairs exceed references")
    tm.replace(out / "matching_results.tsv")
    tc.replace(out / "candidate_pairs.tsv")
    return {"source1": nr, "candidate_pairs": nc, "matching_pairs": nm,
            "matching": str(out / "matching_results.tsv"), "candidate": str(out / "candidate_pairs.tsv")}


def export(data, runs, calibration, out):
    data, out = path(data).resolve(), path(out).resolve()
    cal = _json(calibration)
    if cal.get("kind") != "full-corpus-calibration" or not cal.get("train_coverage", {}).get("complete"):
        raise ValueError("export requires a full-pool calibration")
    sel = cal.get("tune", {}).get("selected", {})
    if sel.get("decoder") not in {"plain_threshold", "target_top1_then_threshold"} or "threshold" not in sel:
        raise ValueError("invalid calibration selection")
    rs = _runs(data, runs, "test", _ids(data, "test"))
    man = rs[0]["manifest"]
    if cal.get("config_sha256") != man["config_sha256"] or cal.get("model_sha256") != man["config"]["model"]["sha256"]:
        raise ValueError("calibration and inference model/configuration mismatch")
    return _export(data, rs, sel, calibration, out)


def provisional(data, runs, cutoff, decoder, out):
    data, out = path(data).resolve(), path(out).resolve()
    if isinstance(cutoff, bool) or not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("provisional cutoff must be finite and in [0, 1]")
    if decoder not in {"plain_threshold", "target_top1_then_threshold"}:
        raise ValueError("invalid provisional decoder")
    rs = _runs(data, runs, "test", _ids(data, "test"))
    man = rs[0]["manifest"]
    sel = {"decoder": decoder, "threshold": cutoff}
    selection = {"kind": "provisional-export-selection", "selection": sel,
                 "basis": "diagnostic cutoff pending full-pool calibration",
                 "config": man["config"], "config_sha256": man["config_sha256"],
                 "model_sha256": man["config"]["model"]["sha256"],
                 "test_coverage": {"complete": True, "targets": sum(len(r["coverage"]) for r in rs)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    selection_path = out.parent / (out.name + "_provisional_selection.json")
    _write(selection_path, selection)
    z = _export(data, rs, sel, selection_path, out)
    z["provisional"] = True
    return z


def _export(data, rs, sel, calibration, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    lf = _lf(rs, ("qid", "tid", "prob"))
    if sel["decoder"] == "target_top1_then_threshold":
        mf = lf.sort(["tid", "prob", "qid"], descending=[False, True, False]).group_by("tid", maintain_order=True).first()
    else:
        mf = lf
    mf = mf.filter(pl.col("prob") >= float(sel["threshold"]))
    ref = pl.scan_parquet(data / "test/ref.parquet").select(
        pl.col("rid").alias("qid"), pl.col("eid").alias("source1_entity_id"))
    tar = pl.concat([pl.scan_parquet(data / "test" / f"s{sr}.parquet").select(
        pl.col("rid").alias("tid"), pl.col("eid").alias("target_entity_id")) for sr in (2, 3)])
    with tf.TemporaryDirectory(dir=out.parent) as d:
        d = path(d)
        rp, cp, mp = d / "references.parquet", d / "candidates.parquet", d / "matches.parquet"
        _sink(ref.select("source1_entity_id").sort("source1_entity_id"), rp)
        _sink(lf.join(ref, on="qid", how="left").join(tar, on="tid", how="left").select(
            "source1_entity_id", "target_entity_id").sort(["source1_entity_id", "target_entity_id"]), cp)
        _sink(mf.join(ref, on="qid", how="left").join(tar, on="tid", how="left").select(
            "source1_entity_id", "target_entity_id").sort(["source1_entity_id", "target_entity_id"]), mp)
        z = _tsv(rp, cp, mp, out)
    z.update({"calibration": str(path(calibration).resolve()), "manifests": [str(r["dir"]) for r in rs]})
    print(json.dumps(z, indent=2))
    return z


def _check_data(root):
    data = root / "data"
    for sp in ("train", "test"):
        (data / sp).mkdir(parents=True)
    _write(data / "meta.json", {"check": True})
    ref = pl.DataFrame({"rid": [0, 1, 2, 3], "eid": ["S1-us", "S1-empty", "S1-fr", "S1-fit"],
                        "nm": ["alpha", "empty", "ecole", "fit"], "ad": ["1 main", "2 main", "3 rue", "4 road"],
                        "nn": ["alpha", "empty", "ecole", "fit"], "an": ["1 main", "2 main", "3 rue", "4 road"],
                        "co": ["us", "us", "france", "us"], "sr": [1, 1, 1, 1], "deg": [2, 0, 1, 0],
                        "uni": [0, 0, 0, 0], "blank": [0, 0, 0, 0], "fold": [0, 0, 1, 2]},
                       schema_overrides={"rid": pl.UInt32, "sr": pl.UInt8, "deg": pl.UInt32, "fold": pl.UInt8})
    for sp in ("train", "test"):
        ref.write_parquet(data / sp / "ref.parquet")
    rows = {2: [[10, "S2-us", "alpha", "1 main", "alpha", "1 main", "us", 2, 0],
                [11, "S2-fr", "ecole", "3 rue", "ecole", "3 rue", "france", 2, 2]],
            3: [[20, "S3-us", "alpha", "1 main", "alpha", "1 main", "us", 3, 0],
                [21, "S3-none", "none", "9 lane", "none", "9 lane", "us", 3, -1]]}
    for sp in ("train", "test"):
        for sr, rs in rows.items():
            d = pl.DataFrame(rs, schema=["rid", "eid", "nm", "ad", "nn", "an", "co", "sr", "own"], orient="row").with_columns(
                pl.col("rid").cast(pl.UInt32), pl.col("sr").cast(pl.UInt8), pl.col("own").cast(pl.Int32))
            d.write_parquet(data / sp / f"s{sr}.parquet")
    return data


def check():
    a = pl.DataFrame({"rid": [1, 2, 3], "deg": [0, 2, 2], "fold": [0, 0, 1]})
    g = pl.DataFrame({"qid": [2, 2], "prob": [.7, .7], "n": [1, 1], "tp": [1, 1], "deg": [2, 2]})
    z = _curve(g, a.filter(pl.col("fold") == 0))
    assert z["threshold"] == .7 and z["macro_f05"] == 1.0
    assert train.f05(pl.DataFrame({"rid": [1], "deg": [2]}), np.array([1, 1, 1]), np.array([1, 1, 0], np.uint8), np.array([True, True, True])) == 5 / 7
    with tf.TemporaryDirectory() as d:
        d = path(d)
        data = _check_data(d)
        models = d / "models"
        models.mkdir()
        (models / "fake.bin").write_bytes(b"fake")
        _write(models / "metadata.json", {"feature_names": feat.ff, "score_version": block.sv,
                                         "models": ["fake"], "model_files": {"fake": "fake.bin"}})
        old = _json(models / "metadata.json")
        _write(models / "metadata.json", {**old, "score_version": 1})
        try:
            _cfg(data, models, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("score version mismatch accepted")
        _write(models / "metadata.json", old)

        class fake:
            def predict_proba(self, x):
                p = np.clip(np.maximum(x[:, feat.ff.index("ns")], x[:, feat.ff.index("ads")]), 0, 1)
                return np.column_stack([1 - p, p])

        ms = {"fake": fake()}
        tr, te, cache = d / "train", d / "test", d / "cache"
        from unittest.mock import patch
        with patch.object(block, "setup", side_effect=ValueError("broken index")):
            try:
                infer(data, models, d / "broken", cache, "train", 1, 1, 1, ms=ms)
            except ValueError as e:
                assert str(e) == "broken index"
            else:
                raise AssertionError("index failure was hidden")
        infer(data, models, tr, cache, "train", 1, 1, 1, ms=ms)
        infer(data, models, te, cache, "test", 1, 1, 1, ms=ms)
        assert any(x["pairs"] == 0 for x in _json(te / "manifest.json")["parts"])
        cal = calibrate(data, [tr], d / "cal.json", audit=False)
        aud = calibrate(data, [tr], d / "audit.json", audit=True)
        assert aud["tune"]["selected"] == cal["tune"]["selected"] and aud["audit"]["fold"] == 1
        out = export(data, [te], d / "cal.json", d / "out")
        assert out["matching_pairs"] >= 2 and "S1-fr\tS2-fr" in (d / "out/matching_results.tsv").read_text()
        early = provisional(data, [te], .99, "target_top1_then_threshold", d / "early")
        assert early["provisional"]
        assert _json(d / "early_provisional_selection.json")["kind"] == "provisional-export-selection"
        assert (d / "early/candidate_pairs.tsv").read_bytes() == (d / "out/candidate_pairs.tsv").read_bytes()
        for bad in (-.1, 1.1, float("nan")):
            try:
                provisional(data, [te], bad, "target_top1_then_threshold", d / "bad")
            except ValueError:
                pass
            else:
                raise AssertionError("invalid provisional cutoff accepted")
        ids = [x.split("\t", 1)[0] for x in (d / "out/matching_results.tsv").read_text().splitlines()[1:]]
        assert ids == sorted(ids)
        bad = _json(d / "cal.json")
        bad["model_sha256"] = "wrong"
        _write(d / "bad.json", bad)
        try:
            export(data, [te], d / "bad.json", d / "bad")
        except ValueError:
            pass
        else:
            raise AssertionError("model mismatch accepted")
        (te / "parts/stale.npy").write_bytes(b"stale")
        try:
            _run(data, te, "test", _ids(data, "test"))
        except ValueError:
            pass
        else:
            raise AssertionError("unlisted artifact accepted")
        (te / "parts/stale.npy").unlink()
        _npy(te / "parts/part_0000000.npy", np.array([10, 10], np.uint32))
        try:
            _run(data, te, "test", _ids(data, "test"))
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate coverage accepted")
    print("checks passed")


def main():
    root = path(__file__).resolve().parents[1]
    p = ap.ArgumentParser()
    p.add_argument("--check", action="store_true")
    s = p.add_subparsers(dest="cmd")
    i = s.add_parser("infer")
    i.add_argument("--data", type=path, default=root / "cache/data")
    i.add_argument("--models", type=path, required=True)
    i.add_argument("--out", type=path, required=True)
    i.add_argument("--cache", type=path, default=root / "cache/block")
    i.add_argument("--split", choices=["train", "test"], default="test")
    i.add_argument("--k", type=int, default=10)
    i.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    i.add_argument("--batch", type=int, default=4096)
    i.add_argument("--country")
    i.add_argument("--rid-start", type=int)
    i.add_argument("--rid-stop", type=int)
    c = s.add_parser("calibrate")
    c.add_argument("--data", type=path, default=root / "cache/data")
    c.add_argument("--runs", type=path, nargs="+", required=True)
    c.add_argument("--out", type=path, required=True)
    c.add_argument("--audit", action="store_true")
    e = s.add_parser("export")
    e.add_argument("--data", type=path, default=root / "cache/data")
    e.add_argument("--runs", type=path, nargs="+", required=True)
    e.add_argument("--calibration", type=path, required=True)
    e.add_argument("--out", type=path, default=root / "output")
    v = s.add_parser("export-provisional")
    v.add_argument("--data", type=path, default=root / "cache/data")
    v.add_argument("--runs", type=path, nargs="+", required=True)
    v.add_argument("--cutoff", type=float, required=True)
    v.add_argument("--decoder", choices=["plain_threshold", "target_top1_then_threshold"],
                   default="target_top1_then_threshold")
    v.add_argument("--out", type=path, required=True)
    a = p.parse_args()
    if a.check:
        check()
    elif a.cmd == "infer":
        infer(a.data, a.models, a.out, a.cache, a.split, a.k, a.threads, a.batch, a.country, a.rid_start, a.rid_stop)
    elif a.cmd == "calibrate":
        calibrate(a.data, a.runs, a.out, a.audit)
    elif a.cmd == "export":
        export(a.data, a.runs, a.calibration, a.out)
    elif a.cmd == "export-provisional":
        provisional(a.data, a.runs, a.cutoff, a.decoder, a.out)
    else:
        p.error("choose infer, calibrate, export, or --check")


if __name__ == "__main__":
    main()
