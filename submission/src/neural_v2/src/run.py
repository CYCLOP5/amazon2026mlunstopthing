import argparse as ap
import concurrent.futures as cf
import hashlib as hh
import json
import math
import os
import queue
import shutil as sh
import subprocess as sp
import sys
import tempfile as tf
from pathlib import Path as path

import numpy as np

import embed
import hybrid
import infer


def _json(p):
    with path(p).open(encoding="utf-8") as f:
        return json.load(f)


def _write(p, z):
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(z, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    q.replace(p)


def _hash(z):
    return hh.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()


def _dir(co, lo=None, hi=None):
    z = "country_" + hh.sha256(co.encode("utf-8")).hexdigest()[:16]
    return z if lo is None and hi is None else f"{z}_r_{lo}_{hi}"


def _countries(data, split, selected):
    allx = infer._countries(data, split, None)
    if selected is None:
        return allx
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate country selection")
    bad = sorted(set(selected) - set(allx))
    if bad:
        raise ValueError(f"unknown countries {bad}")
    return [x for x in allx if x in set(selected)]


def _ids(data, split, countries, lo, hi):
    xs = [infer._ids(data, split, x, lo, hi) for x in countries]
    z = np.sort(np.concatenate(xs)) if xs else np.empty(0, np.uint32)
    if len(z) != len(np.unique(z)):
        raise ValueError("selected target ids overlap")
    return z


def _work(data, split, countries, lo, hi, shard_size):
    out = []
    for co in countries:
        ids = infer._ids(data, split, co, lo, hi)
        if shard_size is None:
            out.append((co, lo, hi))
        elif not len(ids):
            out.append((co, lo, hi))
        else:
            for start in range(0, len(ids), shard_size):
                stop = start + shard_size
                out.append((co, int(ids[start]), int(ids[stop]) if stop < len(ids) else
                            (hi if hi is not None else int(ids[-1]) + 1)))
    return out


def _key(co, lo, hi):
    return co, lo, hi


def _warm_command(data, cache, split, co, retrievers, device, encoder_batch, retrievers_file=None):
    command = [sys.executable, str(path(__file__).resolve()), "--cache-only", "--data", str(data), "--cache",
             str(cache), "--split", split, "--country", co, "--retrievers", *retrievers, "--device", device,
             "--encoder-batch", str(encoder_batch)]
    return command + (["--retrievers-file", str(retrievers_file)] if retrievers_file else [])


def _cache_only(data, cache, split, co, retrievers, device, encoder_batch, retrievers_file=None):
    hybrid.setup(data, cache, co, split, 0, hybrid.configs(retrievers, retrievers_file), device, encoder_batch)


def _tree(p):
    p = path(p)
    if not p.is_dir():
        raise ValueError(f"missing model directory {p}")
    h = hh.sha256()
    files = 0
    for q in sorted(p.rglob("*")):
        if q.is_symlink():
            raise ValueError(f"invalid model artifact {q}")
        if q.is_dir():
            continue
        if not q.is_file():
            raise ValueError(f"invalid model artifact {q}")
        h.update(q.relative_to(p).as_posix().encode("utf-8"))
        h.update(b"\0")
        files += 1
        with q.open("rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
    if not files:
        raise ValueError(f"empty model directory {p}")
    return h.hexdigest()


def _owner(data, gate, neural, split, countries, lo, hi, klex, kdense, kgate, retrievers, device, threads,
           encoder_batch, neural_batch, query_batch, neural_weight, shard_size=None):
    z = {"data_meta_sha256": infer._sha(path(data) / "meta.json"), "gate_sha256": _tree(gate),
         "neural_sha256": _tree(neural), "split": split, "countries": countries, "rid_start": lo,
         "rid_stop": hi, "k_lex": klex, "k_dense": kdense, "k_gate": kgate, "retrievers": list(retrievers),
         "device": device, "threads": threads, "encoder_batch": encoder_batch, "neural_batch": neural_batch,
         "query_batch": query_batch, "neural_weight": neural_weight}
    if shard_size is not None:
        z["shard_size"] = shard_size
    return z


def _command(data, cache, gate, neural, out, split, co, lo, hi, klex, kdense, kgate, retrievers, device, threads,
              encoder_batch, neural_batch, query_batch, neural_weight, retrievers_file=None, neural_floor=None, gate_floor=None):
    z = [sys.executable, str(path(__file__).with_name("match.py")), "--data", str(data), "--cache", str(cache),
         "--gate", str(gate), "--neural", str(neural), "--out", str(out), "--split", split, "--country", co,
         "--k-lex", str(klex), "--k-dense", str(kdense), "--k-gate", str(kgate), "--retrievers", *retrievers,
         "--device", device, "--threads", str(threads), "--encoder-batch", str(encoder_batch), "--neural-batch",
         str(neural_batch), "--query-batch", str(query_batch), "--neural-weight", str(neural_weight)]
    if lo is not None:
        z.extend(("--rid-start", str(lo)))
    if hi is not None:
        z.extend(("--rid-stop", str(hi)))
    if retrievers_file is not None:
        z.extend(("--retrievers-file", str(retrievers_file)))
    if neural_floor is not None:
        z.extend(("--neural-floor", str(neural_floor)))
    if gate_floor is not None:
        z.extend(("--gate-floor", str(gate_floor)))
    return z


def _child(argv, log, env=None):
    with path(log).open("ab") as f:
        return sp.run(argv, stdin=sp.DEVNULL, stdout=f, stderr=sp.STDOUT, check=False, env=env).returncode


def _index(out, owner, full):
    p = out / "runs.json"
    if not p.exists():
        if any(out.iterdir()):
            raise ValueError("run output exists without runs.json")
        return {"version": 2, "owner": owner, "owner_sha256": _hash(owner), "full": full, "complete": False,
                "runs": []}
    z = _json(p)
    if z.get("version") != 2 or z.get("owner") != owner or z.get("owner_sha256") != _hash(owner):
        raise ValueError("existing run index ownership/configuration mismatch")
    if z.get("full") != full or not isinstance(z.get("runs"), list):
        raise ValueError("invalid existing run index")
    return z


def _runrel(name):
    return (path("countries") / name).as_posix()


def _prior(old, units, shard_size):
    names = {_key(co, lo, hi): _dir(co, lo, hi) if shard_size is not None else _dir(co) for co, lo, hi in units}
    if set(old) - set(names):
        raise ValueError("invalid existing country ownership")
    for k, x in old.items():
        if (x.get("country"), x.get("rid_start"), x.get("rid_stop")) != k or x.get("runpath") != _runrel(names[k]):
            raise ValueError("invalid existing country run path")
        if not isinstance(x.get("command"), list) or not all(isinstance(a, str) for a in x["command"]):
            raise ValueError("invalid existing country command")


def _check_args(countries, workers, threads, lo, hi, klex, kdense, kgate, encoder_batch, neural_batch, query_batch,
                neural_weight, shard_size, gpu_ids, device):
    cores = os.cpu_count() or 1
    if not countries:
        raise ValueError("no countries selected")
    if min(workers, threads, klex, kdense, kgate, encoder_batch, neural_batch, query_batch) < 1:
        raise ValueError("workers and matching options must be positive")
    if workers > cores or threads > cores or workers * threads > cores:
        raise ValueError(f"workers * threads must not exceed available cores ({cores})")
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError("rid start must be below rid stop")
    if shard_size is not None and shard_size < 1:
        raise ValueError("shard size must be positive")
    if gpu_ids is not None and (not gpu_ids or min(gpu_ids) < 0 or len(gpu_ids) != len(set(gpu_ids))):
        raise ValueError("gpu ids must be unique non-negative integers")
    if gpu_ids and device == "cpu":
        raise ValueError("gpu ids require device auto or cuda")
    if not math.isfinite(neural_weight) or not 0 <= neural_weight <= 1:
        raise ValueError("neural weight must be finite and in [0, 1]")


def run(data, cache, gate, neural, out, split="test", countries=None, rid_start=None, rid_stop=None, workers=1,
        threads=1, k_lex=10, k_dense=50, k_gate=20, retrievers=("e5",), device="auto", encoder_batch=64,
        neural_batch=32, query_batch=4096, neural_weight=1.0, calibration_out=None, audit=False, calibration=None,
        export_out=None, runner=None, shard_size=None, gpu_ids=None, retrievers_file=None, neural_floor=None, stack_dir=None, gate_floor=None):
    data, cache, gate, neural, out = (path(x).resolve() for x in (data, cache, gate, neural, out))
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    countries = _countries(data, split, countries)
    _check_args(countries, workers, threads, rid_start, rid_stop, k_lex, k_dense, k_gate, encoder_batch, neural_batch,
                 query_batch, neural_weight, shard_size, gpu_ids, device)
    if neural_floor is not None and (not math.isfinite(neural_floor) or not 0 <= neural_floor <= 1):
        raise ValueError("neural floor must be finite and in [0,1]")
    if gate_floor is not None and (not math.isfinite(gate_floor) or not 0 <= gate_floor <= 1):
        raise ValueError("gate floor must be finite and in [0,1]")
    sel = hybrid.configs(retrievers, retrievers_file) if retrievers_file else None
    if sel is not None:
        retrievers = tuple(embed.family(x["model"]) for x in sel)
    full = countries == infer._countries(data, split, None) and rid_start is None and rid_stop is None
    if calibration_out is not None and (split != "train" or not full):
        raise ValueError("calibration requires complete full training coverage")
    if audit and calibration_out is None:
        raise ValueError("audit requires --calibration-out")
    if (calibration is None) != (export_out is None):
        raise ValueError("test export requires both --calibration and --export-out")
    if calibration is not None and (split != "test" or not full):
        raise ValueError("export requires complete full test coverage")
    owner = _owner(data, gate, neural, split, countries, rid_start, rid_stop, k_lex, k_dense, k_gate, retrievers,
                    device, threads, encoder_batch, neural_batch, query_batch, neural_weight, shard_size)
    if sel is not None:
        owner["retriever_specs"] = sel
        retrievers_file = out / "retrievers.json"
    if neural_floor is not None:
        owner["neural_floor"] = neural_floor
    if gate_floor is not None:
        owner["gate_floor"] = gate_floor
    out.mkdir(parents=True, exist_ok=True)
    z = _index(out, owner, full)
    old = {}
    for x in z["runs"]:
        if not isinstance(x, dict):
            raise ValueError("invalid existing country ownership")
        co, lo, hi = x.get("country"), x.get("rid_start"), x.get("rid_stop")
        if (not isinstance(co, str) or (lo is not None and (not isinstance(lo, int) or isinstance(lo, bool))) or
                (hi is not None and (not isinstance(hi, int) or isinstance(hi, bool)))):
            raise ValueError("invalid existing country ownership")
        k = _key(co, lo, hi)
        if k in old:
            raise ValueError("invalid existing country ownership")
        old[k] = x
    units = _work(data, split, countries, rid_start, rid_stop, shard_size)
    _prior(old, units, shard_size)
    runroot, logroot = out / "countries", out / "logs"
    runroot.mkdir(exist_ok=True)
    logroot.mkdir(exist_ok=True)
    jobs = []
    for co, lo, hi in units:
        name = _dir(co, lo, hi) if shard_size is not None else _dir(co)
        rd = (runroot / name).resolve()
        if rd.parent != runroot.resolve():
            raise ValueError("unsafe country directory")
        cmd = _command(data, cache, gate, neural, rd, split, co, lo, hi, k_lex, k_dense, k_gate, retrievers,
                       device, threads, encoder_batch, neural_batch, query_batch, neural_weight, retrievers_file, neural_floor, gate_floor)
        rec = {"country": co, "rid_start": lo, "rid_stop": hi, "runpath": _runrel(name), "childstatus": "pending",
               "config": owner, "command": cmd}
        if _key(co, lo, hi) in old:
            prior = old[_key(co, lo, hi)]
            if prior.get("config") != owner:
                raise ValueError("existing country run ownership/configuration mismatch")
        jobs.append((co, lo, hi, rd, logroot / f"{name}.log", cmd, rec))
    z["runs"] = [x[-1] for x in jobs]
    z["complete"] = False
    z.pop("verification", None)
    z.pop("warmups", None)
    _write(out / "runs.json", z)
    if sel is not None:
        _write(retrievers_file, sel)
    runner = _child if runner is None else runner

    slots = None
    if gpu_ids:
        slots = queue.Queue()
        for _ in range(math.ceil(workers / len(gpu_ids))):
            for gpu in gpu_ids:
                slots.put(str(gpu))

    def invoke(cmd, log):
        gpu = slots.get() if slots is not None else None
        env = None if gpu is None else {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        try:
            return runner(cmd, log, env), gpu
        finally:
            if slots is not None:
                slots.put(gpu)

    warm = []
    for co in countries:
        if sum(x[0] == co for x in jobs) > 1:
            cmd = _warm_command(data, cache, split, co, retrievers, device, encoder_batch, retrievers_file)
            rec = {"country": co, "childstatus": "running", "command": cmd}
            warm.append(rec)
            z["warmups"] = warm
            _write(out / "runs.json", z)
            try:
                code, gpu = invoke(cmd, logroot / f"{_dir(co)}.warmup.log")
                rec["gpu_id"] = gpu
                if code:
                    rec.update({"childstatus": "failed", "returncode": code, "error": f"warmup exited {code}"})
                    raise RuntimeError(f"cache warmup failed for {co}")
                rec.update({"childstatus": "succeeded", "returncode": code})
            except Exception as e:
                if rec["childstatus"] != "failed":
                    rec.update({"childstatus": "failed", "returncode": None, "error": f"{type(e).__name__}: {e}"})
                _write(out / "runs.json", z)
                raise
            _write(out / "runs.json", z)

    def one(co, lo, hi, log, cmd):
        try:
            code, gpu = invoke(cmd, log)
            if code != 0:
                return _key(co, lo, hi), "failed", code, f"child exited {code}", gpu
            return _key(co, lo, hi), "succeeded", code, None, gpu
        except Exception as e:
            return _key(co, lo, hi), "failed", None, f"{type(e).__name__}: {e}", None

    by_country = {_key(x[0], x[1], x[2]): x[-1] for x in jobs}
    with cf.ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        fs = []
        for co, lo, hi, _, log, cmd, rec in jobs:
            rec["childstatus"] = "running"
            rec["log"] = str(log.resolve())
            fs.append(pool.submit(one, co, lo, hi, log, cmd))
        _write(out / "runs.json", z)
        for f in cf.as_completed(fs):
            k, status, code, error, gpu = f.result()
            rec = by_country[k]
            rec["childstatus"], rec["returncode"] = status, code
            rec["gpu_id"] = gpu
            if error is None:
                rec.pop("error", None)
            else:
                rec["error"] = error
            _write(out / "runs.json", z)
    failed = [_key(x["country"], x["rid_start"], x["rid_stop"]) for x in z["runs"] if x["childstatus"] != "succeeded"]
    if failed:
        z["verification"] = {"status": "skipped", "reason": "failed children", "countries": failed}
        _write(out / "runs.json", z)
        raise RuntimeError(f"matching children failed for {failed}")
    exp = _ids(data, split, countries, rid_start, rid_stop)
    paths = [x[3] for x in jobs]
    try:
        infer._runs(data, paths, split, exp)
    except Exception as e:
        z["verification"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        _write(out / "runs.json", z)
        raise
    z["complete"] = True
    z["verification"] = {"status": "exact_coverage", "targets": len(exp), "full": full}
    _write(out / "runs.json", z)
    _write(out / "runpaths.json", {"runs": [x["runpath"] for x in z["runs"]], "split": split, "full": full})
    res = {"runs": [str(x) for x in paths], "targets": len(exp), "full": full}
    if calibration_out is not None:
        res["calibration"] = infer.calibrate(data, paths, calibration_out, audit=audit)
    if calibration is not None:
        cal = _json(path(calibration))
        if cal.get("kind") == "segmented-postprocessor":
            import post
            prepared = out / "post-base.parquet"
            if not prepared.exists():
                post.prepare(data, paths, "test", prepared)
            else:
                meta = post.verified(prepared, "test", paths)
                if meta["config_sha256"] != _json(paths[0] / "manifest.json")["config_sha256"]:
                    raise ValueError("cached postprocessor scores use different inference runs")
            if cal.get("stack_model"):
                import stack2
                if stack_dir is None or stack2.bundle(stack_dir)[1] != cal["stack_model"]["sha256"]:
                    raise ValueError("postprocessor requires its selected pairwise stack model")
                stacked = out / "post-stacked.parquet"
                if not stacked.exists():
                    stack2.score(prepared, stack_dir, stacked, cache, threads)
                else:
                    stacked_meta = post.verified(stacked, "test", paths)
                    if stacked_meta.get("parent_scores_sha256") != post.verified(prepared, "test")["score_sha256"]:
                        raise ValueError("cached stack uses different base scores")
                prepared = stacked
            elif stack_dir is not None:
                raise ValueError("unselected pairwise stack supplied")
            normalizer = None
            if cal.get("rules"):
                gm = _json(path(gate_dir) / "metadata.json")
                normalizer = path(gate_dir) / gm["normalizer"]["file"]
            res["export"] = post.export(prepared, calibration, export_out, threads, normalizer, cache)
        else:
            if stack_dir is not None:
                raise ValueError("legacy calibration cannot use a pairwise stack")
            res["export"] = infer.export(data, paths, calibration, export_out)
    return res


def check():
    def fake(data, fail=None, model=None):
        calls = []

        def runner(argv, log, env):
            calls.append((argv, env))
            if "--cache-only" in argv:
                return 0
            co, out = argv[argv.index("--country") + 1], path(argv[argv.index("--out") + 1])
            split = argv[argv.index("--split") + 1]
            lo = int(argv[argv.index("--rid-start") + 1]) if "--rid-start" in argv else None
            hi = int(argv[argv.index("--rid-stop") + 1]) if "--rid-stop" in argv else None
            if model is not None:
                guard = out / "model_guard.json"
                if guard.exists() and _json(guard)["sha256"] != model[0]:
                    return 9
                _write(guard, {"sha256": model[0]})
            ids = infer._ids(data, split, co, lo, hi)
            parts = out / "parts"
            parts.mkdir(parents=True, exist_ok=True)
            p, c = parts / "part_0000000.parquet", parts / "part_0000000.npy"
            infer._empty(split).write_parquet(p)
            np.save(c, ids)
            cfg = {"data_meta_sha256": infer._sha(data / "meta.json"), "model": {"sha256": "check"}}
            man = {"version": infer.ver, "split": split, "config": cfg,
                    "config_sha256": _hash(cfg), "parts": [{"name": "parts/part_0000000.parquet",
                    "coverage": "parts/part_0000000.npy", "queries": len(ids), "pairs": 0,
                    "pair_sha256": infer._sha(p), "coverage_sha256": infer._sha(c)}]}
            _write(out / "manifest.json", man)
            return 7 if co == fail else 0

        return runner, calls

    with tf.TemporaryDirectory() as tmp:
        root = path(tmp)
        data = infer._check_data(root)
        gate, neural = root / "gate", root / "neural"
        gate.mkdir(); neural.mkdir()
        (gate / "weights.bin").write_bytes(b"gate")
        (neural / "weights.bin").write_bytes(b"neural")
        q = infer.pl.read_parquet(data / "test/s3.parquet").with_columns(infer.pl.lit("neverland").alias("co"))
        q.write_parquet(data / "test/s3.parquet")
        assert _countries(data, "test", None) == ["france", "neverland", "us"]
        assert _dir("France") == _dir("France") and _dir("France") != _dir("neverland") and "/" not in _dir("France")
        model = ["first"]
        runner, calls = fake(data, model=model)
        out = root / "out"
        z = run(data, root / "cache", gate, neural, out, workers=2, threads=1, k_lex=3,
                k_dense=4, k_gate=5, retrievers=("e5", "qwen3"), device="cpu", encoder_batch=7, neural_batch=5,
                query_batch=123, neural_weight=.6, runner=runner)
        assert z["full"] and len(calls) == 3 and _json(out / "runs.json")["complete"]
        first = {tuple(x) for x, _ in calls}
        run(data, root / "cache", gate, neural, out, workers=2, threads=1, k_lex=3, k_dense=4,
            k_gate=5, retrievers=("e5", "qwen3"), device="cpu", encoder_batch=7, neural_batch=5, query_batch=123,
            neural_weight=.6, runner=runner)
        assert len(calls) == 6 and {tuple(x) for x, _ in calls[3:]} == first
        cmd = calls[0][0]
        for flag, value in (("--k-lex", "3"), ("--k-dense", "4"), ("--k-gate", "5"), ("--encoder-batch", "7"),
                            ("--neural-batch", "5"), ("--query-batch", "123"), ("--neural-weight", "0.6")):
            assert cmd[cmd.index(flag) + 1] == value
        assert cmd[cmd.index("--retrievers") + 1:cmd.index("--device")] == ["e5", "qwen3"]
        assert {x[x.index("--cache") + 1] for x, _ in calls[:3]} == {str((root / "cache").resolve())}
        model[0] = "changed-at-same-path"
        try:
            run(data, root / "cache", gate, neural, out, workers=2, threads=1, k_lex=3,
                k_dense=4, k_gate=5, retrievers=("e5", "qwen3"), device="cpu", encoder_batch=7, neural_batch=5,
                query_batch=123, neural_weight=.6, runner=runner)
        except RuntimeError as e:
            assert "matching children failed" in str(e) and len(calls) == 9
        else:
            raise AssertionError("resumed model guard bypassed")
        injected = _command(data, root / "cache", gate, neural, root / "x", "test", "x; touch bad",
                            None, None, 1, 1, 1, ("e5",), "cpu", 1, 1, 1, 1, 1.)
        assert injected[injected.index("--country") + 1] == "x; touch bad"
        adaptive = _command(data, root / "cache", gate, neural, root / "x", "test", "us",
                            None, None, 10, 50, 50, ("e5-small",), "cpu", 1, 1, 1, 1, 1., gate_floor=.01)
        assert adaptive[adaptive.index("--gate-floor") + 1] == "0.01"
        try:
            run(data, root / "cache", gate, neural, root / "shard", countries=["us"],
                calibration_out=root / "calibration.json", runner=runner)
        except ValueError as e:
            assert "complete full" in str(e)
        else:
            raise AssertionError("shard calibration accepted")
        try:
            run(data, root / "cache", gate, neural, root / "shard-export", countries=["us"],
                calibration=root / "calibration.json", export_out=root / "export", runner=runner)
        except ValueError as e:
            assert "complete full" in str(e)
        else:
            raise AssertionError("shard export accepted")
        hi = int(_ids(data, "test", _countries(data, "test", None), None, None)[-1]) + 2
        empty = run(data, root / "cache", gate, neural, root / "empty", countries=["us"],
                    rid_start=hi, rid_stop=hi + 1, shard_size=1, runner=runner)
        assert not empty["full"] and empty["targets"] == 0
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        sharded, seen = fake(data, "france")
        try:
            run(data, root / "cache", gate, neural, root / "failed", workers=1, threads=1,
                shard_size=1, gpu_ids=(0,), device="cuda", runner=sharded)
        except RuntimeError as e:
            assert "france" in str(e)
        else:
            raise AssertionError("failed worker accepted")
        failed = _json(root / "failed/runs.json")
        assert any(x["childstatus"] == "failed" for x in failed["runs"])
        assert any(x["childstatus"] == "succeeded" for x in failed["runs"])
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == old_cuda
        assert all(env["CUDA_VISIBLE_DEVICES"] == "0" for _, env in seen)
        runner, shard_calls = fake(data)
        shard = run(data, root / "cache", gate, neural, root / "sharded", workers=2, threads=1,
                    shard_size=1, gpu_ids=(0, 1), device="cuda", runner=runner)
        index = _json(root / "sharded/runs.json")
        match_calls = [(x, env) for x, env in shard_calls if "--cache-only" not in x]
        warm_calls = [(x, env) for x, env in shard_calls if "--cache-only" in x]
        assert shard["full"] and len(index["runs"]) == shard["targets"]
        assert len({x["runpath"] for x in index["runs"]}) == len(index["runs"])
        assert warm_calls and shard_calls[:len(warm_calls)] == warm_calls
        assert all(env["CUDA_VISIBLE_DEVICES"] in {"0", "1"} for _, env in match_calls + warm_calls)
        assert _json(root / "sharded/runs.json")["verification"]["targets"] == shard["targets"]
        assert all("--rid-start" in x and "--rid-stop" in x for x, _ in match_calls)
        runner, _ = fake(data)
        cal = root / "calibration.json"
        train = run(data, root / "cache", gate, neural, root / "sharded-train", split="train",
                    workers=2, threads=1, shard_size=1, calibration_out=cal, runner=runner)
        test = run(data, root / "cache", gate, neural, root / "sharded-test", workers=2,
                   threads=1, shard_size=1, calibration=cal, export_out=root / "export", runner=runner)
        assert train["full"] and test["full"] and len(train["runs"]) > len(_countries(data, "train", None))
        assert len(test["runs"]) > len(_countries(data, "test", None))
        moved = root / "moved"
        moved.mkdir()
        sh.copytree(data, moved / "data")
        sh.copytree(gate, moved / "gate")
        sh.copytree(neural, moved / "neural")
        sh.copytree(root / "sharded", moved / "out")
        runner, moved_calls = fake(moved / "data")
        resumed = run(moved / "data", moved / "new-cache", moved / "gate", moved / "neural", moved / "out",
                      workers=2, threads=1, shard_size=1, gpu_ids=(6, 7), device="cuda", runner=runner)
        moved_index = _json(moved / "out/runs.json")
        moved_paths = _json(moved / "out/runpaths.json")["runs"]
        assert resumed["targets"] == shard["targets"] and moved_index["complete"]
        assert all(not path(x).is_absolute() and (moved / "out" / x).is_dir() for x in moved_paths)
        assert str(root / "sharded") not in json.dumps(moved_index)
        assert {x[x.index("--cache") + 1] for x, _ in moved_calls if "--cache-only" not in x} == {
            str((moved / "new-cache").resolve())}
        (moved / "gate/weights.bin").write_bytes(b"changed-gate")
        try:
            run(moved / "data", moved / "new-cache", moved / "gate", moved / "neural", moved / "out",
                workers=2, threads=1, shard_size=1, gpu_ids=(6, 7), device="cuda", runner=runner)
        except ValueError as e:
            assert "ownership/configuration mismatch" in str(e)
        else:
            raise AssertionError("changed gate accepted")
        (moved / "gate/weights.bin").write_bytes(b"gate")
        (moved / "neural/weights.bin").write_bytes(b"changed-neural")
        try:
            run(moved / "data", moved / "new-cache", moved / "gate", moved / "neural", moved / "out",
                workers=2, threads=1, shard_size=1, gpu_ids=(6, 7), device="cuda", runner=runner)
        except ValueError as e:
            assert "ownership/configuration mismatch" in str(e)
        else:
            raise AssertionError("changed neural accepted")
        partial = root / "moved-partial"
        partial.mkdir()
        sh.copytree(data, partial / "data")
        sh.copytree(gate, partial / "gate")
        sh.copytree(neural, partial / "neural")
        sh.copytree(root / "failed", partial / "out")
        runner, _ = fake(partial / "data")
        resumed = run(partial / "data", partial / "new-cache", partial / "gate", partial / "neural", partial / "out",
                      workers=1, threads=1, shard_size=1, gpu_ids=(9,), device="cuda", runner=runner)
        assert resumed["full"] and _json(partial / "out/runs.json")["complete"]
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
    p.add_argument("--country", nargs="+")
    p.add_argument("--rid-start", type=int)
    p.add_argument("--rid-stop", type=int)
    p.add_argument("--shard-size", type=int)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--k-lex", type=int, default=10)
    p.add_argument("--k-dense", type=int, default=50)
    p.add_argument("--k-gate", type=int, default=20)
    p.add_argument("--retrievers", choices=("e5", "qwen3", "e5-large", "e5-small"), nargs="+", default=["e5"])
    p.add_argument("--retrievers-file", type=path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--gpu-ids", type=int, nargs="+")
    p.add_argument("--encoder-batch", type=int, default=64)
    p.add_argument("--neural-batch", type=int, default=32)
    p.add_argument("--query-batch", type=int, default=4096)
    p.add_argument("--neural-weight", type=float, default=1.0)
    p.add_argument("--neural-floor", type=float)
    p.add_argument("--gate-floor", type=float)
    p.add_argument("--stack-model-dir", type=path)
    p.add_argument("--calibration-out", type=path)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--calibration", type=path)
    p.add_argument("--export-out", type=path)
    p.add_argument("--cache-only", action="store_true")
    a = p.parse_args()
    if a.check:
        check()
    elif a.cache_only and a.country and len(a.country) == 1:
        _check_args(a.country, a.workers, a.threads, a.rid_start, a.rid_stop, a.k_lex, a.k_dense, a.k_gate,
                    a.encoder_batch, a.neural_batch, a.query_batch, a.neural_weight, a.shard_size, a.gpu_ids, a.device)
        _cache_only(a.data, a.cache, a.split, a.country[0], a.retrievers, a.device, a.encoder_batch, a.retrievers_file)
    elif a.gate and a.neural and a.out:
        z = run(a.data, a.cache, a.gate, a.neural, a.out, a.split, a.country, a.rid_start, a.rid_stop, a.workers,
                 a.threads, a.k_lex, a.k_dense, a.k_gate, a.retrievers, a.device, a.encoder_batch, a.neural_batch,
                 a.query_batch, a.neural_weight, a.calibration_out, a.audit, a.calibration, a.export_out,
                 shard_size=a.shard_size, gpu_ids=a.gpu_ids, retrievers_file=a.retrievers_file, neural_floor=a.neural_floor,
                 stack_dir=a.stack_model_dir, gate_floor=a.gate_floor)
        print(json.dumps(z, indent=2))
    else:
        p.error("use --check, --cache-only --country, or --gate --neural --out")


if __name__ == "__main__":
    main()
