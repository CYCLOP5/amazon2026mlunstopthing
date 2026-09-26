"""learned gate, inference and final-stack stages for shared azure assets"""

import argparse as ap
import concurrent.futures as cf
import os
import shutil as sh
import subprocess as sp
import sys
from pathlib import Path as path

import polars as pl

import gfeat
import infer
import post
import rfeat
import stack2
import train


def limits(gpu_count, threads):
    import torch
    visible = torch.cuda.device_count()
    cores = os.cpu_count() or 1
    if not isinstance(gpu_count, int) or not 1 <= gpu_count <= min(visible, cores) or not isinstance(threads, int) or threads < 1:
        raise ValueError("invalid worker resources for the visible hardware")
    return min(threads, cores)


def command(script, args, log, gpu=None, threads=16):
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "POLARS_MAX_THREADS": str(threads),
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        sp.run([sys.executable, str(path(__file__).parent / script), *map(str, args)], env=env,
               stdout=stream, stderr=sp.STDOUT, check=True)


def retrievers(assets, reverse_root, out):
    import retr
    rows = [{"model": retr.model_id, "revision": retr.revision, "checkpoint": str(path(assets) / "encoder"),
             "reverse_root": str(reverse_root)}]
    infer._write(out, rows)
    return out


def gate(assets, out, work, gpu_count=4, threads=80, previous=None):
    threads = limits(gpu_count, threads)
    assets, out, work = map(path, (assets, out, work))
    source = infer._json(assets / "assets.json")
    data = assets / "data"
    if infer._sha(data / "meta.json") != source["data_meta_sha256"]:
        raise ValueError("gate input data changed")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    scopes = [(split, co) for split in ("train", "test")
              for co in sorted(pl.read_parquet(data / split / "ref.parquet", columns=["co"])["co"].unique())]
    scopes.sort(key=lambda x: pl.read_parquet(data / x[0] / "ref.parquet", columns=["co"]).filter(pl.col("co") == x[1]).height)
    reverse_root = out / "reverse"
    def index(item, gpu):
        split, country = item
        command("reverse.py", ["--data", data, "--cache", work / "reverse-cache", "--checkpoint", assets / "encoder",
                               "--out", reverse_root, "--split", split, "--country", country, "--k", 5,
                               "--blank-only", "--batch", 1024, "--threads", max(1, threads // gpu_count)],
                out / "logs" / f"reverse-{split}-{country}.log", gpu, max(1, threads // gpu_count))
    if previous:
        sh.copytree(path(previous) / "reverse", reverse_root)
    else:
        index(scopes[0], 0)
        with cf.ThreadPoolExecutor(max_workers=gpu_count) as pool:
            futures = [pool.submit(index, item, i % gpu_count) for i, item in enumerate(scopes[1:])]
            for future in futures:
                future.result()
    for file in reverse_root.glob("*/pairs.parquet"):
        file.unlink()
    config = retrievers(assets, reverse_root, work / "retrievers.json")
    selected = source["queries"]
    def candidates(item, gpu):
        import block
        import embed
        label = item["directory"]
        src = work / "queries" / label
        sh.copytree(assets / "queries" / label, src)
        _, _, _, country, fold = embed.load_run(data, src)
        if country != item["country"] or fold != item["fold"]:
            raise ValueError("gate query manifest changed")
        infer._write(src / "metrics.json", {"country": country, "fold": fold, "score_version": block.sv})
        command("hybrid.py", ["--data", data, "--cache", work / ("fit-cache" if item["fold"] == 2 else "validation-cache"),
                              "--run", src, "--out", work / "runs" / label,
                              "--retrievers-file", config, "--k-lex", 10, "--k-dense", 50,
                              "--threads", max(1, threads // gpu_count), "--batch", 512, "--device", "cuda"],
                out / "logs" / f"candidates-{label}.log", gpu, max(1, threads // gpu_count))
    with cf.ThreadPoolExecutor(max_workers=gpu_count) as pool:
        futures = [pool.submit(candidates, item, i % gpu_count) for i, item in enumerate(selected)]
        for future in futures:
            future.result()
    rfeat.corpus(data, "train", assets / "normalizer.json", work / "features")
    gfeat.frequencies(data, "train", assets / "normalizer.json", work / "features")
    metadata = train.train(data, [work / "runs" / x["directory"] for x in selected if x["fold"] == 2],
                           [work / "runs" / x["directory"] for x in selected if x["fold"] == 0],
                           work / "gate", "lgb", threads, 1200, "hybrid-v3", assets / "normalizer.json", work / "features")
    (out / "gate").mkdir()
    for name in ("metadata.json", "normalizer.json", *metadata["model_files"].values()):
        sh.copyfile(work / "gate" / name, out / "gate" / name)
    infer._write(out / "result.json", {"gate": metadata, "retrieval": [infer._json(work / "runs" / x["directory"] / "metrics.json") for x in selected]})
    return {"gate": str(out / "gate"), "reverse": str(reverse_root)}


def score(assets, gate_root, neural_root, split, out, work, gpu_count=4, threads=80, country=None, lo=None, hi=None,
          k_gate=3, gate_floor=None, neural_floor=.01):
    threads = limits(gpu_count, threads)
    workers = gpu_count * min(2, max(1, (os.cpu_count() or 1) // gpu_count))
    worker_threads = max(1, threads // workers)
    assets, gate_root, neural_root, out, work = map(path, (assets, gate_root, neural_root, out, work))
    data = assets / "data"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    config = retrievers(assets, gate_root / "reverse", work / "retrievers.json")
    normalizer = gate_root / "gate/normalizer.json"
    rfeat.corpus(data, split, normalizer, work / "cache")
    gfeat.frequencies(data, split, normalizer, work / "cache")
    extra = (["--country", country] if country else [])
    if lo is not None:
        extra += ["--rid-start", lo]
    if hi is not None:
        extra += ["--rid-stop", hi]
    if gate_floor is not None:
        extra += ["--gate-floor", gate_floor]
    command("run.py", ["--data", data, "--cache", work / "cache", "--split", split, "--gate", gate_root / "gate",
                       "--neural", neural_root, "--out", out / split, "--retrievers-file", config,
                       "--k-lex", 10, "--k-dense", 50, "--k-gate", k_gate, "--neural-floor", neural_floor, "--shard-size", 200000,
                       "--workers", workers, "--gpu-ids", *range(gpu_count), "--threads", worker_threads,
                       "--query-batch", 2048, "--encoder-batch", 512, "--neural-batch", 128, "--device", "cuda", *extra],
            out / "inference.log", threads=worker_threads)
    return {"split": split, "runs": str(out / split)}


def finish(assets, gate_root, train_root, test_root, out, work, threads=64, stack_parameters=None, optuna_trials=64):
    assets, gate_root, out, work = map(path, (assets, gate_root, out, work))
    threads = min(threads, os.cpu_count() or 1)
    if threads < 1 or optuna_trials < 0:
        raise ValueError("invalid postprocessing resources")
    stack_parameters = stack_parameters or path(__file__).resolve().parents[1] / "reports/optuna-search.json"
    stack2.fit_params(threads, stack_parameters)
    data = assets / "data"
    normalizer = gate_root / "gate/normalizer.json"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    for split, roots in (("train", train_root), ("test", test_root)):
        roots = [roots] if isinstance(roots, (str, path)) else roots
        manifests = sorted(p for root in roots for p in path(root).glob("**/manifest.json"))
        if not manifests:
            raise ValueError("full scoring output has no manifests")
        post.prepare(data, [p.parent for p in manifests], split, work / f"{split}.parquet")
        rfeat.corpus(data, split, normalizer, work / "cache")
        gfeat.frequencies(data, split, normalizer, work / "cache")
    if optuna_trials:
        import opt
        search = out / "tuning-cache"
        stack2.fit(work / "train.parquet", normalizer, search / "fit", work / "cache", threads,
                   cache_only=True, parameters=stack_parameters)
        opt.prepare(work / "train.parquet", search, work / "cache", threads)
        workers = min(8, threads, optuna_trials)
        opt.run(search, out / "stack", optuna_trials, workers, max(1, threads // workers), stack_parameters)
    else:
        stack2.fit(work / "train.parquet", normalizer, out / "stack", work / "cache", threads, trees=800,
                   parameters=stack_parameters)
    def score_stack(split):
        command("stack2.py", ["score", "--scores", work / f"{split}.parquet", "--model", out / "stack",
                              "--out", work / f"stack-{split}.parquet", "--cache", work / "cache", "--threads", max(1, threads // 2)],
                out / f"stack-{split}.log", threads=max(1, threads // 2))
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(score_stack, split) for split in ("train", "test")]:
            future.result()
    command("post_eval.py", ["--scores", work / "stack-train.parquet", "--out", out / "development.json", "--threads", threads],
            out / "development.log", threads=threads)
    recipe = post.fit(work / "stack-train.parquet", work / "stack-test.parquet", out / "calibration.json")
    exports = {"base": post.export(work / "stack-test.parquet", out / "calibration.json", out / "output", threads)}
    import frule
    rule_recipe = {**recipe, "rules": frule.config(normalizer)}
    post.validate_recipe(rule_recipe)
    infer._write(out / "calibration-rules.json", rule_recipe)
    exports["rules"] = post.export(work / "stack-test.parquet", out / "calibration-rules.json", out / "output-rules",
                                    threads, normalizer, work / "cache")
    raw_test = work / "validation-test"
    raw_test.mkdir()
    for sr in (1, 2, 3):
        file = data / "test" / ("ref.parquet" if sr == 1 else f"s{sr}.parquet")
        (pl.scan_parquet(file).select(pl.col("eid").alias("entity_id"), pl.col("nm").alias("business_name"),
                                      pl.col("ad").alias("business_address"), pl.col("co").alias("country"))
         .sink_csv(raw_test / f"test_source{sr}.tsv", separator="\t"))
    from validate import validate as validate_output
    for name, folder in (("base", out / "output"), ("rules", out / "output-rules")):
        validation = validate_output(folder / "matching_results.tsv", folder / "candidate_pairs.tsv", raw_test)
        infer._write(out / f"validation-{name}.json", validation)
    saved = out / "scores"
    saved.mkdir()
    for name in ("train", "test", "stack-train", "stack-test"):
        for suffix in (".parquet", ".json"):
            sh.copyfile(work / (name + suffix), saved / (name + suffix))
    infer._write(out / "result.json", {"exports": exports, "development": infer._json(out / "development.json")})
    return exports


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("gate", "score", "ensemble", "finish"))
    p.add_argument("--assets", type=path)
    p.add_argument("--gate", type=path)
    p.add_argument("--neural", type=path)
    p.add_argument("--previous", type=path)
    p.add_argument("--stack-parameters", type=path)
    p.add_argument("--optuna-trials", type=int, default=64)
    p.add_argument("--members", nargs="+", type=path)
    p.add_argument("--train-runs", type=path, nargs="+")
    p.add_argument("--test-runs", type=path, nargs="+")
    p.add_argument("--split", choices=("train", "test"))
    p.add_argument("--country")
    p.add_argument("--rid-start", type=int)
    p.add_argument("--rid-stop", type=int)
    p.add_argument("--k-gate", type=int, default=3)
    p.add_argument("--gate-floor", type=float)
    p.add_argument("--neural-floor", type=float, default=.01)
    p.add_argument("--out", type=path, required=True)
    p.add_argument("--work", type=path, default=path("work/learned"))
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--threads", type=int, default=80)
    a = p.parse_args()
    if a.stage == "gate":
        print(gate(a.assets, a.out, a.work, a.gpus, a.threads, a.previous), flush=True)
    elif a.stage == "score":
        print(score(a.assets, a.gate, a.neural, a.split, a.out, a.work, a.gpus, a.threads, a.country, a.rid_start, a.rid_stop,
                    a.k_gate, a.gate_floor, a.neural_floor), flush=True)
    elif a.stage == "finish":
        print(finish(a.assets, a.gate, a.train_runs, a.test_runs, a.out, a.work, a.threads, a.stack_parameters, a.optuna_trials), flush=True)
    else:
        import ce
        print(ce.ensemble(a.members, a.out), flush=True)
