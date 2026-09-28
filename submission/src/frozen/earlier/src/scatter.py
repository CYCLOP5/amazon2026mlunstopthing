"""resume explicitly assigned, disjoint inference units from persistent checkpoints."""

import argparse as ap
import concurrent.futures as cf
import json
import shutil
from pathlib import Path as path

import numpy as np

import infer
import run


def assignment(plan, task):
    jobs = plan["workers"]
    seen = set()
    for job in jobs:
        for unit in job["units"]:
            rp = path(unit["runpath"])
            if rp.is_absolute() or len(rp.parts) != 2 or rp.parts[0] != "countries" or ".." in rp.parts:
                raise ValueError("unsafe unit path")
            key = (job["parent"], str(rp))
            if key in seen:
                raise ValueError("duplicate work ownership")
            seen.add(key)
    expected = {(x["parent"], x["runpath"]) for x in plan["expected_units"]}
    if seen != expected or len(expected) != len(plan["expected_units"]):
        raise ValueError("work ownership does not cover the original units exactly once")
    if not 0 <= task < len(jobs) or not jobs[task]["units"]:
        raise ValueError("invalid worker assignment")
    return jobs[task]


def execute(data, cache, gate, neural, previous, plan_file, task, out):
    plan = json.loads(plan_file.read_text())
    job = assignment(plan, task)
    owner = plan["owners"][str(job["parent"])]
    actual = run._owner(data, gate, neural, owner["split"], owner["countries"], owner["rid_start"],
                        owner["rid_stop"], owner["k_lex"], owner["k_dense"], owner["k_gate"],
                        owner["retrievers"], owner["device"], owner["threads"], owner["encoder_batch"],
                        owner["neural_batch"], owner["query_batch"], owner["neural_weight"], owner["shard_size"])
    if actual != owner:
        raise ValueError("checkpoint owner differs from current data or models")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    expected, paths, commands = [], [], []
    for unit in job["units"]:
        rd = out / unit["runpath"]
        src = previous / owner["split"] / unit["runpath"]
        rd.mkdir(parents=True, exist_ok=True)
        if (src / "manifest.json").is_file() and not (rd / "manifest.json").exists():
            manifest = json.loads((src / "manifest.json").read_text())
            for part in manifest["parts"]:
                for key in ("name", "coverage"):
                    rel = path(part[key])
                    if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "parts":
                        raise ValueError("unsafe checkpoint part")
                    (rd / rel).parent.mkdir(exist_ok=True)
                    shutil.copy2(src / rel, rd / rel)
            infer._write(rd / "manifest.json", manifest)
        co, lo, hi = unit["country"], unit["rid_start"], unit["rid_stop"]
        ids = infer._ids(data, owner["split"], co, lo, hi)
        expected.append(ids)
        paths.append(rd)
        cmd = run._command(data, cache, gate, neural, rd, owner["split"], co, lo, hi, owner["k_lex"],
                           owner["k_dense"], owner["k_gate"], owner["retrievers"], owner["device"],
                           owner["threads"], owner["encoder_batch"], owner["neural_batch"],
                           owner["query_batch"], owner["neural_weight"])
        commands.append((cmd, out / "logs" / (rd.name + ".log")))
    ids = np.sort(np.concatenate(expected))
    if len(ids) != len(np.unique(ids)):
        raise ValueError("overlapping target ownership")
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda x: run._child(*x), commands))
    if any(results):
        raise RuntimeError(f"inference children failed: {results}")
    infer._runs(data, paths, owner["split"], ids)
    z = {"kind": "redistributed-inference", "complete": True, "task": task, "owner": owner,
         "verification": {"status": "exact_coverage", "targets": len(ids), "full": False},
         "runs": [{**x, "childstatus": "succeeded"} for x in job["units"]]}
    infer._write(out / "runs.json", z)
    infer._write(out / "runpaths.json", {"runs": [x["runpath"] for x in job["units"]],
                                        "split": owner["split"], "full": False})
    print(json.dumps(z, indent=2))


def check():
    import copy
    u = {"runpath": "countries/a"}
    p = {"workers": [{"parent": 0, "units": [u]}], "expected_units": [{"parent": 0, **u}]}
    assert assignment(p, 0)["units"] == [u]
    for broken in ("duplicate", "missing", "escape"):
        q = copy.deepcopy(p)
        if broken == "duplicate":
            q["workers"].append(copy.deepcopy(q["workers"][0]))
        elif broken == "missing":
            q["expected_units"].append({"parent": 1, "runpath": "countries/b"})
        else:
            q["workers"][0]["units"][0]["runpath"] = "../a"
        try:
            assignment(q, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(broken)
    print("scatter ownership checks passed")


def main():
    p = ap.ArgumentParser()
    p.add_argument("--check", action="store_true")
    for x in ("data", "cache", "gate", "neural", "previous", "plan", "out"):
        p.add_argument("--" + x, type=path)
    p.add_argument("--task", type=int)
    a = p.parse_args()
    if a.check:
        return check()
    if any(getattr(a, x) is None for x in ("data", "cache", "gate", "neural", "previous", "plan", "out", "task")):
        p.error("all worker inputs are required")
    execute(a.data, a.cache, a.gate, a.neural, a.previous, a.plan, a.task, a.out)


if __name__ == "__main__":
    main()
