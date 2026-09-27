"""balanced, disjoint country/rid ownership for multi-node scoring."""

import argparse as ap
import hashlib as hh
from pathlib import Path as path

import numpy as np
import polars as pl

import infer


def allocate(groups, capacities):
    if not groups or len(capacities) < len(groups) or any(n < 1 for n in capacities):
        raise ValueError("each nonempty country needs an inference allocation")
    counts = {key: len(ids) for key, ids in groups.items()}
    if not all(counts.values()):
        raise ValueError("empty scoring country")
    assigned = {key: [] for key in groups}
    for capacity in capacities:
        key = max(groups, key=lambda k: counts[k] / sum(assigned[k]) if assigned[k] else float("inf"))
        assigned[key].append(capacity)
    jobs = []
    for (split, country), ids in groups.items():
        ids = np.sort(ids)
        widths = assigned[(split, country)]
        ends = (np.cumsum(widths) * len(ids) // sum(widths)).tolist()
        start = 0
        for gpu_count, end in zip(widths, ends):
            selected = ids[start:end]
            if not len(selected):
                raise ValueError("empty inference shard")
            jobs.append({"split": split, "country": country, "rid_start": int(selected[0]),
                         "rid_stop": int(selected[-1]) + 1, "targets": len(selected),
                         "gpus": gpu_count, "target_ids_sha256": hh.sha256(np.asarray(selected, dtype="<u8").tobytes()).hexdigest()})
            start = end
    return jobs


def build(data, out, capacities):
    groups = {}
    for split in ("train", "test"):
        frame = pl.concat([pl.read_parquet(data / split / f"s{sr}.parquet", columns=["rid", "co"]) for sr in (2, 3)])
        for (country,), part in frame.partition_by("co", as_dict=True).items():
            groups[(split, country)] = part["rid"].to_numpy()
    jobs = allocate(groups, capacities)
    for i, job in enumerate(jobs):
        job["name"] = f"score{i:02d}"
    result = {"data_meta_sha256": infer._sha(data / "meta.json"), "gpus": sum(capacities), "jobs": jobs}
    infer._write(out, result)
    return result


def check():
    groups = {("train", "us"): np.array([0, 2, 3, 7, 9, 11]), ("test", "france"): np.array([0, 2, 5, 6])}
    jobs = allocate(groups, [4, 4, 2, 1])
    assert sum(j["gpus"] for j in jobs) == 11
    for key, ids in groups.items():
        owned = [ids[(ids >= j["rid_start"]) & (ids < j["rid_stop"])] for j in jobs if (j["split"], j["country"]) == key]
        assert np.array_equal(np.sort(np.concatenate(owned)), ids)
        assert len(np.unique(np.concatenate(owned))) == len(ids)
    print("multi-node ownership checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=path, default=path("cache/data"))
    p.add_argument("--out", type=path)
    p.add_argument("--capacities", type=int, nargs="+", default=[4] * 15 + [2])
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        check()
    elif a.out:
        print(build(a.data, a.out, a.capacities))
    else:
        p.error("--out or --check required")
