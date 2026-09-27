'collect the compact, hash-bound inputs used by the final replay'
import argparse as ap
import hashlib as hh
import json
import shutil as sh
from pathlib import Path as path
from types import SimpleNamespace as ns
from urllib.parse import urlparse

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq

import cloud


def sha(p):
    with p.open("rb") as f:
        return hh.file_digest(f, "sha256").hexdigest()


def project(ml, uri, file, cols, mask, out):
    c, prefix = cloud.blobs(ml, uri)
    b = c.get_blob_client(prefix + file)
    before = b.get_blob_properties()
    az = fs.AzureFileSystem(account_name=urlparse(c.url).hostname.split(".")[0], sas_token=c.credential.signature)
    src = pq.ParquetFile(c.container_name + "/" + prefix + file, filesystem=az)
    temp = out.with_suffix(".part")
    rows = 0
    with pq.ParquetWriter(temp, pa.schema([src.schema_arrow.field(k) for k in cols]), compression="zstd") as w:
        for batch in src.iter_batches(batch_size=262144, columns=cols):
            t = pa.Table.from_batches([batch])
            q = t["qid"].to_numpy()
            if len(q) and (q.min() < 0 or q.max() >= len(mask)):
                raise ValueError("score query ids exceed the reference universe")
            t = t.filter(pa.array(mask[q]))
            w.write_table(t)
            rows += len(t)
    if before.etag != b.get_blob_properties().etag:
        raise ValueError("source blob changed during projection")
    temp.replace(out)
    return {"sha256": sha(out), "bytes": out.stat().st_size, "rows": rows, "columns": cols,
            "source": uri + file, "source_etag": before.etag, "source_bytes": before.size, "projection": "all France candidate pairs"}


def run(ledger, data, scores, out):
    out.mkdir(parents=True, exist_ok=True)
    ml = cloud.client(ns(**cloud.read(ledger)))
    refs = pl.read_parquet(data / "test/ref.parquet", columns=["rid", "co"]).sort("rid")
    if not np.array_equal(refs["rid"].to_numpy(), np.arange(len(refs))):
        raise ValueError("reference row order differs")
    mask = refs["co"].to_numpy() == "france"
    inv = {"schema": 1, "files": {}}
    dest = out / "collective.parquet"
    if not dest.exists():
        sh.copyfile(scores / "test_predictions.parquet", dest)
    inv["files"][dest.name] = {"sha256": sha(dest), "bytes": dest.stat().st_size,
                               "rows": pq.ParquetFile(dest).metadata.num_rows, "columns": ["qid", "tid", "p"],
                               "source": "azureml://datastores/workspaceblobstore/paths/azureml/amazites-model-collective-graph-20260927-04/out/test_predictions.parquet", "projection": "complete test candidate pool"}
    for name, uri, file, cols in (
        ("france.parquet", "azureml://datastores/workspaceblobstore/paths/azureml/e1279250-583f-4921-a2cf-b310ffaef6ec/out/", "prepared-test.parquet", ["qid", "tid", "co", "newest", "friend", "graph"]),
        ("swap-pool.parquet", "azureml://datastores/workspaceblobstore/paths/azureml/jolly_heart_lfccvw0y0k/out/experiments/stack_upgraded_lex/stack/", "test_pred.parquet", ["qid", "tid", "p2"]),
    ):
        print("projecting", name, flush=True)
        inv["files"][name] = project(ml, uri, file, cols, mask, out / name)
        (out / "manifest.json").write_text(json.dumps(inv, indent=2))
        print(name, inv["files"][name]["rows"], "rows", flush=True)
    return inv


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    for key in ("ledger", "data", "scores", "out"):
        p.add_argument("--" + key, type=path, required=True)
    a = p.parse_args()
    run(a.ledger, a.data, a.scores, a.out)
