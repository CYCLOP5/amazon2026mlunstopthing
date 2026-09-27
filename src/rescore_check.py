'real-data replay smoke; requires the retained learned caches'
import json
import subprocess as sp
import sys
import tempfile as tf
from pathlib import Path as path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "src")
import infer
import post
import stack2

root = path.cwd()
model = root / "artifacts/retune-r2/finalists/run01"
_, digest = stack2.bundle(model)
with tf.TemporaryDirectory(prefix="rescore-check-") as tmp:
    work = path(tmp)
    (work / "scores").mkdir()
    (work / "context/cache").mkdir(parents=True)
    (work / "context/cache/rich").symlink_to(root / "cache/rich", target_is_directory=True)
    for split in ("train", "test"):
        source = root / f"artifacts/full-result/scores/{split}.parquet"
        frame = next(pq.ParquetFile(source).iter_batches(batch_size=64))
        dest = work / "scores" / f"{split}.parquet"
        pq.write_table(pa.Table.from_batches([frame]), dest)
        metadata = {**post.verified(source, split), "pairs": len(frame), "score_sha256": infer._sha(dest), "scope": "64-row replay smoke"}
        infer._write(dest.with_suffix(".json"), metadata)
    (work / "output").mkdir()
    proc = sp.run([sys.executable, "src/rescore.py", "--data", str(root / "cache/data"), "--scores", str(work / "scores"),
                   "--model", str(model), "--context", str(work / "context"), "--out", str(work / "output/replay"), "--threads", "1"])
    if proc.returncode:
        for log in (work / "output/replay").glob("*.log"):
            (root / "artifacts" / ("rescore-check-" + log.name)).write_bytes(log.read_bytes())
        raise RuntimeError("replay smoke failed; see artifacts/rescore-check-*.log")
    result = infer._json(work / "output/replay/result.json")
    assert result["model_sha256"] == digest and len(result["scores"]) == 2
    for split in ("train", "test"):
        file = work / f"output/replay/scores/stack-{split}.parquet"
        metadata = post.verified(file, split)
        probability = pq.read_table(file, columns=["stack_prob"])["stack_prob"].to_numpy()
        assert len(probability) == 64 and np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()
        assert metadata["pairs"] == 64 and metadata["stack_model"]["sha256"] == digest
    assert stack2.bundle(work / "output/replay/stack")[1] == digest
    print(json.dumps({"train_rows": 64, "test_rows": 64, "model_sha256": digest, "passed": True}))
