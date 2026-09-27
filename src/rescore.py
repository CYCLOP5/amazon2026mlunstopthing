"""replay a selected stack over cached train/test scores"""
import argparse as ap
import concurrent.futures as cf
import os
import shutil as sh
import tempfile as tf
from pathlib import Path as path

import full
import infer
import post
import stack2


def run(data, scores, model, context, out, threads=64):
    if not 1 <= threads <= (os.cpu_count() or 1) or out.exists():
        raise ValueError("invalid threads or existing replay output")
    metadata, digest = stack2.bundle(model)
    out.mkdir(parents=True)
    workers = min(2, threads)
    per = threads // workers
    with tf.TemporaryDirectory(prefix="stack-replay-") as tmp:
        work = path(tmp)
        sh.copytree(context / "cache", work / "cache")
        for split in ("train", "test"):
            for suffix in (".parquet", ".json"):
                sh.copyfile(scores / (split + suffix), work / (split + suffix))
            post.rebase(work / (split + ".parquet"), data)

        def score(split):
            dest = out / "scores" / f"stack-{split}.parquet"
            full.command("stack2.py", ["score", "--scores", work / f"{split}.parquet", "--model", model,
                                      "--out", dest, "--cache", work / "cache", "--threads", per],
                         out / f"{split}.log", threads=per)
            checked = post.verified(dest, split)
            if checked["stack_model"]["sha256"] != digest:
                raise ValueError("replayed model fingerprint changed")
            return {"split": split, "pairs": checked["pairs"], "score_sha256": checked["score_sha256"]}

        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(score, ("train", "test")))
    (out / "stack").mkdir()
    for name in ("metadata.json", *metadata["files"]):
        sh.copyfile(model / name, out / "stack" / name)
    result = {"model_sha256": digest, "scores": results}
    infer._write(out / "result.json", result)
    return result


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    for name in ("data", "scores", "model", "context", "out"):
        p.add_argument("--" + name, required=True, type=path)
    p.add_argument("--threads", type=int, default=64)
    a = p.parse_args()
    print(run(a.data, a.scores, a.model, a.context, a.out, a.threads), flush=True)
