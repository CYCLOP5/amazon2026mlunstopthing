'stage immutable learned-pipeline assets and unseen-owner gate samples'

import argparse as ap
import json
import os
import shutil as sh
from pathlib import Path as path

import polars as pl

import block
import embed
import infer
import post
import retr


def copy(src, dest):
    dest = path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(path(src).resolve(), dest)
    except OSError:
        sh.copyfile(path(src).resolve(), dest)


def queries(data, encoder, out, fit=30000, validation=5000):
    refs = pl.read_parquet(path(data) / "train/ref.parquet")
    seen = pl.read_parquet(path(encoder) / "training_pairs.parquet", columns=["own"])["own"].unique().cast(pl.UInt32)
    targets = pl.concat([pl.read_parquet(path(data) / "train" / f"s{i}.parquet") for i in (2, 3)])
    rs = []
    for country in sorted(refs["co"].unique()):
        for label, fold, n in (("fit", 2, fit), ("val", 0, validation)):
            pool = refs.filter((pl.col("co") == country) & (pl.col("fold") == fold))
            if fold == 2:
                pool = pool.filter(~pl.col("rid").is_in(seen.implode()))
            else:
                pool = pool.filter(pl.Series(post.partition(pool["rid"].to_numpy(), 3) == 1))
            if len(pool) < n:
                raise ValueError("too few encoder-unseen gate entities")
            anchors = pool.sort("rid").sample(n=n, seed=42).sort("rid")
            local = targets.filter(pl.col("co") == country)
            positive = local.filter(pl.col("own").is_in(anchors["rid"].cast(pl.Int32).implode()))
            negative = local.filter(pl.col("own") < 0).with_columns(
                (pl.concat_str("co", "nn", "an", separator="\t").hash(seed=42) % 10).alias("h"))
            negative = negative.filter(pl.col("h") >= 2) if fold == 2 else negative.filter(pl.col("h") == 0)
            negative = negative.sample(n=min(n, len(negative)), seed=43).drop("h")
            frame = pl.concat([positive, negative]).sort("rid")
            if len(positive) != anchors["deg"].sum():
                raise ValueError("gate sample lost a selected alias")
            dest = path(out) / f"{label}-{country}"
            infer._pq(anchors, dest / "anchors.parquet")
            infer._pq(frame, dest / "queries.parquet")
            embed.load_run(path(data), dest)
            infer._write(dest / "metrics.json", {"country": country, "fold": fold, "score_version": block.sv})
            rs.append({"directory": dest.name, "fold": fold, "country": country,
                            "anchors": len(anchors), "queries": len(frame), "positive": len(positive),
                            "encoder_owners_excluded": fold == 2})
    return rs


def stage(data, encoder, hard, normalizer, out, hf=None):
    data, encoder, hard, out = map(path, (data, encoder, hard, out))
    if out.exists():
        raise ValueError("full-pipeline asset output must be new")
    out.mkdir(parents=True)
    for file in data.rglob("*"):
        if file.is_file():
            copy(file, out / "data" / file.relative_to(data))
    model, digest = retr.bundle(encoder)
    for name in (*model["files"], "retriever.json", "training_pairs.parquet"):
        copy(encoder / name, out / "encoder" / name)
    for file in hard.iterdir():
        if file.is_file():
            copy(file, out / "hard" / file.name)
    copy(normalizer, out / "normalizer.json")
    sources = infer._json(path(__file__).parents[1] / "reports/model_sources.json")
    wanted = {"intfloat/multilingual-e5-base": "base", "intfloat/multilingual-e5-large-instruct": "large",
              "BAAI/bge-reranker-v2-m3": "bge", "intfloat/multilingual-e5-small": "small"}
    hf = path(hf or path.home() / ".cache/huggingface/hub")
    models = []
    for source in sources:
        if source["model"] not in wanted:
            continue
        name = wanted[source["model"]]
        original = hf / ("models--" + source["model"].replace("/", "--")) / "snapshots" / source["revision"]
        if not (original / "model.safetensors").is_file():
            raise ValueError("selected cross-encoder base weights are unavailable")
        dest = out / "bases" / name
        for file in original.iterdir():
            if file.is_file() and file.suffix in (".json", ".safetensors", ".model", ".txt"):
                copy(file, dest / file.name)
        infer._write(dest / "source.json", source)
        models.append({"name": name, "source": source})
    sel = queries(data, encoder, out / "queries")
    files = {str(p.relative_to(out)): infer._sha(p) for p in out.rglob("*") if p.is_file()}
    manifest = {"version": 1, "kind": "learned-pipeline-inputs", "data_meta_sha256": infer._sha(data / "meta.json"),
                "encoder_sha256": digest, "queries": sel, "models": models, "files": files}
    infer._write(out / "assets.json", manifest)
    return manifest


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=path, default=path("cache/data"))
    p.add_argument("--encoder", type=path, default=path("artifacts/retr-small-final"))
    p.add_argument("--hard", type=path, default=path("artifacts/ce-hard"))
    p.add_argument("--normalizer", type=path, default=path("artifacts/norm2.json"))
    p.add_argument("--out", type=path, required=True)
    a = p.parse_args()
    print(json.dumps(stage(a.data, a.encoder, a.hard, a.normalizer, a.out), indent=2))
