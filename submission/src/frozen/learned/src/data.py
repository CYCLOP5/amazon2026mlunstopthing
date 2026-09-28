import argparse as ap
import csv
import hashlib as hh
import json
import string as st
import tempfile as tf
import time
import unicodedata as ud
from pathlib import Path as path

import numpy as np
import polars as pl
from anyascii import anyascii


tab = str.maketrans({c: " " for c in st.punctuation})
hdr = ["entity_id", "business_name", "business_address", "country"]


def norm(s):
    return " ".join(ud.normalize("NFKC", s).casefold().translate(tab).split())


def asc(s):
    s = norm(s)
    return norm(anyascii(s)) if not s.isascii() else s


def load(p, sr, off=0):
    d = pl.read_csv(p, separator="\t", infer_schema=False, empty_string_is_null=False)
    if d.columns != hdr:
        raise ValueError(f"invalid source header {p}")
    d = d.select(pl.col("entity_id").alias("eid"), pl.col("business_name").fill_null("").alias("nm"),
                 pl.col("business_address").fill_null("").alias("ad"),
                 pl.col("country").fill_null("").str.strip_chars().str.to_lowercase().alias("co"))
    if d["eid"].n_unique() != len(d) or not d["eid"].str.starts_with(f"S{sr}-").all():
        raise ValueError(f"invalid source ids {p}")
    d = d.with_row_index("rid", offset=off).with_columns(
        pl.lit(sr, dtype=pl.UInt8).alias("sr"),
        pl.col("nm").map_elements(asc, return_dtype=pl.String).alias("nn"),
        pl.col("ad").map_elements(asc, return_dtype=pl.String).alias("an"))
    return d


def save(d, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    d.write_parquet(tmp, compression="zstd")
    tmp.replace(p)


def folds(co, deg, uni, blank, seed=42):
    _, ci = np.unique(co, return_inverse=True)
    z = ci * 32 + np.minimum(deg, 7) * 4 + uni * 2 + blank
    f = np.full(len(co), 2, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for k in np.unique(z):
        ix = np.flatnonzero(z == k)
        rng.shuffle(ix)
        n = len(ix) // 10
        f[ix[:n]] = 0
        f[ix[n:2 * n]] = 1
    return f


def prep(data, dest, seed=42):
    (dest / "meta.json").unlink(missing_ok=True)
    meta = {"seed": seed, "norm": "nfkc casefold punctuation spaces anyascii", "files": {}, "splits": {}}
    for sp in ("train", "test"):
        t = time.monotonic()
        ref = load(data / sp / f"{sp}_source1.tsv", 1)
        nr = len(ref)
        deg = np.zeros(nr, dtype=np.uint32)
        uni = np.zeros(nr, dtype=np.uint8)
        blank = np.zeros(nr, dtype=np.uint8)
        gt = None
        if sp == "train":
            g = pl.read_csv(data / sp / "train_ground_truth.tsv", separator="\t", infer_schema=False,
                            empty_string_is_null=False)
            if g.columns != ["source1_entity_id", "matched_entity_ids"]:
                raise ValueError("invalid truth header")
            g = g.rename({"source1_entity_id": "eid", "matched_entity_ids": "ms"})
            if len(g) != nr or g["eid"].n_unique() != nr:
                raise ValueError("invalid truth coverage")
            g = g.join(ref.select("eid", "rid"), on="eid", how="left", validate="1:1")
            if g["rid"].null_count():
                raise ValueError("unknown truth anchor")
            if g.filter(pl.col("ms").str.contains(r"(^,|,$|,,|\s)")).height:
                raise ValueError("invalid truth id list")
            gt = g.select(pl.col("rid").cast(pl.Int32).alias("own"),
                          pl.col("ms").str.split(",").alias("eid")).explode("eid", empty_as_null=False).filter(pl.col("eid") != "")
            if gt["eid"].n_unique() != len(gt):
                raise ValueError("reused truth target")
            deg = np.bincount(gt["own"].to_numpy(), minlength=nr).astype(np.uint32)
            del g
        off = 0
        linked = 0
        counts = {}
        for sr in (2, 3):
            d = load(data / sp / f"{sp}_source{sr}.tsv", sr, off)
            if gt is not None:
                d = d.join(gt, on="eid", how="left", validate="1:1", maintain_order="left").with_columns(
                    pl.col("own").fill_null(-1).cast(pl.Int32))
                pos = d.filter(pl.col("own") >= 0)
                linked += len(pos)
                cc = pos.select("own", "co").join(
                    ref.select(pl.col("rid").cast(pl.Int32).alias("own"), pl.col("co").alias("qc")),
                    on="own", how="left", validate="m:1")
                if cc.filter(pl.col("co") != pl.col("qc")).height:
                    raise ValueError("cross country truth requires unpartitioned retrieval")
                stats = pos.group_by("own").agg(
                    pl.col("nm").str.contains(r"[^\x00-\x7f]").any().alias("uni"),
                    pl.col("ad").str.strip_chars().eq("").any().alias("blank"))
                ix = stats["own"].to_numpy()
                uni[ix] |= stats["uni"].to_numpy().astype(np.uint8)
                blank[ix] |= stats["blank"].to_numpy().astype(np.uint8)
                del cc, stats, pos
            else:
                d = d.with_columns(pl.lit(-1, dtype=pl.Int32).alias("own"))
            if not np.array_equal(d["rid"].to_numpy(), np.arange(off, off + len(d), dtype=np.uint32)):
                raise ValueError("target row order changed")
            save(d, dest / sp / f"s{sr}.parquet")
            counts[f"s{sr}"] = len(d)
            off += len(d)
            print(sp, f"s{sr}", len(d), "seconds", round(time.monotonic() - t, 1), flush=True)
            del d
        if gt is not None and linked != len(gt):
            raise ValueError("truth target missing from source data")
        f = folds(ref["co"].to_numpy(), deg, uni, blank, seed) if sp == "train" else np.full(nr, 3, np.uint8)
        ref = ref.with_columns(pl.Series("deg", deg), pl.Series("uni", uni),
                               pl.Series("blank", blank), pl.Series("fold", f))
        save(ref, dest / sp / "ref.parquet")
        meta["splits"][sp] = {"refs": nr, "targets": off, "sources": counts, "links": linked,
                              "countries": ref["co"].unique().sort().to_list(),
                              "folds": {str(k): int((f == k).sum()) for k in np.unique(f)}}
        del ref, gt
    for sp in ("train", "test"):
        for p in sorted((data / sp).glob("*.tsv")):
            with p.open("rb") as f:
                sha = hh.file_digest(f, "sha256").hexdigest()
            meta["files"][p.name] = {"bytes": p.stat().st_size, "sha256": sha}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def check():
    assert norm("राम") == "राम"
    assert asc("ACME, Inc") == "acme inc"
    assert asc("École, S.A.S.") == "ecole s a s"
    assert asc("राम") == "ram"
    co = np.array(["us"] * 100 + ["india"] * 100)
    f = folds(co, np.ones(200, dtype=np.uint32), np.zeros(200, np.uint8), np.zeros(200, np.uint8))
    assert list(np.bincount(f)) == [20, 20, 160]
    with tf.TemporaryDirectory() as tmp:
        p = path(tmp)
        for sp in ("train", "test"):
            (p / sp).mkdir()
            for sr in (1, 2, 3):
                rs = [[f"S{sr}-a", "राम", "1 Main Rd", "India"],
                      [f"S{sr}-b", "école", "2 Rue", "France" if sp == "test" else "US"]]
                with (p / sp / f"{sp}_source{sr}.tsv").open("w", encoding="utf-8", newline="") as h:
                    w = csv.writer(h, delimiter="\t")
                    w.writerow(hdr)
                    w.writerows(rs)
        with (p / "train/train_ground_truth.tsv").open("w", encoding="utf-8", newline="") as h:
            w = csv.writer(h, delimiter="\t")
            w.writerows([["source1_entity_id", "matched_entity_ids"], ["S1-a", "S2-a,S3-a"], ["S1-b", ""]])
        m = prep(p, p / "out")
        r = pl.read_parquet(p / "out/train/ref.parquet")
        t = pl.read_parquet(p / "out/train/s3.parquet")
        assert r["deg"].to_list() == [2, 0]
        assert t["own"].to_list() == [0, -1]
        assert t["rid"].to_list() == [2, 3]
        assert m["splits"]["test"]["countries"] == ["france", "india"]
        assert r["nm"][0] == "राम" and r["nn"][0] == "ram"
    print("checks passed")


def main():
    pa = ap.ArgumentParser()
    root = path(__file__).resolve().parents[1]
    pa.add_argument("--data", type=path, default=root / "student_resource/dataset")
    pa.add_argument("--out", type=path, default=root / "cache/data")
    pa.add_argument("--check", action="store_true")
    a = pa.parse_args()
    if a.check:
        check()
    else:
        prep(a.data, a.out)


if __name__ == "__main__":
    main()
