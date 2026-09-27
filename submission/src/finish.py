'replay the frozen final matching policy from complete cached scores'
import argparse as ap
import hashlib as hh
import json
from pathlib import Path as path

import numpy as np
import polars as pl

legal = ("sarl", "sas", "sa", "sasu", "eurl", "sci", "ei", "snc", "scop", "llc", "inc", "ltd", "pvt", "private", "limited", "co", "corp", "company", "lp", "llp", "pllc", "pc", "plc", "the", "l", "c", "s", "a", "r")


def sha(p):
    with p.open("rb") as f:
        return hh.file_digest(f, "sha256").hexdigest()


def top(q, t, p):
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid probability")
    ix = np.lexsort((q, -p, t))
    return ix[np.r_[True, t[ix][1:] != t[ix][:-1]]] if len(ix) else ix


def tok(c):
    return (pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()
            .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" ")
            .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(legal))).list.unique())


def pat(d):
    d = d.with_columns(tok("tn").alias("a"), tok("rn").alias("b"))
    d = d.with_columns(pl.col("a").list.set_difference("b").alias("extra"), pl.col("b").list.set_difference("a").alias("missing"))
    return d.with_columns((pl.col("extra").list.len().eq(1) & pl.col("missing").list.len().eq(1)).alias("swap"),
                         pl.col("extra").list.first().alias("wa"), pl.col("missing").list.first().alias("wb"),
                         (pl.col("ta").fill_null("").str.extract(r"(\d+)", 1) == pl.col("ra").fill_null("").str.extract(r"(\d+)", 1)).fill_null(False).alias("house"))


def blend(d, weights):
    n, den = np.zeros(len(d)), np.zeros(len(d))
    for col, w in zip(("newest", "friend", "graph", "collective"), weights):
        v = d[col].to_numpy()
        ok = np.isfinite(v)
        n[ok] += v[ok] * w
        den[ok] += w
    return np.divide(n, den, out=np.zeros(len(d)), where=den > 0).astype(np.float32)


def prep(data, dest):
    dest.mkdir(parents=True, exist_ok=False)
    off = 0
    for sr in (1, 2, 3):
        p = data / f"test_source{sr}.tsv"
        d = pl.scan_csv(p, separator="\t", infer_schema=False, row_index_name="rid", row_index_offset=0 if sr == 1 else off)
        d = d.select("rid", pl.col("entity_id").alias("eid"), pl.col("business_name").fill_null("").alias("nm"),
                     pl.col("business_address").fill_null("").alias("ad"), pl.col("country").fill_null("").str.strip_chars().str.to_lowercase().alias("co"))
        target = dest / ("ref.parquet" if sr == 1 else f"s{sr}.parquet")
        d.sink_parquet(target, compression="zstd")
        if sr > 1:
            off += pl.scan_parquet(target).select(pl.len()).collect().item()
    return dest


def write(d, r, t, dest, col, tail=False):
    ids = t.select("tid", "eid").sort("tid").collect(engine="streaming")
    if not np.array_equal(ids["tid"].to_numpy(), np.arange(len(ids), dtype=np.uint32)):
        raise ValueError("target row ids differ")
    order = pl.concat([r.filter(pl.col("co") != "france"), r.filter(pl.col("co") == "france")]) if tail else r
    with dest.open("w") as f:
        for start in range(0, len(order), 100000):
            refs = order.slice(start, 100000)
            part = d.filter(pl.col("qid").is_in(refs["qid"].implode()))
            part = part.with_columns(ids["eid"].gather(part["tid"]).alias("eid_t"))
            g = part.group_by("qid").agg(pl.col("eid_t").unique().sort().str.join(",").alias(col))
            out = refs.select("qid", pl.col("eid").alias("source1_entity_id")).join(g, on="qid", how="left", maintain_order="left", validate="1:1")
            out.select("source1_entity_id", pl.col(col).fill_null("")).write_csv(f, separator="\t", quote_style="never", include_header=start == 0)


def run(data, assets, cfg, out):
    spec = json.loads(cfg.read_text())
    inv = json.loads((assets / "manifest.json").read_text())
    if set(inv["files"]) != {"collective.parquet", "france.parquet", "swap-pool.parquet"}:
        raise ValueError("incomplete replay asset inventory")
    for name, info in inv["files"].items():
        if sha(assets / name) != info["sha256"]:
            raise ValueError("score asset hash mismatch: " + name)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    records = data / "test" if (data / "test/ref.parquet").is_file() else data
    if not (records / "ref.parquet").is_file():
        raw = data / "test" if (data / "test/test_source1.tsv").is_file() else data
        for name, info in spec["data"].items():
            if sha(raw / name) != info["sha256"]:
                raise ValueError("challenge data hash mismatch: " + name)
        records = prep(raw, out / "records")
    else:
        for name, digest in spec["prepared_sha256"].items():
            if sha(records / name) != digest:
                raise ValueError("prepared data hash mismatch: " + name)
    r = pl.read_parquet(records / "ref.parquet", columns=["rid", "eid", "nm", "ad", "co"]).rename({"rid": "qid"}).sort("qid")
    t = pl.concat([pl.scan_parquet(records / f"s{s}.parquet").select(pl.col("rid").alias("tid"), "eid", "nm", "ad", "co") for s in (2, 3)])
    if len(r) != spec["source1_rows"] or not np.array_equal(r["qid"].to_numpy(), np.arange(len(r), dtype=np.uint32)):
        raise ValueError("reference row ids differ")
    scores = pl.read_parquet(assets / "collective.parquet", columns=["qid", "tid", "p"])
    if len(scores) != spec["candidate_pairs"]:
        raise ValueError("candidate coverage differs")
    winner = top(*(scores[c].to_numpy() for c in ("qid", "tid", "p")))
    known = scores[winner].join(r.select("qid", "co"), on="qid", validate="m:1")
    known = known.filter(pl.any_horizontal([(pl.col("co") == co) & (pl.col("p") >= cut) for co, cut in spec["cuts"].items()])).select("qid", "tid")
    del winner
    refs = r.filter(pl.col("co") == "france").select("qid", pl.col("nm").alias("rn"), pl.col("ad").alias("ra"))
    r = r.select("qid", "eid", "co")
    fr_scores = scores.filter(pl.col("qid").is_in(refs["qid"].implode())).select("qid", "tid", pl.col("p").alias("collective"))
    fr = pl.read_parquet(assets / "france.parquet").join(fr_scores, on=["qid", "tid"], how="left", validate="1:1")
    if fr["collective"].null_count():
        raise ValueError("missing collective score")
    p = blend(fr, spec["france"]["weights"])
    ix = top(fr["qid"].to_numpy(), fr["tid"].to_numpy(), p)
    accepted = fr[ix[p[ix] >= spec["france"]["cut"]]].select("qid", "tid")
    del fr, fr_scores, p, ix
    targets = t.select("tid", pl.col("nm").alias("tn"), pl.col("ad").alias("ta"))
    pool = pl.read_parquet(assets / "swap-pool.parquet")
    ix = top(*(pool[c].to_numpy() for c in ("qid", "tid", "p2")))
    raw = pool[ix].select("qid", "tid").lazy().join(refs.lazy(), on="qid").join(targets, on="tid").collect(engine="streaming")
    del pool, ix
    counts = pat(raw).filter(pl.col("swap")).group_by("wa", "wb").len("n")
    del raw
    counts = counts.join(counts.rename({"wa": "wb", "wb": "wa", "n": "rev"}), on=["wa", "wb"], how="left").with_columns(pl.col("rev").fill_null(0))
    bad = counts.filter((pl.col("n") + pl.col("rev") >= 20) & (pl.col("n") / (pl.col("n") + pl.col("rev")) < .7)).select("wa", "wb")
    rows = pat(accepted.lazy().join(refs.lazy(), on="qid").join(targets, on="tid").collect(engine="streaming"))
    drop = rows.filter(pl.col("swap") & pl.col("house")).join(bad, on=["wa", "wb"], how="semi").select("qid", "tid")
    accepted = accepted.join(drop, on=["qid", "tid"], how="anti")
    if spec["france"]["positional_rule"]:
        manual = rows.filter(pl.col("tn").str.contains(r"(?i)\bgroupe\s+(?:sarl|sas|sasu|eurl|sci|snc|scop)\b") &
                             pl.col("rn").str.contains(r"(?i)\b(?:sarl|sas|sasu|eurl|sci|snc|scop)\b") &
                             (pl.col("extra").list.len() == 1) & (pl.col("extra").list.first() == "groupe") & (pl.col("missing").list.len() == 0)).select("qid", "tid")
        accepted = accepted.join(manual, on=["qid", "tid"], how="anti")
    accepted = pl.concat([known, accepted])
    if accepted["tid"].n_unique() != len(accepted) or len(accepted) != spec["matches"]:
        raise ValueError("matching ownership or row count differs")
    del rows, counts, bad, drop
    write(accepted, r, t, out / "matching_results.tsv", "matched_entity_ids")
    write(scores.select("qid", "tid"), r, t, out / "candidate_pairs.tsv", "candidate_entity_ids", tail=True)
    hashes = {name: sha(out / name) for name in ("matching_results.tsv", "candidate_pairs.tsv")}
    if hashes != spec["output_sha256"]:
        raise ValueError("replay output differs from submitted files: " + json.dumps(hashes))
    (out / "replay.json").write_text(json.dumps({"variant": spec["variant"], "sha256": hashes}, indent=2))
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=path, required=True)
    p.add_argument("--assets", type=path, default=path(__file__).resolve().parents[1] / "assets")
    p.add_argument("--config", type=path, default=path(__file__).resolve().parents[1] / "configs/release.json")
    p.add_argument("--out", type=path, required=True)
    a = p.parse_args()
    run(a.data, a.assets, a.config, a.out)
