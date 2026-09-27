'materialize two precomputed france blend alternatives'
import argparse as ap
import json
from pathlib import Path as path

import polars as pl

import decode
from sprintfr import blend, load, patterns

plans = {"equal": {"weights": [1, 1, 1, 1], "cut": .7041199803352356},
         "neural": {"weights": [2, 1, 1, 0], "cut": .6099718809127808}}


def run(data, test, scores, pool, out):
    out.mkdir(parents=True, exist_ok=True)
    frame = load(test / "prepared-test.parquet", scores / "test_predictions.parquet")
    refs = pl.scan_parquet(data / "test/ref.parquet").filter(pl.col("co") == "france").select(pl.col("rid").alias("qid"), pl.col("nm").alias("s1_name"), pl.col("ad").alias("s1_addr")).collect()
    targets = pl.concat([pl.scan_parquet(data / "test" / f"s{i}.parquet").select(pl.col("rid").alias("tid"), pl.col("nm").alias("rec_name"), pl.col("ad").alias("rec_addr")) for i in (2, 3)])
    old = pl.scan_parquet(pool / "test_pred.parquet").select("qid", "tid", "p2").join(refs.select("qid").lazy(), on="qid", how="semi").collect(engine="streaming")
    ids = decode.winners(old["qid"].to_numpy(), old["tid"].to_numpy(), old["p2"].to_numpy())
    raw = old[ids].select("qid", "tid").lazy().join(refs.lazy(), on="qid").join(targets, on="tid").collect(engine="streaming")
    pats = patterns(raw).filter(pl.col("swap"))
    counts = pats.group_by("wa", "wb").len("n")
    direction = counts.join(counts.rename({"wa": "wb", "wb": "wa", "n": "rev"}), on=["wa", "wb"], how="left").with_columns(pl.col("rev").fill_null(0))
    bad = direction.filter((pl.col("n") + pl.col("rev") >= 20) & (pl.col("n") / (pl.col("n") + pl.col("rev")) < .7)).select("wa", "wb")
    del old, raw, pats, ids
    rs = {}
    for name, plan in plans.items():
        prob = blend(frame, plan["weights"])
        ids = decode.winners(frame["qid"].to_numpy(), frame["tid"].to_numpy(), prob)
        accepted = frame[ids[prob[ids] >= plan["cut"]]].select("qid", "tid")
        rows = accepted.lazy().join(refs.lazy(), on="qid").join(targets, on="tid").collect(engine="streaming")
        dropped = patterns(rows).filter(pl.col("swap") & pl.col("same_house")).join(bad, on=["wa", "wb"], how="semi").select("qid", "tid")
        accepted = accepted.join(dropped, on=["qid", "tid"], how="anti")
        if accepted["tid"].n_unique() != len(accepted):
            raise ValueError("France variant assigned multiple owners")
        accepted.write_parquet(out / f"{name}.parquet")
        rs[name] = {**plan, "matches": len(accepted), "removed_two_way": len(dropped)}
        print(name, rs[name], flush=True)
    (out / "result.json").write_text(json.dumps(rs, indent=2))


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    for k in ("data", "test", "scores", "pool", "out"):
        p.add_argument("--" + k, type=path, required=True)
    a = p.parse_args()
    run(a.data, a.test, a.scores, a.pool, a.out)
