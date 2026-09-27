'Diagnostics for an experiment: why are matches missed / wrongly merged?'
import glob
import json
import os
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from er.blocking.filters import final_candidate_filter
from er.decision import DECISIONS
from er.evaluation import per_entity_fbeta
from er.io import load_gt_pairs
from er.safe import THREADS

N_EXAMPLES = 300
SEED = 11


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=THREADS, dtype=np.float32)


def _tok_overlap(c1, c2):
    return pl.col(c1).str.split(" ").list.set_intersection(pl.col(c2).str.split(" ")) \
             .list.eval(pl.element().filter(pl.element() != "")).list.len()


def _attrs(paths, split, ids1, ids2):
    cols = ["entity_id", "country", "name_core", "name_skel", "addr_can", "addr_skel", "addr_nums",
            "name_nonascii", "addr_missing", "name_alt"]
    s1 = pl.scan_parquet(paths.prep_file(split, 1)).select(cols).filter(pl.col("entity_id").is_in(ids1)).collect()
    o = pl.concat([pl.scan_parquet(paths.prep_file(split, i)).select(cols) for i in (2, 3)]) \
          .filter(pl.col("entity_id").is_in(ids2)).collect()
    return s1, o


def _raw(paths, split, ids):
    R = dict(separator="\t", quote_char=None, infer_schema_length=0)
    return pl.concat([pl.scan_csv(paths.raw(split, i), **R).filter(pl.col("entity_id").is_in(ids)) for i in (1, 2, 3)]) \
             .select("entity_id", "business_name", "business_address").collect()


def _enrich(pairs: pl.DataFrame, paths, split) -> pl.DataFrame:
    'attach raw + normalised text of both records and a few similarity numbers'
    ids1, ids2 = pairs["s1_id"].unique().implode(), pairs["o_id"].unique().implode()
    s1, o = _attrs(paths, split, ids1, ids2)
    raw = _raw(paths, split, pl.concat([pairs["s1_id"], pairs["o_id"]]).unique().implode())
    p = (pairs.join(s1.rename(lambda c: c if c == "entity_id" else c + "_1"), left_on="s1_id", right_on="entity_id", how="left")
              .join(o.rename(lambda c: c if c == "entity_id" else c + "_2"), left_on="o_id", right_on="entity_id", how="left")
              .join(raw.rename({"business_name": "raw_name_1", "business_address": "raw_addr_1"}), left_on="s1_id", right_on="entity_id", how="left")
              .join(raw.rename({"business_name": "raw_name_2", "business_address": "raw_addr_2"}), left_on="o_id", right_on="entity_id", how="left"))
    p = p.with_columns([pl.col(c).fill_null("") for c in p.columns if p[c].dtype == pl.String])
    return p.with_columns(
        pl.Series("name_sim", _cp(p["name_core_1"].to_list(), p["name_core_2"].to_list(), fuzz.token_set_ratio)),
        pl.Series("addr_sim", _cp(p["addr_can_1"].to_list(), p["addr_can_2"].to_list(), fuzz.token_set_ratio)),
        _tok_overlap("name_skel_1", "name_skel_2").alias("shared_name_tokens"),
        _tok_overlap("addr_skel_1", "addr_skel_2").alias("shared_addr_tokens"),
        _tok_overlap("addr_nums_1", "addr_nums_2").alias("shared_numbers"),
    )


def _table(df: pl.DataFrame) -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def blocking_section(cfg, paths, out_dir, lines):
    t0 = time.time()
    gt = load_gt_pairs(paths.raw("train", "gt"))
    found = []
    for f in sorted(glob.glob(os.path.join(paths.blocking("train"), "*.parquet"))):
        c = pl.read_parquet(f).with_columns(pl.col("bscore").max().over("o_id").alias("o_bmax"),
                                            pl.len().over("o_id").alias("o_ncand"),
                                            pl.col("bscore").min().over("o_id").alias("o_bmin"))
        rec = c.select("o_id", "o_bmax", "o_ncand", "o_bmin").unique("o_id")
        hit = c.join(gt, on=["s1_id", "o_id"], how="inner").with_columns(
            final_candidate_filter(cfg["candidates"], c.columns).alias("in_final"))
        found.append((hit.select("s1_id", "o_id", "brank", "bscore", "in_final"), rec))
    hits = pl.concat([h for h, _ in found])
    recs = pl.concat([r for _, r in found])
    g = gt.join(hits, on=["s1_id", "o_id"], how="left").join(recs, on="o_id", how="left")
    g = g.with_columns(
        pl.when(pl.col("in_final")).then(pl.lit("candidate (ok)"))
          .when(pl.col("brank").is_not_null()).then(pl.lit("in top-k but removed by relative-score filter"))
          .when(pl.col("o_ncand").is_null()).then(pl.lit("record got NO candidates at all"))
          .otherwise(pl.lit("record has candidates, true owner not in top-k")).alias("status"))
    s1c = pl.scan_parquet(paths.prep_file("train", 1)).select(pl.col("entity_id").alias("s1_id"), "country").collect()
    g = g.join(s1c, on="s1_id", how="left").with_columns(pl.col("o_id").str.slice(0, 2).alias("source"))
    lines += ["## 1. Blocking: where do true pairs end up?", "",
              _table(g.group_by("status").agg(pl.len().alias("pairs")).with_columns(
                  (pl.col("pairs") / g.height).alias("share")).sort("pairs", descending=True)), "",
              "By country / source (share of true pairs NOT in the final candidate set):", "",
              _table(g.group_by("country", "source").agg((pl.col("status") != "candidate (ok)").mean().alias("miss_rate"),
                                                         pl.len().alias("true_pairs")).sort("country", "source")), ""]
    miss = g.filter(pl.col("status") != "candidate (ok)")
    sample = miss.sample(min(20_000, miss.height), seed=SEED)
    e = _enrich(sample.select("s1_id", "o_id", "status", "brank", "bscore", "o_bmax", "o_bmin", "o_ncand"), paths, "train")
    e = e.with_columns(
        pl.when((pl.col("shared_name_tokens") == 0) & (pl.col("shared_addr_tokens") == 0)).then(pl.lit("shares no name and no address token"))
          .when(pl.col("addr_missing_2") | (pl.col("addr_can_2") == "")).then(pl.lit("other record has no address"))
          .when(pl.col("shared_name_tokens") == 0).then(pl.lit("no shared name token (address only)"))
          .when(pl.col("shared_addr_tokens") == 0).then(pl.lit("no shared address token (name only)"))
          .otherwise(pl.lit("shares name AND address tokens but outranked")).alias("cause"))
    lines += [f"Causes (sample of {e.height:,} missed pairs):", "",
              _table(e.group_by("cause").agg(pl.len().alias("n"), pl.col("name_nonascii_2").mean().alias("nonascii_name2"),
                                             pl.col("shared_numbers").gt(0).mean().alias("share_a_number"),
                                             pl.col("o_ncand").mean().alias("avg_candidates")).sort("n", descending=True)), "",
              "Similarity quantiles of missed pairs (token-set 0-100):", "",
              _table(e.select([pl.col(c).quantile(q).alias(f"{c}_q{int(q*100)}") for c in ("name_sim", "addr_sim")
                               for q in (0.25, 0.5, 0.75)])), ""]
    keep = ["cause", "status", "country_1", "s1_id", "o_id", "raw_name_1", "raw_name_2", "raw_addr_1", "raw_addr_2",
            "name_core_1", "name_core_2", "addr_can_1", "addr_can_2", "name_sim", "addr_sim", "shared_name_tokens",
            "shared_addr_tokens", "shared_numbers", "brank", "bscore", "o_bmax", "o_ncand"]
    e.sample(min(N_EXAMPLES, e.height), seed=SEED).select(keep).write_csv(os.path.join(out_dir, "blocking_misses.csv"))
    print(f" blocking section done {time.time()-t0:.0f}s", flush=True)


def validation_section(cfg, paths, out_dir, lines, metrics):
    t0 = time.time()
    va = pl.read_parquet(os.path.join(paths.exp, "val_pred.parquet"))
    thr = metrics["threshold"]
    pred = DECISIONS.get(cfg["decision"]["rule"])(va, thr).select("s1_id", "o_id").with_columns(pl.lit(True).alias("pred"))
    v = va.join(pred, on=["s1_id", "o_id"], how="left").with_columns(pl.col("pred").fill_null(False))
    s1c = pl.scan_parquet(paths.prep_file("train", 1)).select(pl.col("entity_id").alias("s1_id"), "country").collect()
    v = v.join(s1c, on="s1_id", how="left").with_columns(pl.col("o_id").str.slice(0, 2).alias("source"))
    v = v.with_columns(pl.when(pl.col("pred") & (pl.col("label") == 1)).then(pl.lit("TP"))
                         .when(pl.col("pred")).then(pl.lit("FP"))
                         .when(pl.col("label") == 1).then(pl.lit("FN")).otherwise(pl.lit("TN")).alias("kind"))
    by = v.group_by("country", "source").agg(
        (pl.col("kind") == "TP").sum().alias("TP"), (pl.col("kind") == "FP").sum().alias("FP"),
        (pl.col("kind") == "FN").sum().alias("FN")).with_columns(
        (pl.col("TP") / (pl.col("TP") + pl.col("FP"))).alias("precision"),
        (pl.col("TP") / (pl.col("TP") + pl.col("FN"))).alias("recall_in_cands")).sort("country", "source")

    tr = cfg["training"]
    s1_val = (pl.scan_parquet(paths.prep_file("train", 1)).select(pl.col("entity_id").alias("s1_id"), "country")
                .filter(((pl.col("s1_id").hash(seed=tr["hash_seed"]) % 100) >= tr["train_pct"])
                        & ((pl.col("s1_id").hash(seed=tr["hash_seed"]) % 100) < tr["val_pct"])).collect())
    truth = load_gt_pairs(paths.raw("train", "gt")).filter(pl.col("s1_id").is_in(s1_val["s1_id"].implode()))
    ent = per_entity_fbeta(pred.select("s1_id", "o_id"), truth, s1_val["s1_id"]).join(s1_val, on="s1_id")
    ent = ent.with_columns(pl.when(pl.col("n_true") == 0).then(pl.lit("singleton")).otherwise(pl.lit("has matches")).alias("type"))
    lines += ["## 2. Model errors on validation", "",
              f"threshold {thr}, validation macro F0.5 {metrics['val_f05']:.4f}, oracle {metrics['oracle_f05']:.4f}", "",
              "Pair-level (within candidates):", "", _table(by), "",
              "Entity-level:", "",
              _table(ent.group_by("country", "type").agg(pl.len().alias("entities"), pl.col("f").mean().alias("mean_F05"),
                                                         (pl.col("f") == 1).mean().alias("perfect"),
                                                         (pl.col("n_pred") > pl.col("tp")).mean().alias("has_wrong_merge"),
                                                         (pl.col("tp") < pl.col("n_true")).mean().alias("misses_some"))
                     .sort("country", "type")), "",
              "Probability of errors (quantiles):", "",
              _table(v.filter(pl.col("kind").is_in(["FP", "FN"])).group_by("kind").agg(
                  [pl.col("p").quantile(q).alias(f"p_q{int(q*100)}") for q in (0.1, 0.25, 0.5, 0.75, 0.9)])), ""]
    for kind, fname in (("FP", "val_false_positives.csv"), ("FN", "val_false_negatives.csv")):
        s = v.filter(pl.col("kind") == kind)
        s = s.sample(min(N_EXAMPLES, s.height), seed=SEED).select("s1_id", "o_id", "p", "label")
        e = _enrich(s, paths, "train")

        if kind == "FP":
            gt = load_gt_pairs(paths.raw("train", "gt"))
            e = e.join(gt.rename({"s1_id": "true_owner"}), on="o_id", how="left")
        cols = [c for c in ["s1_id", "o_id", "p", "true_owner", "country_1", "raw_name_1", "raw_name_2", "raw_addr_1",
                            "raw_addr_2", "name_core_1", "name_core_2", "addr_can_1", "addr_can_2", "name_sim",
                            "addr_sim", "shared_numbers"] if c in e.columns]
        e.select(cols).write_csv(os.path.join(out_dir, fname))
    print(f" validation section done {time.time()-t0:.0f}s", flush=True)


def test_section(cfg, paths, lines, metrics):
    t0 = time.time()
    files = glob.glob(os.path.join(paths.exp, "test_scored", "*.parquet"))
    if not files:
        lines += ["## 3. Test", "", "(no test_scored/ - run predict first)", ""]
        return
    sc = pl.read_parquet(files)
    thr = metrics["threshold"]
    pred = DECISIONS.get(cfg["decision"]["rule"])(sc.filter(pl.col("p") >= thr), thr)
    s1c = pl.read_parquet(paths.prep_file("test", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    per = (s1c.join(pred.group_by("s1_id").agg(pl.len().alias("n")), on="s1_id", how="left").fill_null(0)
              .group_by("country").agg(pl.len().alias("S1"), (pl.col("n") > 0).mean().alias("share_with_match"),
                                       pl.col("n").mean().alias("matches_per_S1")))
    oc = pl.concat([pl.read_parquet(paths.prep_file("test", i), columns=["entity_id", "country"]) for i in (2, 3)]) \
           .group_by("country").agg(pl.len().alias("S2S3_records"))
    best = sc.group_by("o_id").agg(pl.col("p").max().alias("pmax"), pl.col("s1_id").first())
    best = best.join(s1c, on="s1_id", how="left")
    dist = best.group_by("country").agg([pl.col("pmax").quantile(q).alias(f"best_p_q{int(q*100)}") for q in (0.1, 0.25, 0.5, 0.75)]
                                        + [(pl.col("pmax") >= thr).mean().alias("records_assigned")])
    lines += ["## 3. Test predictions by country", "",
              _table(per.join(oc, on="country").join(dist, on="country").sort("country")), "",
              "(Compare France with India/US: a much lower share_with_match or best-probability distribution "
              "means the model is unsure on the unseen country.)", ""]
    print(f" test section done {time.time()-t0:.0f}s", flush=True)


def run(cfg, paths):
    out_dir = os.path.join(paths.exp, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    metrics = json.load(open(os.path.join(paths.exp, "metrics.json")))
    lines = [f"# Diagnostics - experiment `{paths.exp_name}` (model {metrics['model']})", "",
             f"validation macro F0.5 **{metrics['val_f05']:.4f}**, oracle {metrics['oracle_f05']:.4f}, "
             f"threshold {metrics['threshold']}", ""]
    for name, fn in (("blocking", lambda: blocking_section(cfg, paths, out_dir, lines)),
                     ("validation", lambda: validation_section(cfg, paths, out_dir, lines, metrics)),
                     ("test", lambda: test_section(cfg, paths, lines, metrics))):
        try:
            fn()
        except Exception as ex:
            lines += [f"## {name}: FAILED - {type(ex).__name__}: {ex}", ""]
            print(f" {name} section failed: {ex}", flush=True)
    imp = metrics.get("importance", [])[:25]
    lines += ["## 4. Feature importance (top 25)", "", _table(pl.DataFrame(imp, schema=["feature", "gain"], orient="row")) if imp else "", ""]
    open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"wrote {out_dir}")
