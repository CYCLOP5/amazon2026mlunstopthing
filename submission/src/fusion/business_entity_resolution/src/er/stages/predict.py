"stage predict: score every test candidate with the experiment's model, apply the"
import glob
import json
import os
import time

import polars as pl

from er.decision import DECISIONS
from er.io import write_id_lists
from er.models import MODELS


def is_done(cfg, paths, split=None):
    return os.path.exists(os.path.join(paths.output, "matching_results.tsv"))


def _test_feature_parts(cfg, paths):
    if cfg["features"]["cache_test"]:
        for f in sorted(glob.glob(os.path.join(paths.features("test"), "*.parquet"))):
            yield pl.read_parquet(f)
    else:
        from er.stages.features import feature_chunks
        for _, _, feat in feature_chunks(cfg, paths, "test"):
            yield feat


def run(cfg, paths, split=None):
    t0 = time.time()
    metrics = json.load(open(os.path.join(paths.exp, "metrics.json")))
    model = MODELS.get(cfg["model"]["name"]).load(os.path.join(paths.exp, "model"))
    feats, thr = model.feature_names, metrics["threshold"]
    scored_dir = os.path.join(paths.exp, "test_scored")
    os.makedirs(scored_dir, exist_ok=True)
    for f in glob.glob(os.path.join(scored_dir, "*.parquet")):
        os.remove(f)
    n = 0
    for k, feat in enumerate(_test_feature_parts(cfg, paths)):
        p = model.predict(feat.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy())
        (feat.select("s1_id", "o_id").with_columns(pl.Series("p", p, dtype=pl.Float32))
             .write_parquet(os.path.join(scored_dir, f"part_{k:04d}.parquet")))
        n += feat.height
        if k % 5 == 0:
            print(f" scored {n:,} pairs  t={time.time()-t0:.0f}s", flush=True)
    scored = pl.scan_parquet(os.path.join(scored_dir, "*.parquet"))

    rule = DECISIONS.get(metrics.get("decision_rule", cfg["decision"]["rule"]))
    matches = rule(scored.filter(pl.col("p") >= thr).collect(), thr)
    s1_ids = pl.read_parquet(paths.prep_file("test", 1), columns=["entity_id"])["entity_id"]
    os.makedirs(paths.output, exist_ok=True)
    write_id_lists(matches, s1_ids, "matched_entity_ids", os.path.join(paths.output, "matching_results.tsv"))
    write_id_lists(scored.select("s1_id", "o_id"), s1_ids, "candidate_entity_ids",
                   os.path.join(paths.output, "candidate_pairs.tsv"))
    print(f"threshold={thr:.3f}  candidate pairs={n:,}  matches={matches.height:,}  "
          f"S1 with >=1 match: {matches['s1_id'].n_unique():,}/{len(s1_ids):,}  t={time.time()-t0:.0f}s")
