# extended overnight cpu screening

same cached safe 54-feature lexical selected-query pairs, fold2 owner-disjoint early stop and full-fold2 refit. first-sweep predictions reused, not retrained. fold0 only; no full-pool or leaderboard result.

| gate | top3 retention | recall at precision ≥ .995 | neural top3 blend recall at precision ≥ .995 | seconds | trees |
| --- | ---: | ---: | ---: | ---: | ---: |
| existing_safe_lgb | 0.9600 | 0.9121 | 0.9332 | reused | reused |
| cat_depth4 | 0.9601 | 0.8520 | 0.9235 | reused | reused |
| cat_depth6_bag | 0.9604 | 0.8947 | 0.9275 | reused | reused |
| lgb_hard_neg_2x | 0.9601 | 0.9101 | 0.9318 | 34.48 | 186 |
| lgb_missing_addr_2x | 0.9601 | 0.9095 | 0.9316 | 30.7 | 185 |
| cat_depth4_1400 | 0.9603 | 0.8597 | 0.9238 | 225.19 | 1400 |
| cat_depth6_bag_1400 | 0.9603 | 0.9087 | 0.9323 | 354.2 | 1392 |

**outcome:** catboost depth6 improved from .8947 to .9087 recall at .995 precision (neural blend .9275 to .9323) but remained below the reused safe lgb (.9121; blend .9332). depth4 improved .8520 to .8597 and hit 1400 rounds; it is still budget-limited. both weighted lgb variants also remained below the safe gate. none merits promotion from this sampled screen.

positive-target denominator 13827; candidate recall 0.9648; total 649.89s; peak process rss 2.519 gib.

these precision cutoffs were selected on the same sampled fold0 pairs and are not independently audited. no owner macro or stack is valid on this population. fold1 source1 ids appear only as negative competitors; no fold1 target labels were used.

run from repo root: `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12 .venv/bin/python -u src/tune.py --extended --threads 12 --rounds 1400 > artifacts/overnight-tune-extended/run.log 2>&1`.

parameters, input hashes and all operating points: [json](overnight-tune-extended.json); models and pair predictions: `artifacts/overnight-tune-extended/`.
