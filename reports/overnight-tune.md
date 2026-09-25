# overnight cpu tuning — selected-query screening only

fold2 owner-disjoint internal early stopping; refit each tree on all fold2 pairs. fold0 reused across variants. no fold1-owned targets or audit labels used. no full-pool or 99.8% claim.

| variant | top1 recall | top3 recall | recall @ precision ≥ .995 | neural blend recall @ precision ≥ .995 | seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| existing_safe_lgb | 0.9539 | 0.9600 | 0.9121 | 0.9332 | 3.05 |
| lgb_repro | 0.9536 | 0.9599 | 0.9098 | 0.9330 | 27.73 |
| lgb_small_leaf | 0.9542 | 0.9601 | 0.9095 | 0.9314 | 48.0 |
| lgb_regularized | 0.9544 | 0.9600 | 0.9092 | 0.9324 | 58.27 |
| lgb_seed_bag | 0.9539 | 0.9599 | 0.9084 | 0.9314 | 24.16 |
| lgb_extra_splits | 0.9539 | 0.9603 | 0.8895 | 0.9184 | 65.22 |
| cat_depth4 | 0.9517 | 0.9601 | 0.8520 | 0.9235 | 99.81 |
| cat_depth6_bag | 0.9533 | 0.9604 | 0.8947 | 0.9275 | 134.76 |
| extra_trees_diagnostic | 0.9518 | 0.9600 | 0.6816 | 0.9046 | 10.52 |
| lgb_mean | 0.9539 | 0.9601 | 0.9105 | 0.9325 | 0.95 |
| lgb_mean_logit | 0.9539 | 0.9602 | 0.9108 | 0.9327 | — |
| lgb_cat_mean | 0.9531 | 0.9600 | 0.8970 | 0.9296 | 0.93 |
| lgb_cat_mean_logit | 0.9533 | 0.9602 | 0.8996 | 0.9301 | — |

**outcome:** no new tree or fixed ensemble improved the existing safe gate's .995-precision recall, with or without the same cached neural blend. these sampled results do not justify replacing it.

selected positive target denominator: 13827; candidate recall: 0.9648.
internal split: 1188024 fit / 81446 early stop / 588834 unused crossing pairs.
total 477.85s; peak process rss 2.591 GiB; 4 training threads.
budget-limited fits hit the 450-round cap: lgb_extra_splits, cat_depth4, cat_depth6_bag.

validation includes fold1 source1 ids **only as negative competitor references** (see json counts), never fold1-owned targets. no owner macro: the sampled candidate population omits many reference links and distractors. sklearn extra trees is diagnostic only pending model-license eligibility. no learned stacking: the available base fit predictions have no independent level1 meta-fit and separate cutoff/audit owner partitions.

run from the repo root: `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 .venv/bin/python -u src/tune.py --threads 4 --rounds 450 > artifacts/overnight-tune/run.log 2>&1`; small check: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python src/tune.py --check`.

sources and detailed thresholds/parameters: [json](overnight-tune.json); saved models and exact-pair predictions: `artifacts/overnight-tune/`.
