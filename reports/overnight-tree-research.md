# overnight cpu tree research for entity matching

**conclusion:** our team retained the validated gate and target-top1/threshold decoder as controls. the completed cached 13-variant cpu screen did not improve recall at 0.995 precision; it is a selected-query diagnostic, not a full-pool result. further regularization or blend work needs the same candidates and complete-pool evaluation. neither ranking loss nor extra trees can repair missing retrieved links. the 0.998 source1 macro f0.5 target was not a measured result of these changes. [challenge] [baseline] [overnight-tune] [lgb-params] [stack]

**question:** which cpu-only lightgbm, catboost, extra-trees, bagging, and stacking changes might improve this challenge without mistaking pair accuracy for set quality?

**scope:** the repository's supplied records and existing feature/prediction artifacts; official library documentation and original lightgbm/catboost papers. the cached cpu screen below is an executed selected-query diagnostic; remaining experiments are proposals. no cloud jobs or outside business-identity data were used.

**date:** 2026-09-26.

## observed baseline and constraints

- each s1 business has zero to many true s2/s3 links. scoring averages per-s1 f0.5, including a score of **1** for a correctly empty singleton and **0** for a singleton with any predicted link. for nonempty truth, `f0.5 = 1.25*tp/(1.25*tp + fp + 0.25*fn)`. a false positive enters the denominator with four times the coefficient of a false negative; this is **not** a claim that changing a binary loss weight by four will optimize the metric. [challenge] [plan]
- prepared fold `2` fits models; fold `0` selects the decoder/cutoff; fold `1` is the locked audit. the original baseline scored **all 10,320,219 labeled targets** and obtained **0.9755015568** source1 macro f0.5 on fold 1 at the unchanged fold-0 target-top1 cutoff `0.5527569056` (pair precision `0.98750`, recall `0.95688`). the team-reported baseline public score is **0.964**, an observed offline/public difference of **0.0115015568** across different populations. upgraded v1's **0.969** public score is also team-reported; no upgraded full-pool locked-audit score is recorded here. neither difference identifies a cause. the early tree's sampled-query `0.986` estimate had only `0.580` pair precision at its selected cutoff and is not full-pool evidence. [baseline] [baseline-public] [v1] [early-tree] [infer]
- installed/locked: lightgbm `4.7.0`, catboost `1.2.10`, scikit-learn `1.9.1`. `src/train.py` fits numeric-feature `LGBMClassifier` (`800`-round cap, rate `0.05`, `31` leaves, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.9`) or catboost (`depth=6`, rate `0.05`, logloss); both stop on sampled fold-0 binary logloss. the native gate already averages lightgbm/catboost probabilities; the deployed upgrade instead uses a 54-feature, 248-tree lightgbm gate. `src/match.py` joins lexical and frozen dense retrieval, gates each **target** to at most three references in the deployed configuration, then blends gate/neural logits (`0.4/0.6`); it saves component probabilities. its generic `--k-gate` default is `20`, not the deployed `3`. [train] [match] [arch] [lock]
- local `cache/models/v2/features/*.npz`, `cache/models/v2/validation_predictions.parquet`, and the corresponding lexical run parts are present. those validation predictions contain *selected-query* pairs, not a complete distractor pool. the published full-pool baseline artifact contains **aggregate** calibration/audit metrics, not enough per-pair scores to re-optimize a cutoff from that json alone. verify availability and matching fingerprints of inference `parts/*.parquet` before proposing a score-only full-pool re-decode. [train] [baseline] [index]
- the native gate already sees retrieved hard competitors, field similarity, missingness, digit agreement, source, rank/gaps, and reference-name/address frequency; the neural fit already uses retrieved hard negatives plus a small random remainder. the supervised tree fit does not inject missed retrieval positives. our no-s1-aggregate comparison removed four reference-side features that revealed sampled-owner selection. restoring them on sampled queries would inflate evidence. on a selected-query diagnostic, the gate's top three retained `0.9840` of positive owners: better downstream trees cannot recover those it drops, but that figure is not a full-pool macro bound. [features] [neural] [review] [gate]

## native and production gates use different bagging settings

the native `src/train.py` lightgbm wrapper sets `subsample=0.8` but leaves `subsample_freq` at its default `0`, so that code path does not perform row bagging. our deployed upgraded gate is a separate 54-feature checkpoint with active `bagging_freq=1`. the new lightgbm variants also use `bagging_freq=1` and vary other parameters or seeds; this is not an on/off bagging ablation because the deployed control already uses active bagging. [train] [gate-metadata] [overnight-code] [overnight-tune]

the completed cached screen compared **13 variants**. none improved the production-safe gate's recall at **0.995 precision**, either alone or with the same cached neural blend. this result applies only to the selected fold-0 cached lexical-candidate sample; it is not a full-pool or hybrid-retrieval candidate ceiling. [overnight-tune] [overnight-tune-json]

## what the primary sources support — and do not

| option | documented mechanism | challenge-specific inference / starting test, not an established gain |
| --- | --- | --- |
| lightgbm binary | `LGBMClassifier` defaults to binary logloss for two classes; leaf-wise trees need `num_leaves`, `min_data_in_leaf`, and possibly `max_depth` to control overfit. `subsample=0.8` has no row-bagging effect in the native wrapper when `subsample_freq=0`; setting `subsample_freq=1` activates it. `colsample_bytree=0.9` already samples features. `extra_trees=true` tests one random split threshold per feature/node. [lgb-api] [lgb-params] [lgb-tune] | our controlled comparisons use identical features/candidates and vary tree size, leaf size, regularization and sampling separately. the deployed control already uses `bagging_freq=1`. proposed starting points are not measured optima. |
| catboost binary | `Logloss` handles two classes. `depth`, `l2_leaf_reg`, `rsm`, and `bootstrap_type` control complexity/randomness; cpu supports `rsm`; `subsample` applies with bernoulli or mvs, **not** bayesian bootstrap. cpu defaults can already use mvs, so do not claim catboost has no sampling today. the catboost paper's categorical-feature advantage does not imply a gain on this numeric-only matrix. [cat-common] [cat-boot] [cat-paper] | on the same rows try `depth=4` versus existing `6`, `l2_leaf_reg=3` versus `10`, then one `bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8` variant; keep `loss_function="Logloss"`, `learning_rate=0.05`, cap `iterations=800`, early-stop patience `50`. compare elapsed cpu time and **held-out** precision/recall as well as macro score. |
| extremely randomized trees / bagging | sklearn `ExtraTreesClassifier` randomizes split thresholds and averages tree probabilities; defaults (`min_samples_leaf=1`, unbounded depth) can grow very large. sklearn `BaggingClassifier` samples **rows**, not entity groups; its ordinary out-of-bag score is not this challenge's macro set metric. lightgbm has cheaper in-model bagging and `extra_trees` controls. [extra] [bag] [lgb-params] | only on a bounded, **whole-target-group** sample, test `ExtraTreesClassifier(n_estimators=128, max_depth=14, min_samples_leaf=20, max_features=0.5, bootstrap=False, n_jobs=4, random_state=42)` against one lightgbm `extra_trees=True` fit. score identical validation pairs; keep neither if memory, latency, or false merges rise. see license caveat below. |
| ranking objective | lightgbm `lambdarank`/`rank_xendcg` and catboost `PairLogit`/`YetiRank` optimize ordered relevance within groups, not binary probabilities or per-s1 f0.5. [lgb-params] [cat-rank] | a target-group ranker *might* help choose its one true owner among candidates, but cannot alone decide that an unlinked target matches nobody; grouping by s1 would also have to allow multiple true links. retain a separately tuned abstain/set decoder. not a first experiment. |

the lightgbm paper introduces gradient-based sampling and exclusive-feature bundling; the catboost paper introduces ordered boosting and categorical handling. **neither paper demonstrates 0.998 on this dataset**. [lgb-paper] [cat-paper]

**cpu/runtime:** lightgbm `n_jobs`/`num_threads` and catboost `thread_count` control native training threads; sklearn extra trees parallelize over trees. on the documented 12-logical-thread local machine, start with **one fit at a time**, `n_jobs=4` or `thread_count=4`, and bounded feature-generation workers; compare wall time and peak memory against the existing 8-thread tree baseline before increasing parallelism. avoid `n_jobs=4` across four concurrent fits *plus* 4-thread native/blas work; sklearn documents this oversubscription trap. `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=4` are *optional profiling settings*, not a proven faster default. adding a 128-tree gate would score **every pre-gate retrieved pair**, not just the post-gate exported pairs; memory/latency may dominate tiny score gains. [lgb-params] [cat-perf] [extra] [parallel] [match] [plan]

## negatives, imbalance, calibration, and stacking

1. **fix the evaluation population before tuning the loss.** the tree receives all retrieved negatives in the saved fit runs, including plausible rival refs; a target with `own=-1` is genuinely unlinked, while a candidate with `qid != own` for a linked target is a wrong owner. measure false positives into otherwise unrelated/singleton s1 refs across **all targets**, not only the selected fold's aliases. inspect errors separately for same-name/different-address, same-address/different-name, blank address, common names, and cross-script aliases. if fit is needed later, change the *mix or modest sample weights* of these existing fit-fold negatives, rather than copying sampled-query reference aggregates or mining tune/audit labels into fit. an injected true owner missing from retrieval does not make that owner available to the final gate. [train] [neural] [review] [infer]
2. **do not equate an imbalanced pair dataset with the objective's costs.** start with unweighted binary logloss and tune a decoder on deployment-like candidates. test a mild positive/anchor weighting variant only against the unchanged candidate universe and recall/false-merge slices; avoid blindly setting positive weight to `neg/pos`. lightgbm warns `is_unbalance`/`scale_pos_weight` distort individual probability estimates; catboost says not to combine `auto_class_weights`, `class_weights`, and `scale_pos_weight`. pair prevalence changes with negative mining and gate width. [lgb-api] [cat-common] [challenge]
3. **calibration is a different experiment from ranking.** fit a sigmoid calibrator on predictions from anchors absent from the base fit, then choose the threshold on *different* anchors against the full target pool; reserve fold 1 for a single locked audit. isotonic can introduce ties and overfit smaller sets. a strictly monotone sigmoid cannot improve the best attainable score of an exhaustive threshold sweep on the *same* fixed scored pairs; its plausible value is probability interpretation, transferable thresholds, or a subsequent blend. threshold `0.5` and pair `accuracy` are not evidence. [cal] [extra] [infer]
4. **a cheap blend precedes a learned stack.** join the saved native lightgbm/catboost predictions by `(tid,qid)` and compare each alone with their **existing mean**, at identical candidate coverage. for a meta-model, either (a) use frozen base models trained only on fold 2 to score the *complete* fold-0 target pool, split fold-0 s1 anchors into disjoint meta-fit and cutoff-selection partitions, then evaluate fold 1 once; or (b) build true out-of-fold base predictions entirely within fold 2 using group-disjoint ref/owner folds, train a `LogisticRegression(C=0.1 or 1, solver="lbfgs", max_iter=500)` on clipped base-model logits, tune on fold 0, audit fold 1. keep all aliases of each owner together; exclude held-out refs from base fit even as negative `qid`s; group orphan target candidates by `tid`. freeze fitted transforms and candidate construction within each fit partition. sklearn's default `StackingClassifier` uses row-wise stratified folds for binary labels; `cv="prefit"` trained on the same rows risks overfit. the existing neural scorer was trained on fold 2, so **its fold-2 predictions are not oof**; use it only via the disjoint fold-0 option unless new, separately authorized neural oof predictions exist. an upstream top-3 gate means a downstream stack cannot resurrect excluded pairs. [stack] [groups] [leak] [match]

## follow-up experiments

the 13-variant cached screen above is complete. these remaining tests are proposals; none establishes full-pool quality without complete-target evaluation.

| order | reuse / one controlled change | decision gate |
| --- | --- | --- |
| 0, no fit | join `cache/models/v2/validation_predictions.parquet` on `(tid,qid)`; count **disagreeing** true links/false merges for lgb, cat, and existing mean at common cutoffs. use `validation_anchors.parquet` only as a sampled-query diagnostic. [train] [early-tree] | if error sets almost coincide, skip stacking and extra trees. do not promote from the sampled metric. |
| 1, bounded cpu fit | reuse `cache/models/v2/features/*.npz` and saved lexical run ids for any further leaf/regularization or catboost variants; fixed train/validation rows, internal **fold-2** group holdout for early stopping. [train] [lgb-tune] [cat-common] | first require improved positive-owner recall at fixed high pair precision and candidate width; then require complete-target macro f0.5 and no fold-1 regression before promotion. |
| 2, optional bounded cpu fit | extra trees on at most `200,000` fit pairs sampled by whole `tid` groups versus lightgbm `extra_trees=True` at the same validation pairs. [extra] [lgb-params] | stop if latency/memory rises or its errors add no complementary links; extra-trees deployment eligibility is unresolved. |
| 3, only if full-pool part scores are already available | threshold/abstention comparison on existing exact-coverage scored parts; then a two-score logistic blend via disjoint fold-0 meta-fit/cutoff selection, if component scores exist for **the same final candidates**. [match] [infer] [cal] | choose on complete fold 0, audit once on fold 1; re-export only with the exact model/configuration and actual final pre-matcher candidate set. |

**promotion bar:** report full-target source1 macro f0.5, pair precision/recall, correctly empty s1s and false merges into them, retrieval and post-gate positive coverage, per-country/source/degree/blank-address slices, candidate count per s1 (including tails), and cpu peak memory/wall time. any 0.998 claim requires independent complete-pool evidence and cannot be inferred from higher pair accuracy or selected-query recall. test france has no labeled train analogue; a us/india audit is not a france bound. [challenge] [plan] [index]

## uncertainty and assumptions

- numeric settings in the follow-up table are **proposed starting points**, not measured improvements. the 13-variant overnight cpu screen is the only local experiment reported here; no full-pool per-pair rescore was run. cached sampled `.npz` files and model predictions exist, but a complete scored-parts archive for a proposed *new* model has not been verified locally. higher gate accuracy may fail to move end-to-end f0.5 when retrieval or the neural scorer dominates. [review] [match]
- the rules require a final mit/apache-2.0 model. lightgbm's source is mit and catboost's apache-2.0; sklearn's **software** is bsd-3-clause. software license and trained-model licensing are distinct; organizer treatment of a sklearn extra-trees *final component* is unverified. treat sklearn extra trees as diagnostic only unless eligibility is confirmed; lightgbm `extra_trees` is a first-party alternative. [challenge] [lgb-license] [cat-license] [sk-license]

## sources

[challenge]: ../Ml_Challenge.txt
[plan]: ../plan.md
[index]: README.md
[baseline]: full_pool_baseline.json
[baseline-public]: submission_baseline.json
[v1]: submission_v1.json
[early-tree]: tree_baseline.json
[gate]: neural_gate_diagnostics.json
[review]: team_review.md
[lock]: ../uv.lock
[train]: ../src/train.py
[gate-metadata]: ../artifacts/upgraded-gate/metadata.json
[overnight-code]: ../src/tune.py
[overnight-tune]: overnight-tune.md
[overnight-tune-json]: overnight-tune.json
[features]: ../src/feat.py
[neural]: ../src/neural.py
[match]: ../src/match.py
[infer]: ../src/infer.py
[arch]: ../docs/arch.md
[lgb-api]: https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html
[lgb-params]: https://lightgbm.readthedocs.io/en/stable/Parameters.html
[lgb-tune]: https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
[lgb-paper]: https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
[cat-common]: https://catboost.ai/docs/en/references/training-parameters/common
[cat-perf]: https://catboost.ai/docs/en/references/training-parameters/performance
[cat-boot]: https://catboost.ai/docs/en/concepts/algorithm-main-stages_bootstrap-options
[cat-rank]: https://catboost.ai/docs/en/concepts/loss-functions-ranking
[cat-paper]: https://arxiv.org/abs/1706.09516
[extra]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
[bag]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
[stack]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
[groups]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
[leak]: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
[cal]: https://scikit-learn.org/stable/modules/calibration.html
[parallel]: https://scikit-learn.org/stable/computing/parallelism.html#oversubscription-spawning-too-many-threads
[lgb-license]: https://raw.githubusercontent.com/microsoft/LightGBM/master/LICENSE
[cat-license]: https://raw.githubusercontent.com/catboost/catboost/master/LICENSE
[sk-license]: https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/COPYING
