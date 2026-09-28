# release status and checkpoint evidence

## 1. final release state

the integrated source is under [`submission/`](../submission/README.md) on `final-submission-packages`
the best recorded public result is **0.990285 macro f0.5 for final-france**
the final-france variant removes three pairs relative to sprint2's reported 0.990284

| release item | state | evidence |
| --- | --- | --- |
| actual combined source and final selection scripts | included | [q&a/code map](../submission/docs/qa.md) |
| frozen country/france policies | included | [release configuration](../submission/configs/release.json), [positional variant](../submission/configs/final.json) |
| exact submitted matching files | preserved and replayed | [replay evidence](../reports/final-package-replay.json) |
| actual candidate pool | preserved identically for both variants | [release counts](../reports/final-result.json) |
| strict and supplied-validator checks | passed for the release outputs | [package records](../reports/final-packages.json) |
| archive members, hashes and crcs | verified when each archive is built | [package receipts](../reports/final-packages.json) |
| source-style integrity | operations, literals and public interfaces compared | [replay/source evidence](../reports/final-package-replay.json) |
| local source/output custody | recorded independently of downloads | [custody record](../reports/final-source-custody.json) |

the written documentation identifies the actual execution path and the measurement scope of earlier experiments
full trained bundles and full score/feature histories are separate from the compact exact-replay package

## 2. final file contract

| quantity | sprint2 | final-france |
| --- | ---: | ---: |
| required source1 rows | 1,732,544 | 1,732,544 |
| accepted pairs | 5,866,303 | 5,866,300 |
| candidate pairs | 20,177,322 | 20,177,322 |
| mean candidates per reference | 11.6461 | 11.6461 |
| candidate p50 / p95 / p99 | 10 / 23 / 48 | 10 / 23 / 48 |
| candidate maximum | 9,807 | 9,807 |

the candidate file includes rejected pairs and retains every required reference row
one accepted target has one reference owner
the per-target gate cap and per-reference candidate counts describe different directions of the matching graph

## 3. checkpoint and public-result ledger

| checkpoint | known public result | other principal evidence | scope |
| --- | ---: | --- | --- |
| original baseline | 0.964 | locked full-target audit 0.975501557 | different public/offline populations |
| upgraded v1 | 0.969 | complete target scoring and valid output; provisional pairwise-selected cut | early production checkpoint |
| overnight logistic choice | no new public result bound here | one-time fold-1 macro 0.978688888 vs v1 0.975874164 | same saved candidate pool |
| richer/tuned stack | no separately bound public result in these receipts | development 0.984546610 on 73,752 references | cpu improvements over cached upstream scores |
| first learned archive | 0.986416 | development 0.990353893, complete member scores and adaptive candidate pool | labeled-country development plus separate public observation |
| 2,560-trial finalist | no new global public submission | development 0.990555257 | separate standalone retuning experiment |
| combined input baseline | 0.989926 | supplied combined score/source lineage | starting artifact for the late policy sprint |
| sprint1 | 0.990108 | new country/france decision selection | reported late-sprint result |
| sprint2 | 0.990284 | exact final policy and output hashes | released measured artifact |
| final-france | 0.990285 | exact three-pair positional-rule difference | team-reported final submission |

the branch development reports also contain historical 0.984/0.987 observations and a documented 0.984806 to 0.985292 large-model comparison
an exact 0.987 artifact was not bound in that historical review
see the [complete results narrative](../submission/docs/results.md)

## 4. learned component completion

the learned component selected fifteen complete-epoch cross-encoders and a task-trained small retriever
its gate uses the 103-feature contract, a 0.001 floor, a maximum of 50 candidates per target and one fallback candidate when retrieval produced any candidates

the completed gpu scoring covers:

- 10,320,219 training targets
- 9,969,589 test targets
- 20,289,808 total targets
- 106 run manifests and 9,950 score parts across train/test verification
- all `np_m0` through `np_m14` probability columns

the corresponding learned-only test pool contains 14,146,782 pairs
its mean / p50 / p95 / p99 / maximum per reference are 8.1653 / 7 / 15 / 30 / 298
the smaller learned-only pool and the later combined pool are distinct artifacts

[train coverage](../reports/learned-local-train-scores.json) · [test coverage](../reports/learned-local-test-scores.json) · [learned candidate counts](../reports/learned-candidate-counts.json)

## 5. search and development records

the learned 64-trial handoff selected a 101-feature, 182-tree stack
its best search score was 0.9905527681, versus 0.9903537691 for the prior settings on the new score population
the separate development comparison improved from 0.9845466096 to 0.9903538925 on identical 73,752 business ids

the later search used production-consistent tie-breaking and two reference-disjoint calibration folds
it completed 2,560 trials across three 64-core workers and one 16-core worker
four finalists were frozen before their development comparison
trial 689's development score was 0.9905552568, a gain of 0.0002013643 over the preceding learned stack

this later finalist was not the released global replacement
the final sprint used the combined source probabilities and its separately recorded country/france policy

[learned search](../reports/learned-optuna-search.json) · [development comparison](../reports/learned-development-comparison.json) · [wide-search finalists](../reports/learned-r2-finalists.json)

## 6. france status and interpretation

the first learned public artifact used the gate-based unseen-country route
its bounded france-rule export changed 29,986 reference rows relative to the plain learned export, with india/us and candidates preserved
the isolated accuracy effect of those earlier rules was not measured

later labeled-country transfer probes supported richer score routing and empirical calibration, but they did not measure france accuracy
the final combined release uses the full score blend described in the [architecture](../submission/docs/arch.md#15-final-country-decisions)

the final positional derivative removes exactly three pairs
the team's final submission records a public score of 0.990285; a france-only score was not reported

[early rule delta](../reports/learned-france-rule-delta.json) · [transfer proxy](../reports/learned-country-transfer.json) · [earlier full-stack france variant](../reports/learned-france-stack-variant.json)

## 7. implementation checks that support these records

the release checks cover ids, row coverage, unique target ownership, accepted-pair membership, candidate counts and output hashes
source-style comparison covers 325 imported and 37 existing python files
the integrated fusion suite passed 395 tests with one skipped

the final replay has been verified against both matching variants and the shared candidate file
the final-france verification also started from the original raw challenge test files
these are implementation/integrity checks, not additional accuracy estimates

## 8. what would require new work

| change | required dependent work |
| --- | --- |
| written documentation or source packaging | rebuild the package manifest and verify the new zip; submitted tsv hashes can remain fixed |
| final threshold or france rule | rerun the final decision/export stage and record a new output identity |
| cpu tree/fusion head | refit and rescore the full compatible pool, recalibrate and reevaluate |
| feature definition or normalization | rebuild affected feature caches and fit compatible heads; regenerate neural scores if serialized model input changes |
| neural member, pooling or ensemble order | regenerate affected pair scores and downstream stack/calibration artifacts |
| retriever or gate candidate policy | regenerate candidates, score newly retained pairs and repeat the downstream checks |

no development, oracle or search number is automatically a result for such a new artifact
the full evidence chain is in [architecture](../submission/docs/arch.md), [compute](../submission/docs/compute.md), [pipeline](../submission/docs/pipeline.md), [research/eda](../submission/docs/research.md) and the [evidence index](../reports/README.md)
