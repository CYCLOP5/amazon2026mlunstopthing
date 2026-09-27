# evaluation and submission evidence

> updated: 2026-09-27
>
> updated planning cutoff: sunday 2026-09-27 21:00 ist / 15:30 utc, based on 18 hours remaining at about 03:08 ist
>
> team-reported public leaderboard f0.5: baseline **0.964**; upgraded v1 **0.969**; learned france-rule submission **0.986416**

## learned pipeline

15 complete-epoch cross-encoders and the task-trained small retriever are selected, totaling 6,515,651,343 parameters
the gate uses 103 features with a 0.001 retention floor, maximum 50 candidates and one fallback
at 07:43 ist, gpu scoring completed all 20,289,808 train/test targets
all 10,320,219 training and 9,969,589 test targets are cached locally with verified coverage and hashes across 106 manifests and 9,950 parts
all 15 member columns are retained: [train verification](../reports/learned-local-train-scores.json) · [test verification](../reports/learned-local-test-scores.json)
the saved test pool contains 14,146,782 candidate pairs, 52.7% fewer than the earlier 29,908,767
validated per-source1 mean / p50 / p95 / p99 / max: 8.1653 / 7 / 15 / 30 / 298
[complete candidate counts](../reports/learned-candidate-counts.json)

the 64-core cpu handoff completed all 64 optuna trials in 216.4 seconds
best search macro f0.5: 0.9905527681, versus 0.9903537691 for the previous winning parameters on the same new scores
selected stack: 101 features, 182 trees; partition2 was excluded from search: [search result](../reports/learned-optuna-search.json)
full-population stack scoring, calibration and both output validators completed
development macro f0.5 improved from 0.9845466096 to 0.9903538925 on identical 73,752 business ids: [comparison](../reports/learned-development-comparison.json)
the archive variant adds bounded france rules: 5,823,095 matches and 14,146,782 actual candidates
france rules change 29,986 reference rows; their isolated score effect is unmeasured because france labels are unavailable
all other countries and the candidate file match the plain learned export: [rule delta](../reports/learned-france-rule-delta.json)

ready archive: `artifacts/submission-learned/Amazites_submission.zip`, 12,393,792,563 bytes
sha256: `db8d69c13dfd0d1584cf4f36b8f1d94afe28f122eba1e831b1c3028949b120eb`
all 196 manifest-listed files, archive crcs, frozen scoring sources and output-file hashes passed verification
strict and official id-enabled validators passed: [release evidence](../reports/submission-learned.json)
team-reported public leaderboard score: **0.986416**, reported 2026-09-27

all gpu scores, four prepared score caches, fitting/evaluation matrices, 64 trial models and the study journal are verified locally
the complete local inventory covers 20,843 files and 30,936,099,215 logical bytes: [cpu cache check](../reports/learned-local-cpu-cache.json)
local score relocation preserves data/model hashes and reproduces identical exports in its check

## cached cpu retuning

the next search uses corrected production tie-breaking and two-fold reference-disjoint calibration inside partition1
the frozen stack scores 0.9904283336 under this stricter search objective; the original search figure is not directly comparable
3 real-data smoke trials passed, including a learned-model family ablation and bundle validation
planned search: 2,560 trials on 3 × 64-core and 1 × 16-core cpu workers; full gpu scores are reused
france currently uses the gate-only fallback, so its score routing and calibration are a separate comparison
[retuning plan](../reports/learned-retune-plan.md) · [baseline diagnostics](../reports/learned-r2-baseline.json)

## model comparison

| variant | retrieval | gate | matcher | measured result |
| --- | --- | --- | --- | --- |
| original baseline | lexical + e5-base + qwen | native lightgbm/catboost mean | fine-tuned e5 pair classifier | **0.964** team-reported public; **0.9755015568** locked offline source1 macro f0.5 |
| upgraded v1 | lexical + e5-base + qwen + e5-large | 54-feature teammate lightgbm | same fine-tuned e5 pair classifier | **0.969** team-reported public leaderboard f0.5 |
| overnight logistic selection | same saved hybrid pool as upgraded v1 | same 54-feature teammate lightgbm | logistic stack over gate and neural logits | **0.9786888883553567** locked offline source1 macro f0.5 |

the baseline offline/public difference is **0.0115015568** across different evaluation populations; it is descriptive and does not identify a cause
the overnight logistic result is an offline train-pool result, not a new public submission score; the team-reported public scores remain 0.964 for the baseline and 0.969 for upgraded v1
the first upgraded upload uses the documented provisional cutoff of 0.8
the baseline comparison export is complete and uses its full-pool selected cutoff of 0.5527569055557251
both baseline output files passed the strict validator and the supplied validator with id checking
its verified file hashes and counts are in [baseline output evidence](../reports/submission_baseline.json)

## completed upgraded v1 output

| measure | value |
| --- | ---: |
| test targets scored | 9,969,589 |
| source1 rows | 1,732,544 |
| accepted target matches | 5,694,959 |
| empty matching rows | 101,247 |
| candidate pairs | 29,908,767 |
| mean candidates per source1 | 17.2629 |
| median | 13 |
| p95 | 36 |
| p99 | 76 |
| maximum | 70,118 |

both the strict validator and the supplied competition validator passed
the supplied validator ran with `--check-ids`
all reference rows and target identifiers were checked, and every accepted match belongs to the candidate set
the long candidate tail is reported explicitly; three candidates per target does not imply three per reference

| file | bytes | sha256 |
| --- | ---: | --- |
| `matching_results.tsv` | 95,834,040 | `3f096c5491b67cd5188466c98d6657b85062e650000ae2692dcee91329f643f1` |
| `candidate_pairs.tsv` | 407,820,759 | `eca9d5a6589f7998341f8d01e30c8bfc5fb88f229a16255e6c2c9da887de240f` |

## training and execution

- full supplied-data eda and label-integrity audit
- entity-grouped folds with separate fitting, selection, and locked-audit populations
- two-epoch e5 pair-classifier training on 502,635 supplied-data pairs
- exact teammate transform, 54-feature, and checkpoint-probability parity on the diagnostic sample
- complete country reference pools for multilingual and lexical retrieval
- original and upgraded test scoring completed with exact target coverage
- 16 disjoint single-a100 assignments for accelerated upgraded test inference
- shared reference caches and 4,418,033 existing target predictions reused at repartitioning
- missing-batch recovery checked with exact pair identity and zero probability difference
- cpu-only aggregation and final tsv generation

## completed overnight full-pool audit

all 10,320,219 train targets passed saved-score hash and coverage verification. eight variants were evaluated without retraining base models or regenerating raw features: six weighted-logit blends, a logistic stack, and a nonlinear lightgbm model. fold0 held-out anchors selected the logistic stack at cutoff `0.8649235367774963`; its tune source1 macro f0.5 was `0.9784026779294054`. selection was transductive because full-pool tune negatives reused fit-side target groups. fold1 remained locked until the one-time audit.

| result | source1 macro f0.5 | cutoff | scope |
| --- | ---: | ---: | --- |
| frozen v1 | 0.9758741644794233 | 0.8 | locked fold1 audit |
| selected logistic stack | 0.9786888883553567 | 0.8649235367774963 | locked fold1 audit |
| postgate perfect-matcher oracle | 0.9955514140602323 | — | fold0 saved postgate pairs |

the oracle remains below 0.998 for the saved postgate candidate set. it is not a global ceiling because changed retrieval or gate candidates can change the set. the json `gate_lost` value is a pre-matcher missing count that combines retrieval misses and gate pruning; the saved final-pair artifact cannot separate those causes. among 763,741 linked targets, 33,646 (4.4%) have blank addresses. those targets account for 8,081/11,080 (72.9%) of pre-matcher missing targets and 3,664/4,373 (83.8%) of wrong-top1 outcomes. the ambiguity audit found zero exact raw-input groups with multiple owners, which does not prove that missing-address cases are intrinsically irresolvable.

the logistic stack's gain was largely present in the simple weighted-logit blend: w=0.6 scored `0.9781291446144351` on fold0 tune, compared with `0.9784026779294054` for the selected stack. these source1 macro results must not be conflated with cached selected-lexical pair diagnostics.

## interpreting the evidence

sampled pair precision, retrieval recall, offline source1 macro f0.5, and public leaderboard f0.5 are distinct measurements
the selected-query cached diagnostics helped compare gates but did not establish test accuracy or source1 macro f0.5
the public result is recorded as reported, without attributing the entire difference to one changed component
the baseline comparison changes retrieval, gate, and selected cutoff together

## remaining evaluation

- keep the selected-query overnight cpu results separate from full-pool evidence
- use validation and leaderboard feedback for the remaining submissions
- finalize selected-model methodology and reproducibility archive, including whether the offline logistic selection is used for a later submission

the cached gate screen covered 17 variants: 13 first-sweep and four extended fits. the safe lightgbm remained best at recall for pair precision ≥0.995. widening blank-address lexical candidates by 7.6% improved retention by 0.32 percentage points but did not improve high-precision recall. these selected-pair results are not full-pool source1 macro scores. see [overnight tuning](../reports/overnight-tune.md), [extended tuning](../reports/overnight-tune-extended.md), [width diagnostic](../reports/overnight-width.md), [full-pool audit](../reports/overnight-fullpool.md), and the [research plan](../plan.md).

see [arch](arch.md), [training](training.md), [reproduction commands](ops.md), and [evidence](../reports/README.md)
