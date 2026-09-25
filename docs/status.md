# evaluation and submission evidence

> updated: 2026-09-26
>
> deadline: sunday 2026-09-27 08:00 ist / 02:30 utc
>
> team-reported public leaderboard f0.5: baseline **0.964**; upgraded v1 **0.969**

## model comparison

| variant | retrieval | gate | matcher | measured result |
| --- | --- | --- | --- | --- |
| original baseline | lexical + e5-base + qwen | native lightgbm/catboost mean | fine-tuned e5 pair classifier | **0.964** team-reported public; **0.9755015568** locked offline source1 macro f0.5 |
| upgraded v1 | lexical + e5-base + qwen + e5-large | 54-feature teammate lightgbm | same fine-tuned e5 pair classifier | **0.969** team-reported public leaderboard f0.5 |

the baseline offline/public difference is **0.0115015568** across different evaluation populations; it is descriptive and does not identify a cause
the upgraded full-pool calibration and locked audit remain pending
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

## interpreting the evidence

sampled pair precision, retrieval recall, offline source1 macro f0.5, and public leaderboard f0.5 are distinct measurements
the selected-query diagnostic helped choose a provisional operating point; it did not establish test accuracy
the public result is recorded as reported, without attributing the entire difference to one changed component
the baseline comparison changes retrieval, gate, and selected cutoff together

## remaining evaluation

- complete upgraded full-pool calibration and locked audit
- keep the selected-query overnight cpu results separate from full-pool evidence
- use validation and leaderboard feedback for the remaining submissions
- finalize selected-model methodology and reproducibility archive

the overnight cached screen tested 13 tree and fixed-ensemble variants; none improved recall at 0.995 precision over the production-safe teammate gate. the related candidate-gap results are selected-fold lexical diagnostics, not a hybrid full-pool ceiling. see [overnight tuning](../reports/overnight-tune.md), [gap diagnostic](../reports/overnight-gap.md), and the [research plan](../plan.md).

see [arch](arch.md), [training](training.md), [reproduction commands](ops.md), and [evidence](../reports/README.md)
