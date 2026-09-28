# evidence index

current trained-model and large checkpoint delivery: [Kaggle version 1](https://www.kaggle.com/datasets/cycl0p5/amazites-ml-2026-reproduction-assets/versions/1), with the [automatic verified download workflow](../submission/docs/reproduce.md) included in the under-1024-mb Unstop zip

> sampled retrieval recall pairwise precision and macro f0.5 are different measurements
> cutoff comparisons retain their declared complete truth and candidate competition
> team-reported leaderboard results are labeled separately from offline metrics

**best recorded public leaderboard f0.5: 0.990285 for final-france** · [result record](final-result.json)

for the submitted execution path, start with the [q&a code map](../submission/docs/qa.md) and [final package receipts](final-packages.json)
the experiment reports below preserve the development history of individual components

our team developed and compared the methods below using fixed candidates, entity-grouped splits and explicit artifact fingerprints

## final evidence chain

| reviewer question | primary record | what it establishes |
| --- | --- | --- |
| which result belongs to the final measured file | [final-result.json](final-result.json) | sprint2's reported public result and variant distinction |
| which exact files were packaged | [final-packages.json](final-packages.json) | per-archive location, size and checksums |
| did the final source regenerate the submitted files | [final-package-replay.json](final-package-replay.json) | raw/prepared replay, output identity and source-style checks |
| are the required inputs preserved independently of downloads | [final-source-custody.json](final-source-custody.json) | verified local release/source custody |
| which final policy is executed | [release.json](../submission/configs/release.json), [final.json](../submission/configs/final.json) | exact country cuts, france blend and positional variant |
| where did the replay scores come from | [replay-inputs.json](../submission/configs/replay-inputs.json) | producer locations, projections, row counts and input hashes |
| how was the final collective model checked | [collective evidence](../submission/configs/collective/evidence.json), [protocol](../submission/configs/collective/protocol.json) | fitting/check scope, selected model and graph workload |
| which training/scoring configuration ran | [execution records](../submission/configs/execution/), [training members](../submission/configs/training/members.json) | worker allocation, selected recipes and score coverage |

the readable explanation is split across [architecture](../submission/docs/arch.md), [compute](../submission/docs/compute.md), [pipeline](../submission/docs/pipeline.md), [research/eda](../submission/docs/research.md), [results](../submission/docs/results.md) and [replay](../submission/docs/reproduce.md)

## measurement vocabulary

| measurement | population or denominator | valid interpretation |
| --- | --- | --- |
| retrieval true-link recall | all true links in the declared query/reference comparison | whether retrieval found the true owner |
| post-gate candidate recall | true links after the actual selection policy | whether the matcher can still see the correct pair |
| candidate macro oracle | original reference truth degrees, predicting only retained true pairs | ceiling for that candidate pool on that evaluated population |
| pair precision/recall | the stated pair or target population | diagnostic trade-off, not automatically reference macro f0.5 |
| search score | the optimizer's declared selection references/calibration protocol | ranking configurations within that search |
| development macro | the stated held/reference-group population with complete incident competitors | a separate development comparison under its exposure assumptions |
| conditional new-head check | new-head-isolated groups over previously developed upstream scores | a head comparison, not pristine end-to-end generalization |
| reported public score | organizer's hidden public subset as reported by the team | result for the identified uploaded file |
| hash/validator/test pass | implementation and artifact contracts | reproducibility or correctness evidence, not new accuracy |

we retain these labels because different rows cannot be ranked as if they were measurements on the same test set

## phase-level outcome ledger

| phase | important observation | next decision |
| --- | --- | --- |
| original/upgraded runtime | public 0.964/0.969 and a provisional pair-selected cutoff | audit the complete target pool with the business-level metric |
| full-pool diagnosis | candidate oracle 0.995551 and concentrated blank-address loss | improve retrieval, gate evidence and candidate survival |
| rich cpu stack | development 0.983963, then 0.984547 after tuning | retain richer raw/corpus features and exact score lineage |
| learned retrieval and pair ensemble | better complete-reference recall; learned development 0.990354 | complete gpu scoring and preserve member outputs for reuse |
| first learned public release | reported 0.986416 with its actual france route | investigate routing/calibration and complementary french evidence |
| broad cached search | trial689 development 0.990555 | retain the result as separate research; avoid assuming a global france fix |
| collective fusion | conditional check 0.992280 vs 0.991336 | use the recorded collective score lineage in the final policy comparison |
| final policy sprint | reported 0.989926 → 0.990108 → 0.990284 | freeze exact country/france policy and preserve the submitted bytes |
| positional derivative | exactly three removed pairs | retain separate identity; no unrecorded accuracy claim |

## evaluation and integrity boundaries

complete truth degrees prevent an unretrieved alias from disappearing from the denominator
complete candidate competition prevents evaluation from changing the winner by removing rival owners
population-dependent context must be constructed before the projection used for a new-head comparison

the inherited run-6 sibling-bank ordering is a documented exception/limitation in the frozen upstream evidence
the later full-pool context and structured new-head protocol address specific scope issues without erasing historical upstream exposure
[additional findings](additional-workspace-findings.md) gives concrete examples

format validation, exact replay and model evaluation are separately recorded
the final code suite's 395 passing tests and the zip/tsv hash checks establish implementation integrity
france remains unlabeled; the final positional edit is team-reported at 0.990285 without a france-only result

## completed historical learned submission

- [release evidence](submission-learned.json): validated archive, output hashes, candidate counts and local-cache inventory
- [development comparison](learned-development-comparison.json): 0.9845466096 to 0.9903538925 macro f0.5 on identical 73,752 businesses
- [development details](learned-development.json): full-target competition and doubled-orphan stress comparisons
- [64-trial search](learned-optuna-search.json): 0.9905527681 search macro f0.5; development references excluded
- [candidate counts](learned-candidate-counts.json): 14,146,782 actual candidates, 52.7% fewer than the previous pool
- [france rule delta](learned-france-rule-delta.json): changed rows restricted to france; identical candidate files
- [local training scores](learned-local-train-scores.json), [local test scores](learned-local-test-scores.json), [local cpu cache](learned-local-cpu-cache.json): complete reusable experiment assets
- [team method review](team-latest-review.md): methods and reproduced evaluation findings
- [2,560-trial finalist comparison](learned-r2-finalists.json): selected development macro f0.5 0.9905552568

the historical learned archive is `artifacts/submission-learned/Amazites_submission.zip`
the bounded france-rule archive contains 5,823,095 matches and received team-reported public 0.986416
its isolated france-rule score effect remains unmeasured; the later sprint2 public result is 0.990284

## reported results and upgraded v1

upgraded v1 received a **team-reported** public leaderboard f0.5 of **0.969**
it used our three-retriever pipeline, validated lightgbm gate, fine-tuned e5 matcher, and provisional cutoff 0.8
both complete output files passed strict validation and the supplied validator with id checking
see [v1 result and output evidence](submission_v1.json)

the original baseline public score is **0.964**, also team-reported
its locked offline audit scored **0.9755015568**, an observed offline/public difference of **0.0115015568** across different evaluation populations; this does not identify a cause

the requested baseline comparison uses its original gate/retrievers and full-pool selected cutoff
that comparison changes multiple pipeline components and cannot isolate a single feature set's causal effect
the baseline output is complete and validated; see [baseline file evidence](submission_baseline.json)

## data and eda

| artifact | evidence |
| --- | --- |
| [eda](eda.json) | supplied-data sizes missingness scripts country and string statistics |
| [compact census](data-census-summary.json) | verified per-file/country counts and complete ownership totals |
| [labels](labels.json) | label and ownership audit |
| [research and eda narrative](../submission/docs/research.md) | current primary-source, data and decision account |
| [historical research notebook](../plan.md) | dated implementation notes and experiments |

raw records caches and detailed local diagnostic rows are retained separately from public reports

## retrieval

| artifact | scope |
| --- | --- |
| [lexical india validation](lex_val_india.json) | selected held-out queries |
| [lexical us validation](lex_val_us.json) | selected held-out queries |
| [e5 india k50](e5_india_k50.json) | frozen multilingual retrieval |
| [e5 india k200](e5_india_k200.json) | wider dense shortlist |
| [e5 us k200](e5_us_k200.json) | us dense retrieval |
| [qwen india k200](qwen_india_k200.json) | complementary identity-instructed retrieval |
| [retrieval union](retrieval_union_india.json) | combined lexical/e5/qwen coverage |
| [large-instruct](e5_large_instruct_india.json) | same india query population and full country ref pool |
| [same-width comparison](e5_large_same_width_india.json) | equal-width replacement and third-retriever comparisons |
| [third-retriever matcher check](third_retriever_end_to_end.json) | controlled sampled comparison through the final matcher |

adding large-instruct recovered five extra positive queries at dense width 100
that establishes a sampled blocking gain rather than a guaranteed full macro-score improvement
the combined runtime completed test scoring with exact target coverage; the overnight full-pool train replay and locked audit are documented below

## matcher and feature experiments

| artifact | scope |
| --- | --- |
| [original tree](tree_baseline.json) | early tree baseline |
| [corrected tree](tree_v2.json) | complete per-candidate field scores |
| [neural pairs](neural_e5_pair.json) | supervised matcher training/validation evidence |
| [gate diagnostics](neural_gate_diagnostics.json) | neural shortlist and blend comparisons |
| [indic features](indic_feature_followup.json) | isolated fixed-candidate india comparison |
| [indic neural intersection](indic_neural_followup.json) | explicitly conditional candidate intersection |
| [team review](team_review.md) | our code/report findings and version differences |
| [validated ensemble](team_safe_ensemble.json) | reference-aggregate sampling shortcut removed |
| [neural gate comparison](team_neural_gate.json) | paired comparison using the same neural scores |

our earlier diagnostics reported 0.9688 validation f0.5
the supplied improvement note referenced an earlier 0.958 leaderboard result
neither is the provenance for the current team-reported 0.969 public score

## full-pool baseline result

the original two-retriever native-gate baseline completed scoring all 10,320,219 labeled targets
fold 0 selected target-top1 decoding with cutoff 0.5527569055557251
the unchanged fold 1 audit achieved **0.9755015568 source1 macro f0.5**, pair precision 0.9875047126, and pair recall 0.9568847221

this is an offline result for the original baseline, not an upgraded-model result
the baseline's separate team-reported public score is 0.964; the observed offline/public difference is 0.0115015568, with no causal attribution

- [complete baseline calibration and audit](full_pool_baseline.json)
- [provisional upgraded selection rationale](provisional_selection.json)

the provisional 0.8 cutoff has the highest measured pair f0.5 among the recorded diagnostic operating points
the diagnostic population and metric differ from full source1 macro f0.5
the completed upgraded full-pool calibration and locked audit are reported in the overnight results below

## completed overnight results

the full-pool replay verified hashes and exact coverage for all 10,320,219 train targets. the frozen v1 locked audit scored 0.9758741644794233 source1 macro f0.5; the fold0-selected logistic stack scored 0.9786888883553567 on the same fold1 audit. this is offline evidence, not a new public score. the fold0 postgate oracle was 0.9955514140602323, below 0.998 for the saved candidate set only; changed retrieval or gate candidates can change that set. [full-pool summary](overnight-fullpool.md) · [full-pool data](overnight-fullpool.json)

the json `gate_lost` field counts linked targets missing before the matcher and combines retrieval misses with gate pruning; saved final pairs do not distinguish those causes. blank addresses were 33,646/763,741 linked targets (4.4%); they accounted for 8,081/11,080 pre-matcher missing targets (72.9%) and 3,664/4,373 wrong-top1 outcomes (83.8%). the ambiguity audit found no exact raw-input duplicate groups with multiple owners, so these errors are not proven intrinsically irresolvable. [ambiguity summary](overnight-ambiguity.md) · [ambiguity data](overnight-ambiguity.json)

the first 13 and extended four cached gate variants did not displace the existing safe lightgbm at recall for pair precision ≥0.995; longer catboost fits got closer. these are selected lexical-pair diagnostics, not source1 macro f0.5. blank10 width added 7.6% candidate pairs for 0.32 percentage points of retention, with no high-precision recall gain. [first tuning summary](overnight-tune.md) · [first tuning data](overnight-tune.json) · [extended summary](overnight-tune-extended.md) · [extended data](overnight-tune-extended.json) · [width summary](overnight-width.md) · [width data](overnight-width.json) · [gap summary](overnight-gap.md) · [gap data](overnight-gap.json) · [cpu tree research](overnight-tree-research.md) · [entity-resolution research](overnight-er-research.md)

## implementation and eligibility

| artifact | what it verifies |
| --- | --- |
| [model sources](model_sources.json) | immutable revisions, license metadata and source identity |
| [runtime parity](team_runtime_parity.json) | exact transforms 54-feature arrays and checkpoint predictions on the parity sample |
| [upgraded smoke](upgraded_runtime_smoke.json) | real three-retriever checkpoint execution and sharded launcher coverage |
| [redistribution recovery](scatter_runtime_check.json) | missing checkpoint batch recomputed with exact pair/probability parity and disjoint coverage |
| [offline runtime](offline_runtime_proof.json) | packaged-cache model loading without network access |
| [license notices](../licenses/readme.md) | upstream notices included with packaged assets |

passing a smoke test establishes executable wiring and output contracts
it does not establish competition accuracy

## final reporting

- best recorded public score: 0.990285 for final-france
- per-variant output hashes distinguish checkpoint results from the final team score
- retained scores support reproducible composition, calibration and france-rule diagnostics

see the [final architecture](../submission/docs/arch.md), [reproduction guide](../submission/docs/reproduce.md), [compute design](../submission/docs/compute.md), [method integration](team-integration.md) and [status](../docs/status.md)
