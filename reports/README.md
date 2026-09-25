# evidence index

> sampled retrieval recall pairwise precision and macro f0.5 are different measurements
> only complete full-pool calibration supports a final cutoff claim
> team-reported leaderboard results are labeled separately from offline metrics

## reported results and upgraded v1

upgraded v1 received a **team-reported** public leaderboard f0.5 of **0.969**
it used the three-retriever pipeline, teammate lightgbm gate, fine-tuned e5 matcher, and provisional cutoff 0.8
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
| [labels](labels.json) | label and ownership audit |
| [research plan](../plan.md) | primary sources analysis decisions and experimental history |

raw records caches detailed local probes and teammate diagnostic rows are not public outputs

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
| [teammate review](teammate_review.md) | supplied code/report findings and version differences |
| [safe teammate ensemble](teammate_safe_ensemble.json) | reference-aggregate sampling shortcut removed |
| [teammate neural gate](teammate_neural_gate.json) | paired comparison using the same neural scores |

the teammate diagnostics reported 0.9688 validation f0.5
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
| [model sources](model_sources.json) | immutable revisions license metadata and parameter bounds |
| [runtime parity](teammate_runtime_parity.json) | exact teammate transforms 54-feature arrays and checkpoint predictions on the parity sample |
| [upgraded smoke](upgraded_runtime_smoke.json) | real three-retriever checkpoint execution and sharded launcher coverage |
| [redistribution recovery](scatter_runtime_check.json) | missing checkpoint batch recomputed with exact pair/probability parity and disjoint coverage |
| [offline runtime](offline_runtime_proof.json) | packaged-cache model loading without network access |
| [license notices](../licenses/readme.md) | upstream notices included with packaged assets |

passing a smoke test establishes executable wiring and output contracts
it does not establish competition accuracy

## remaining evidence and release work

- decide whether to submit the offline logistic selection; no public score exists for it
- if submitted, record its exact model/configuration, threshold, full test coverage, and validated matching and candidate tsvs
- portal feedback and the updated submission-attempt count
- finalized reproducibility artifacts and selected-model provenance

see [arch](../docs/arch.md), [ops](../docs/ops.md), and [status](../docs/status.md)
