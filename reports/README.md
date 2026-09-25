# evidence index

> sampled retrieval recall pairwise precision and macro f0.5 are different measurements
> only complete full-pool calibration supports a final cutoff claim
> team-reported leaderboard results are labeled separately from offline metrics

## submitted upgraded v1

upgraded v1 received a reported public leaderboard f0.5 of **0.969**
it used the three-retriever pipeline, teammate lightgbm gate, fine-tuned e5 matcher, and provisional cutoff 0.8
both complete output files passed strict validation and the supplied validator with id checking
see [v1 result and output evidence](submission_v1.json)

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
the combined runtime is now implemented and undergoing complete scoring

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
the files did not verify a new 0.97 leaderboard submission for the uploaded ensemble code

## full-pool baseline result

the original two-retriever native-gate baseline completed scoring all 10,320,219 labeled targets
fold 0 selected target-top1 decoding with cutoff 0.5527569055557251
the unchanged fold 1 audit achieved **0.9755015568 source1 macro f0.5**, pair precision 0.9875047126, and pair recall 0.9568847221

this is an offline result for the original baseline
it is neither an upgraded-model result nor a public leaderboard score

- [complete baseline calibration and audit](full_pool_baseline.json)
- [provisional upgraded selection rationale](provisional_selection.json)

the provisional 0.8 cutoff has the highest measured pair f0.5 among the recorded diagnostic operating points
the diagnostic population and metric differ from full source1 macro f0.5
complete upgraded calibration continues separately

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

## final evidence still required

- complete labeled-pool calibration for each candidate configuration
- locked audit with unchanged selection rules
- exact full test coverage and compatible model/configuration hashes
- validated matching and candidate tsvs
- candidate count mean/tails and actual archive hashes
- portal feedback and the updated submission-attempt count
- finalized reproducibility artifacts and selected-model provenance

see [arch](../docs/arch.md), [ops](../docs/ops.md), and [status](../docs/status.md)
