# evidence index

> sampled retrieval recall pairwise precision and macro f0.5 are different measurements
> only complete full-pool calibration supports a final cutoff claim
> no leaderboard score is claimed by this index

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

## implementation and eligibility

| artifact | what it verifies |
| --- | --- |
| [model sources](model_sources.json) | immutable revisions license metadata and parameter bounds |
| [runtime parity](teammate_runtime_parity.json) | exact teammate transforms 54-feature arrays and checkpoint predictions on the parity sample |
| [upgraded smoke](upgraded_runtime_smoke.json) | real three-retriever checkpoint execution and sharded launcher coverage |
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
- final task-resource cleanup and combined budget audit

see [arch](../docs/arch.md), [ops](../docs/ops.md), and [status](../docs/status.md)
