# additional full-pool findings

## evidence

completed team experiments by raj mathuria were inspected through their saved code and reports. they use the same upstream score configuration `36eb4123836f6a1d46c23e4df2b500a7b932fabd252ed024314ff6917df29dd2` as the cached upgraded baseline. this recovered implementation is distinct from the supplied `amazon-ml-challenge-hmm` checkout

| experiment | recorded result | interpretation |
| --- | --- | --- |
| cpu v4 suite, `placid_heart_sb67495985` | control selected at tune 0.988345; audit 0.988643 | enhanced sibling/occupancy and exact-rescue variants did not beat control on tuning |
| alias graph, `amusing_match_phf918vm0c` | audit selected blend 0.988034 vs paired binary 0.988059 | no ranking gain established; candidate oracle pair recall improved to 0.99209 |
| sibling refinement, `bold_zebra_g8snhj80gj` | audit 0.989091; paired delta +0.001057 | the follow-up was designed after the parent audit; this is not a blind estimate |
| learned edit channel, `plucky_market_6nyd79k5wq` | no eligible tuning improvement; baseline replay | added edit-channel models did not establish an end-to-end gain |

the recovered handoff also records a completed four-view verifier with no tuning improvement. it retained the sibling-refinement output. the graph ranker's 0.98075 public result was user-reported in that handoff; no independently bound public result was available for the refinement. these scores must not be conflated with the separate teammate checkout's reported 0.984

the graph/refinement output contains 5,893,122 matches and 67,977,154 candidates, averaging 39.2355 candidates per source1. the current optuna-stack checkpoint uses 29,908,767 candidates, averaging 17.2629. wider candidate coverage and final matching quality are separate measurements

## confirmed context-projection issue

in the recovered `business_entity_resolution/src/er/stack/pipeline.py`:

1. `build_round1`, lines 90–96, computes base-score aggregates over the full pool, then keeps only target records that can reach the evaluation reference folds
2. `run`, lines 285–289, passes that projected table to `add_groups`
3. `add_groups`, lines 182–192, builds its confident-owner bank from the projected table

base-score aggregates are computed before projection. the problem is the later sibling bank: a competing reference outside the evaluation set can lose confidently matched siblings whose targets did not touch an evaluation reference. its group evidence and the evaluation reference's competition margins then differ from their full-population values

a miniature reproduction using the recovered functions confirmed this. target0 has candidates source1-0 and source1-1; target1 is a confident source1-1 sibling and has no source1-0 candidate. projecting to targets incident to source1-0 removes target1. source1-1's sibling count for target0 falls from 1 to 0, and source1-0's sibling margin changes from -1 to 0

this establishes a feature-scope mismatch, not a measured amount of score inflation. its score effect requires a controlled rerun. build the confident-owner bank and sibling context from the complete population before projection, or retain the full graph until those features are finalized

the cached optuna experiment projects targets only after its raw-name frequency features have been built over the complete raw corpus. its remaining pair features are local. the complete-competitor cache must retain that property if graph or score-context features are added

## useful next directions

- preserve full-population source1 competition and alias evidence; do not derive it from owner-conditioned query samples
- improve learned candidate recall for blank-address records rather than copying failed hard number/sibling rules
- evaluate unseen-country behavior explicitly; higher french uncertainty is drift evidence, not labeled accuracy
- do not rerun the failed edit-channel or frozen four-view branches merely because their model names sound promising

local evidence is preserved under `artifacts/teammate-analysis/`, including the immutable code snapshot, original reports, stdout and `sibling-projection-check.json`
