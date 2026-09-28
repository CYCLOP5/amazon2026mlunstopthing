# additional full-pool findings

the current model/checkpoint release is [Kaggle-hosted](https://www.kaggle.com/datasets/cycl0p5/amazites-ml-2026-reproduction-assets/versions/1); the size-limited submission carries its [automatic checksum-verified fetcher](../submission/docs/reproduce.md)

this report records failure modes found while comparing the team's scored pools and later graph/fusion implementations
it separates a demonstrated implementation or measurement issue from an unmeasured leaderboard effect
the current production stages are described in the [final architecture](../submission/docs/arch.md)

## 1. historical experiment evidence

the compared experiments used the same recorded upstream score configuration as the earlier upgraded baseline
they were separate internal branches, with their own fitting and evaluation scope

| experiment | recorded result | interpretation |
| --- | --- | --- |
| cpu v4 suite, `placid_heart_sb67495985` | control tune 0.988345; audit 0.988643 | enhanced sibling/occupancy and exact-rescue variants did not beat control on tuning |
| alias graph, `amusing_match_phf918vm0c` | selected blend audit 0.988034 vs paired binary 0.988059 | no ranking gain established; candidate-oracle pair recall improved to 0.99209 |
| sibling refinement, `bold_zebra_g8snhj80gj` | audit 0.989091; paired delta +0.001057 | designed after the parent audit, so not a new blind estimate |
| learned edit channel, `plucky_market_6nyd79k5wq` | no eligible tuning improvement; baseline replay | no established end-to-end gain |
| frozen four-view verifier | no tuning improvement | retained the sibling-refinement output |

the graph ranker's historical 0.98075 public result was team-reported; an independently bound public result for the refinement was not available in that review
the separate multilingual branch was reported around 0.984 at that stage

the historical graph/refinement output had 5,893,122 matches and 67,977,154 candidates, averaging 39.2355 candidates per reference
the earlier optuna-stack pool had 29,908,767 candidates, averaging 17.2629
the final released union has 20,177,322 candidates, averaging 11.6461
candidate expansion and matching quality are separate measurements

## 2. missing truth can inflate an apparently perfect score

consider one reference with two true aliases, `t1` and `t2`
suppose only `t1` survives candidate generation and is accepted
the correct reference score is:

```text
tp = 1, fn = 1, fp = 0
f0.5 = 1.25 / (1.25 + 0.25) = 0.833333...
```

if the scorer reconstructs truth only from positive rows still present in the candidate table, it drops `t2` from the denominator and reports 1.0
that rewards the system for failing to retrieve a true alias

complete reference truth degrees must come from the original labels
an in-candidate pair metric may answer a narrower question, but it is not the full reference-set score
this distinction applies to both learned scorers and perfect-matcher candidate oracles

## 3. removing rivals can change the predicted owner

suppose a target's true owner is reference a with score 0.90, while rival b has score 0.95
the full decision selects b
filtering to a before choosing an owner changes that decision into an apparently correct a prediction

the reduced evaluation is now using a different prediction function
all candidate references for an incident target must remain available until the actual ownership rule has run
only then can the chosen reference population be evaluated under its stated scope

## 4. confirmed sibling-context projection issue

the recovered run-6 path is retained under [`submission/src/er/stack/pipeline.py`](../submission/src/er/stack/pipeline.py)

1. `build_round1` computes base-score aggregates over the full pool, then projects to targets reaching the evaluation reference folds
2. `run` passes that projected frame to later group-feature construction
3. `add_groups` builds its confident-owner/sibling bank from the projected population

the earlier base aggregates are not the demonstrated problem
a competing reference outside the evaluation set can lose a confident sibling whose target had no candidate edge to an evaluation reference
the evaluated target's candidate owners may all remain present while their contextual features change

the miniature reproduction showed:

| object | full population | projected population |
| --- | --- | --- |
| target0 candidates | source1-0 and source1-1 | source1-0 and source1-1 |
| target1 | confident source1-1 sibling, no source1-0 candidate | removed |
| source1-1 sibling count for target0 | 1 | 0 |
| source1-0 relative sibling margin | -1 | 0 |

this establishes a feature-scope mismatch
it does not establish its exact score impact or imply that every affected prediction changes in the same direction
the original code, reports and local reproduction were retained with the experiment artifacts

the newer full-pool fusion/collective paths construct their context before new-head reference selection
the inherited run-6 score artifacts remain part of the final lineage with this limitation disclosed

## 5. owner-conditioned aggregates can encode sample selection

a target sample selected because its aliases belong to particular references can make those references look unusually active
candidate counts, top-1 counts or score maxima computed only inside that sample may reveal the selected owners indirectly

excluding literal ids and labels from the model feature list does not remove this risk
the feature's population is still conditioned on the labeled sample construction

in an earlier diagnostic, removing `s1_ncand`, `s1_bmax`, `s1_ntop1` and `s1_gap` reduced apparent precision at a fixed cut from about 99.77% to 98.36%
this demonstrated that the sample-specific feature construction mattered; it was not a newly measured public-score loss

the learned corpus features therefore use the complete raw reference/target population
the newer fusion and collective representations use the complete scored pool before choosing new-head fit/calibration/check references

## 6. tie-breaking is part of the evaluated decision rule

calibration can map different raw scores to the same probability
if production breaks that tie with the candidate model's probability while an optimizer uses an older baseline score, the two paths may choose different owners

the later optuna objective was changed to follow production
its objective version and separate reference-disjoint calibration folds are recorded
older self-calibrated search values are not interchangeable with that corrected objective

the final release also fixes score ties deterministically by reference row id
the incidental order returned by a parquet join is not a decision rule

## 7. feature compatibility is not just array width

the original upgraded gate declared no dense inputs
adding columns to a dataframe could not turn that already-trained checkpoint into a dense-aware gate
the rich feature contract required a new fit and matching metadata

the cached-score stack deliberately excludes original retrieval fields that are not available in its source tables
it preserves ordered feature names and neural member identities instead of silently filling missing fields with placeholders

data, normalization, model, candidate and source identities participate in cache checks
files with the same names or shapes may still represent different records or transformations

## 8. collective context must not be unconditional closure

correlated wrong owners can reinforce one another
the collective implementation therefore constructs edges from text/house-compatible evidence independently of labels and model probabilities

owner hypotheses then compete with each other and an explicit null state
positive reinforcement requires at least two original confident neighbors
reverse-edge contributions are removed from cavity messages, and unsent rival odds remain in the denominator
block size, neighbor degree and message count are bounded

these mechanisms address specific ways a graph can become overconfident
they do not prove every connected component belongs to one business
the learned pair scorer still decides how much the graph representation should matter

## 9. france is still an unlabeled transfer population

score histograms, match volume, model agreement and directional category counts can diagnose behavior in france
they do not directly measure french f0.5

the labeled-country transfer probes also have a limit: the underlying matcher had already been developed on both labeled countries
they evaluate calibration/routing under that condition rather than fully unseen-country neural training

the french rule checks establish deterministic behavior, candidate membership and exact output identity
the isolated accuracy effect of the three-pair positional variant remains unrecorded

## 10. interpreting the later collective result

the collective model's recorded new-head check improved from 0.991336302 to 0.992279682 macro f0.5 on 65,831 references
its normalized-name-group protocol removes the recorded new-head target overlap and retains a historical-upstream-exposure warning

that supports the conditional model comparison
the separately reported public result is sprint2's 0.990284
see the [results ledger](../submission/docs/results.md) and [collective protocol records](../submission/configs/collective/)

## 11. safeguards retained in the final workflow

- source1 truth degrees remain complete when evaluating a subset of candidates
- all rival owners remain present until target-owner selection
- population/sibling feature scope is stated and checked where applicable
- search and production share calibration and tie behavior
- checkpoint metadata declares the ordered feature/text contract
- historically exposed folds remain labeled as such
- france hypotheses are not presented as labeled accuracy
- the candidate file remains the real pre-matcher pool
- raw/scored inputs, selected policies and output hashes remain separately identifiable

## 12. review questions for a new proposed result

1. did missing true aliases disappear from the denominator?
2. were competitors removed before owner selection?
3. were context features built before or after sampling/projection?
4. is the deployed decision rule the rule the optimizer evaluated?
5. does the checkpoint match this exact feature, text and model contract?
6. has an upstream stage already exposed the claimed holdout?
7. does an unseen-country claim actually have labeled evidence?
8. is the candidate export independent of final acceptance?

the final [q&a map](../submission/docs/qa.md), [architecture](../submission/docs/arch.md), [eda/research](../submission/docs/research.md) and [evidence index](README.md) tie these questions to the actual implementation
