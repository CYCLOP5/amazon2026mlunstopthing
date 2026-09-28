# results, submission history and evaluation scope

## 1. the metric we optimized

the challenge scores each source1 business against its complete set of true source2/source3 aliases
for one reference:

```text
f0.5 = 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
```

the final score is the mean across reference businesses
false positives are relatively expensive, so accepting every similar-looking alias is a poor strategy
references with empty truth and empty predictions still matter to the macro denominator

pair precision/recall, pair f0.5, candidate recall, a candidate oracle, a development macro score and a public leaderboard score answer different questions
the records below retain those distinctions

## 2. released artifacts and public results

| release | reported public macro f0.5 | accepted pairs | final candidates |
| --- | ---: | ---: | ---: |
| sprint2 | 0.990284 | 5,866,303 | 20,177,322 |
| final-france positional variant | 0.990285 | 5,866,300 | 20,177,322 |

both files contain all 1,732,544 required source1 rows
the final-france variant removes three accepted pairs and changes no candidate pairs

| artifact | authoritative matching-file hash record |
| --- | --- |
| sprint2 | [`release.json`](../configs/release.json), `output_sha256.matching_results.tsv` |
| final-france | [`final.json`](../configs/final.json), `output_sha256.matching_results.tsv` |

the frozen configuration files and archive manifests are authoritative for the complete hashes
the common candidate hash is `344f114f8c81d4c806cf7019a2ebefc6cb312e16ca7ec04797701ee36a9e5e71`

## 3. reported submission progression

### 3.1 early baseline and complementary branches

the early public baseline was reported at 0.964
that stage established a working lexical/dense retrieval, learned gate and neural pair scorer, but it still lost useful candidates and used a restricted matcher feature set
the first upgraded-v1 submission was reported at 0.969
its provisional 0.8 cutoff had been selected on a sampled pair diagnostic, which prompted the later full-target, per-business calibration audit

other development branches produced reported results around 0.984 and later 0.987
the recorded large-cross-encoder change documents a 0.984806 to 0.985292 improvement
the later 0.987 report was not bound to an exact artifact in that comparison
these historical values describe complementary branch development, not a single controlled sequence of otherwise-identical submissions

the useful lesson from these branches was architectural: task-specific retrieval, independent neural evidence, richer features and country-specific decisions mattered more than repeatedly tuning the same limited feature set

### 3.2 the first learned full-scoring release

the learned-only archive had 5,823,095 accepted pairs and 14,146,782 candidates
its recorded public result was **0.986416**
on the labeled-country development population, the corresponding learned stack had reached **0.990354**

this was a concrete reminder that a stronger labeled-country model does not automatically transfer to an unlabeled country or to the exact production route
the first learned release routed france through gate-based unseen-country handling rather than the full rich score stack
subsequent investigation therefore focused on routing/calibration and the additional french evidence, rather than assuming another global tree search would fix france

### 3.3 combined input baseline

the late combined source package was reported at **0.989926**
it brought together the learned stage, run-6, graph/sibling, bounded hybrid, residual fusion and collective-model work
it became the concrete starting point for the final policy sprint

### 3.4 sprint1

sprint1 was reported at **0.990108**
the final sprint reused existing model probabilities, compared india/us country cuts and evaluated france blends with the existing category-swap filter
the confirmed score is 0.990108; no higher value is used for this checkpoint

### 3.5 sprint2

sprint2 was reported at **0.990284**
its india/us cuts are frozen at 0.9310117959976196 and 0.9249221086502075
its france available-score blend uses weights 2, 1, 1, 16 for learned stack, run-6, sibling graph and collective score
the france cut is 0.8345136046409607

relative to the reported combined input baseline, the public gain is 0.000358 in score units
relative to sprint1, it is 0.000176
these whole-release comparisons do not isolate a causal contribution for every underlying model

### 3.6 final-france positional edit

the final positional variant removes three pairs where an added `groupe` immediately precedes a legal suffix while preserving after-suffix cases
the edit is implemented as a deterministic rule in the release replay, and exact file equality was checked
the team-reported public macro f0.5 is 0.990285, which is 0.000001 above sprint2
no separate france-only evaluation was reported

## 4. early full-pool diagnosis

the early diagnostic replay covered all 10,320,219 training targets
target ownership was chosen across all candidate references before the evaluation anchors were selected

| method | one-time fold-1 macro f0.5 |
| --- | ---: |
| frozen v1 configuration | 0.975874 |
| selected logistic gate/neural stack | 0.978689 |

the selected logistic stack was developed on fold 0 before the one-time audit
its source variables were gate/neural logits, so the improvement was largely a decision/calibration correction over the same saved candidate pool

the fold-0 perfect-matcher oracle on that saved post-gate pool was 0.995551 macro f0.5, with pair recall 0.985492
this was a ceiling for that candidate set, not for all possible retrieval methods

### error concentration

| target address | linked targets | missing before matcher | wrong top-1 |
| --- | ---: | ---: | ---: |
| blank | 33,646 | 8,081 | 3,664 |
| nonblank | 730,095 | 2,999 | 709 |

blank-address targets were about 4.4% of linked targets in this diagnostic slice but contributed 72.9% of missing pre-matcher links and 83.8% of wrong-top-1 outcomes
that observation motivated better retrieval and ambiguity evidence for name-only records
the saved post-gate data could not separate initial retrieval loss from gate pruning, so later tooling recorded those causes separately

## 5. richer features before larger searches

the early cpu sweep tested thirteen tree/ensemble alternatives over the then-available feature set
none beat the established high-precision gate at the tested operating point
this did not rule out a tree matcher with better inputs

the later raw-text/full-corpus pair stack was compared on the same 73,752 development reference businesses:

| method | macro f0.5 | pair precision | pair recall |
| --- | ---: | ---: | ---: |
| v1 configuration | 0.976006 | 0.994191 | 0.941895 |
| richer pairwise stack | 0.983963 | 0.996034 | 0.961168 |
| selected 64-trial optuna stack | 0.984547 | 0.996217 | 0.962220 |

the search itself improved from 0.984374 to 0.985203 on its different search population
the development comparison, not the larger search number, is the appropriate side-by-side check in the table

## 6. learned retrieval and candidate survival

complete-reference top-50 true-link recall improved from 98.43% to 99.73% in india and from 99.39% to 99.67% in the us
the blank-address india subset improved from 79.30% to 94.53%

the gate-budget comparison then found another avoidable loss:

| selection | true-link recall in the gate holdout |
| --- | ---: |
| fixed top 3 | about 99.04% |
| selected adaptive floor/cap | 99.7743% |

the adaptive policy used a 0.001 floor, minimum one and maximum 50 candidates per target
mean selected candidates per sampled query was 1.3544
its owner-complete candidate-only oracle was 0.999308 macro f0.5

these figures establish better candidate access, not a final precision or leaderboard result

## 7. learned full-scoring comparison

the complete learned score pass retained all fifteen member probabilities and covered every train/test target
the separate development comparison used the same 73,752 anchors and 254,763 true links:

| checkpoint | macro f0.5 | pair precision | pair recall |
| --- | ---: | ---: | ---: |
| preceding tuned stack | 0.984546610 | 0.996216508 | 0.962219789 |
| learned retrieval/ensemble stack | 0.990353893 | 0.998561366 | 0.972649090 |

the scope is fold-0 held-out reference development with full target competition and transductive target reuse
it is not a new fold-1 audit and contains no labeled france population

the learned 64-trial search reached 0.990553 on its search partition, versus approximately 0.990354 for the prior settings on those new scores
those search values should not be substituted for the development or public results

## 8. the larger cpu search

the later search completed 2,560 trials using cached neural evidence
the selected finalist, trial 689, scored 0.990555257 on the reserved development comparison versus 0.990353893 for the preceding learned stack
the gain was 0.000201364

that experiment also corrected a tie-breaking mismatch so calibrated ties used the candidate model probability consistently with production
its objective used separate reference-disjoint calibration folds

the finalist was not promoted into a new global optuna-only submitted file
the final released decisions instead use the combined score sources and the recorded final country/france policy

## 9. graph, hybrid and collective evidence

the earlier graph/refinement branch improved candidate coverage, but not every graph or verifier experiment improved the selected matching rule
the learned edit-channel and frozen four-view verifier did not establish an eligible tuning improvement
the sibling refinement's reported audit gain followed examination of the parent audit, so it is historical evidence rather than a fresh blind estimate

the final collective model's recorded new-head check used 65,831 references:

| head | macro f0.5 | pair precision | pair recall |
| --- | ---: | ---: | ---: |
| incumbent | 0.991336302 | 0.999162818 | 0.974661814 |
| collective graph model | 0.992279682 | 0.999353550 | 0.977397161 |

its separate clean calibration population had 20,043 references
the selected model strength was 1.0
the protocol excludes target overlap for the new head and records historical upstream exposure explicitly
see [`configs/collective/`](../configs/collective/)

these conditional head measurements support the model-development decision
the final-france release's public result is the separately team-reported 0.990285

## 10. france transfer evidence and its limits

the early gate-plus-density unseen-country route performed poorly in labeled-country transfer proxies, around 0.961 macro f0.5
using richer stack scores with empirical source calibration was materially better in those proxies

the underlying matcher had still been developed on both labeled countries
those probes therefore test calibration/routing transfer under that condition; they are not labeled france accuracy measurements or fully unseen-country neural training experiments

the final french category-swap and positional rules are text/pool-derived hypotheses
rule correctness checks establish deterministic behavior and candidate membership, not that every dropped pair is truly negative

## 11. what worked and what did not establish a gain

| observation | response or outcome |
| --- | --- |
| provisional pairwise cut did not match the business-level objective | recalibrate using complete-reference macro f0.5 |
| blank-address misses dominated the early diagnostic | improve task retrieval, reverse evidence and ambiguity features |
| more tree trials over restricted evidence plateaued | add dense, corpus and generator-aware features |
| fixed top-3 filtering hid useful candidates | adopt adaptive candidate survival and score the recovered pairs |
| historical sibling projection altered rival context | document the mismatch and build newer full-pool context before selection |
| learned-country gains did not transfer automatically to the public mixture | inspect the actual france route and score calibration |
| edit-channel/four-view branches did not win on tuning | retain their reports without making them mandatory release stages |
| repeated gpu scoring was expensive in elapsed time | freeze member scores for cpu-only model and policy iteration |
| the final policy moved the public score above 0.99 | preserve exact inputs, policy, output hashes and the actual selection code |

## 12. verification evidence

the released matching and candidate files passed strict and supplied-validator checks with ids enabled
both variants reproduced their submitted output hashes exactly
the final-france replay was also checked from the original raw test files

all zip members were checked against their manifests and crcs
the integrated fusion suite passed 395 tests with one skipped
source-style checks compared operations, literals and public signatures for 325 imported and 37 existing python files

these checks establish implementation/reproduction integrity
they do not turn historical development folds, candidate oracles or unlabeled france policies into independent test accuracy measurements

continue with [architecture](arch.md), [compute](compute.md), [full pipeline](pipeline.md) and [reproduction](reproduce.md) for the implementation behind these results
