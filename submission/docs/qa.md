# final submission q&a and code map

model/checkpoint delivery uses [Kaggle version 1](https://www.kaggle.com/datasets/cycl0p5/amazites-ml-2026-reproduction-assets/versions/1) so the Unstop zip stays below 1024 mb
the included [reproduction commands](reproduce.md) fetch missing files with curl and verify their fixed hashes before inference

the full neural graph totals **8,786,248,209 parameters across 21 distinct checkpoints**
the organizer clarified that **8B is a per-model limit**, not a pipeline sum; our largest checkpoint is **595,776,512 (approximately 0.596B)**
the [model breakdown](models.md#verified-full-neural-parameter-inventory) records the exact counts, repeated-checkpoint reuse and MIT/Apache-2.0 licenses

## 1. what problem did we solve

we link noisy business records from sources2 and 3 to the source1 reference catalog, or abstain when no reference is justified
the inputs provide business name, address and country, plus entity ids and labeled links for training

the task is difficult because records can have spelling changes, alternate names, legal suffixes, domains instead of names, transliterated text, missing addresses and near-identical but genuinely different businesses
france appears in test but not in the labeled training population

the metric is macro f0.5 across source1 businesses
we therefore optimize a high-precision set of links for each business, while retaining full target ownership and complete true-alias denominators during evaluation
see [results](results.md) for the formula and the scope of each measurement

## 2. which source is the final implementation

this integrated project is the reviewer entry point
it contains the learned retrieval/pair-model work, the complementary lexical/graph/hybrid implementations, full score fusion and the actual late-sprint selection scripts

| question | source to inspect |
| --- | --- |
| how do i run the complete bundled model | [`reproduce.py`](../src/reproduce.py) and the [reproduction guide](reproduce.md) |
| how are the trained heads loaded without refitting | [`heads.py`](../src/heads.py) |
| how are original ids and folds prepared | [`neural_v2/src/data.py`](../src/neural_v2/src/data.py) |
| how was the task retriever trained | [`neural_v2/src/retr.py`](../src/neural_v2/src/retr.py) and [`training/retriever.json`](../configs/training/retriever.json) |
| which pair-model runs were selected | [`training/members.json`](../configs/training/members.json) |
| how are hard pairs and cross-encoders fitted | [`ceprep.py`](../src/neural_v2/src/ceprep.py), [`ce.py`](../src/neural_v2/src/ce.py) |
| how are rich gate features built | [`rfeat.py`](../src/neural_v2/src/rfeat.py), [`gfeat.py`](../src/neural_v2/src/gfeat.py) |
| how does the run-6 stack work | [`er/stack/pipeline.py`](../src/er/stack/pipeline.py) |
| how are sibling/graph scores produced | [`graph_resolution`](../src/fusion/business_entity_resolution/graph_resolution) |
| how does bounded hybrid retrieval work | [`final_hybrid`](../src/fusion/business_entity_resolution/final_hybrid) |
| how are score sources combined | [`latest_fusion/pipeline.py`](../src/fusion/business_entity_resolution/latest_fusion/pipeline.py) |
| how does collective message passing work | [`innovation_graph.py`](../src/fusion/business_entity_resolution/latest_fusion/innovation_graph.py) |
| how is the collective model fitted | [`model_innovation.py`](../src/fusion/business_entity_resolution/latest_fusion/model_innovation.py), [`innovation_models.py`](../src/fusion/business_entity_resolution/latest_fusion/innovation_models.py) |
| which code selected the late country/blend settings | [`final_tuning`](../src/final_tuning) |
| which code regenerates the submitted final files | [`finish.py`](../src/finish.py) |

`finish.py` consolidates the frozen release policy into a smaller replay entry point
the actual late-run selection scripts are retained separately
the consolidation was verified against the exact uploaded matching and candidate files, including the final three-pair edit
the complete trained neural/tree checkpoints and preprocessing state are provided by the pinned Kaggle asset set
`reproduce.py cold` executes raw-data inference; `predict` uses the complete upstream feature checkpoint and the trained final model; `replay` uses the original recorded final scores

## 3. why not compare every record with every reference

the test catalog has 1,732,544 references and 9,969,589 targets
the unrestricted cartesian product is on the order of seventeen trillion pairs
our retrieval and filtering stages reduce this to a meaningful candidate pool before expensive matching

the final candidate file contains 20,177,322 pairs
it records the actual pre-matcher union, including rejected matches
the candidate set is not reconstructed from accepted outputs after the fact

## 4. why combine lexical and neural retrieval

lexical views preserve exact names, addresses, rare tokens and numbers
multilingual embeddings provide tolerance to spelling, cross-script and semantic variation
neither alone handles the full distribution well

the task-trained small retriever learned the challenge's alias relationships rather than relying only on general-purpose semantic similarity
the full-reference top-50 comparison improved india true-link recall from 98.43% to 99.73%, and the blank-address india subset from 79.30% to 94.53%

reverse retrieval adds another opportunity for a reference to recover a target
the models still need a learned gate and precise matcher: retrieving a plausible link does not prove it is correct

## 5. why did the original tree sweeps plateau

the original upgraded gate had a fixed 54-feature lexical/string/context contract
although retrieval computed embedding scores, that gate did not learn from dense feature inputs
a search over tree algorithms on the same limited evidence could not recover information that was not represented

the richer implementation added full-corpus name frequencies, idf overlap, house-number edits, generator-aware name changes, dense/rank evidence and independent neural scores
the large later gains are associated with this richer pipeline, not a claim that one generic tree library is always superior

## 6. why change top-3 filtering

a true reference removed by the gate cannot be rescued by a later cross-encoder
the owner-complete gate holdout measured about 99.04% true-link recall with fixed top-3 filtering
the selected adaptive policy retained 99.7743% with a 0.001 floor, minimum one and maximum 50 candidates per target

the neural floor was also set to 0.001 so these recovered pairs received the required model evidence
the cap is per target; it is not a limit on the number of aliases that may point to a reference business

## 7. what was actually trained

the pipeline includes a task-trained contrastive retriever, independently fine-tuned cross-encoder members, learned gate/rich-stack models, a run-6 two-round stack, graph/sibling heads, bounded hybrid residuals, residual fusion and a collective pair scorer

the full cross-encoder experiment varied family, seed, learning rate, length and batch size
fifteen complete-epoch members were selected; an interrupted bge checkpoint was excluded from the scoring ensemble
the final decision sprint reused those scores and fitted/selected cpu policies

model ids and licenses are in [models.md](models.md)
the actual gpu allocation, training matrix and software environments are in [compute.md](compute.md)

## 8. how is the neural ensemble combined

the learned ensemble takes the mean of member logits and applies the sigmoid
it does not take an arithmetic mean of probabilities
member probabilities remain available as separate features to later models

the final france blend is different: it uses an available-score arithmetic probability mean with weights 2, 1, 1 and 16
confusing these two aggregation rules would change the output

## 9. what does the collective graph add

it constructs a reciprocal cross-source target-similarity graph from text and house-compatible evidence
bounded name/address blocks and a reciprocal top-four neighbor rule limit common-name cliques

candidate-owner probabilities compete with one another and an explicit null state
three damped message-passing rounds add owner/null posteriors, entropy, neighbor support and shift features
positive reinforcement needs at least two original confident neighbors, and exact reverse-edge messages are removed from cavity updates

the resulting representation is an input to a new lightgbm pair scorer
the collective mode does not create new reference-target candidate pairs
in the recorded test run, 1,750,828 targets had reciprocal neighbors and the graph used 2,628,882 directed owner messages

the detailed equations, limits and source links are in [architecture](arch.md#13-the-competitive-cavity-graph)

## 10. why is there a separate france policy

france has no labeled training counterpart
the first learned release used gate-based unseen-country handling and did not route france through the full rich stack
the strong labeled-country development score consequently did not establish the quality of that production france route

the final release uses the learned stack, run-6, sibling graph and collective probabilities for france, with explicit missing-score handling
the category-swap rule then uses the original french candidate observation pool

the final positional variant adds a narrow `groupe`-before-legal-suffix filter
it changes three pairs; its independent accuracy effect is not known
the architecture distinguishes these final rules from the earlier, broader france-rule experiments

## 11. how do targets choose owners and when do we abstain

the final release selects the best owner for each target using the complete candidate population
score ties choose the smaller reference row id
india and the us then use their frozen collective-score cuts
france uses its blended score and separate frozen cut before the rule layers

each accepted target has one owner
the complete candidate set remains available even for targets that abstain
empty predictions still produce a source1 output row

earlier experiments compared expected-f0.5 prefix decoding against fixed thresholds
those experiments are part of development history; the final frozen country thresholds are recorded directly in the release configuration

## 12. how were leakage and evaluation handled

initial encoder/pair fitting keeps aliases of a business together
the rich stack uses reference/owner-aware fitting and separate deterministic development partitions
full candidate competition and complete truth degrees are retained for macro evaluation

the newer fusion and collective stages construct their contextual representations over the complete candidate pool before selecting their fitting/calibration/check references
the inherited run-6 code has a documented projection caveat for later sibling context, and its frozen scores are treated as inherited evidence

the collective head's structured protocol separates normalized-name groups and removes calibration/check target overlap
it records 110,186 fitting references, 20,043 clean calibration references and 65,831 check references
the protocol also explicitly records historical upstream exposure

we do not describe historically consulted folds as pristine end-to-end holdouts
the [results guide](results.md) labels each measurement as retrieval, oracle, search, development, conditional head check or reported public score

## 13. what crossed the 0.99 public threshold

the combined input baseline was reported at 0.989926
sprint1 was 0.990108, sprint2 was 0.990284 and final-france was 0.990285
the final sprint reused the combined model evidence, selected country cuts and adjusted the france blend with the existing decoy filter

the 2,560-trial standalone rich-stack winner remained a separate research result rather than the released global replacement
the improvement was not simply another broad search over the original restricted matcher

the public comparisons are whole-release results
we do not infer an isolated causal gain for every neural member, graph feature or french rule from them

## 14. how was azure used

we split gpu representation/pair work from cpu context, fitting, calibration and export
the main full-training envelope requested 62 a100s across a four-gpu gate job, fourteen four-gpu cross-encoder jobs and two single-gpu small-model jobs
models and scored targets were partitioned explicitly rather than sharing mutable caches or assigning work by arrival order

full scoring used 16 disjoint assignments and retained every member score
large-memory e64 nodes handled population features, graph/fusion heads and parallel tree search
e16 nodes handled the final policy and output passes
the local workstation supported eda, compact training experiments and sequential verification

the [compute guide](compute.md) records batching, process/thread allocation, environments, checkpointing and the startup/recovery issues we encountered

## 15. how are candidates and matching quality reported together

both final variants export 20,177,322 actual candidates for 1,732,544 references
the mean is 11.6461 candidates per reference, the median 10, p95 23 and p99 48
25 references have no candidates and still receive output rows

the final matching count is 5,866,303 for sprint2 and 5,866,300 for final-france
the candidate list is identical across those variants
this lets the organizer audit blocking independently of which pairs the final rule accepts

## 16. what proves reproducibility

- original raw/prepared data and recorded score contents are hash-bound
- the exact ordered feature and model configurations are preserved
- final policies are frozen rather than tuned during replay
- both variants reproduce the uploaded matching and candidate hashes
- the final-france check also began from original raw test records
- source-style checks preserved operations, literals and public interfaces
- the fusion suite passed 395 tests with one skipped
- zip manifests and crc checks cover every packaged member

these checks establish implementation and artifact integrity
they are not substitutes for an independent accuracy evaluation

## 17. why include recorded scores in the archive

the original model passes are expensive and can vary numerically if retrained on another platform
recorded scores provide a deterministic path from the supplied raw test records to the exact submitted decisions

the source and detailed reconstruction guide remain included so the retrieval, training, feature and scoring stages can be examined and rerun
the [replay guide](reproduce.md) distinguishes exact release replay, cpu reconstruction from full scored pools and complete neural reconstruction

## 18. which limitations should we state clearly

france is unlabeled, historical development reused some upstream information, and identical or weakly described businesses can remain ambiguous
an oracle over one candidate pool is not a universal ceiling, and a higher search score is not automatically a public-score gain
graph propagation is bounded and competitive because correlated mistakes can reinforce one another

the final code, artifact bindings and measured scopes are preserved so these statements can be checked against the actual implementation
continue with [architecture](arch.md), [pipeline](pipeline.md), [compute](compute.md), [results](results.md) and [reproduction](reproduce.md)
