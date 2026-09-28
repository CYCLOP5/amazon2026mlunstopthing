# final system architecture

## 1. what this architecture implements

our system resolves noisy records from sources2 and 3 against a source1 business catalog
it must distinguish an alias from a different business, preserve every required reference row, and provide the actual candidate set used before the final decision

the released solution is a staged ensemble
lexical and neural retrieval create plausible links; learned gates and pair models score them; complementary stacks and graph representations add context; a frozen country policy chooses owners and abstains
the exact release implementation is [`src/finish.py`](../src/finish.py)
the source trees that produced its score inputs are included alongside it

the recorded public result for sprint2 is **0.990284 macro f0.5**
the final-france submission is team-reported at **0.990285**, with the same candidate pool and three fewer accepted pairs
the final-france variant changes three accepted pairs and is team-reported at 0.990285
see [results and measurement scope](results.md)

## 2. end-to-end data flow

```mermaid
flowchart TD
    raw[original train and test records] --> ids[stable row ids and comparison views]
    ids --> old[earlier lexical and multilingual retrieval]
    ids --> learned[task-trained small retriever and reverse retrieval]
    learned --> gate[dense and name-aware gate]
    gate --> ce[cross-encoder member scores]
    ce --> stack[rich learned pair stack]
    old --> run6[run-6 lexical and sibling stack]
    run6 --> siblings[graph rank and sibling refinement]
    siblings --> hybrid[bounded hybrid retrieval and residual model]
    stack --> union[full score union and population features]
    run6 --> union
    siblings --> union
    hybrid --> union
    union --> residual[residual score fusion]
    residual --> graph[competitive cavity graph features]
    graph --> head[collective pair reranker]
    head --> known[india and us owner selection and cuts]
    head --> france[france available-score blend]
    stack --> france
    run6 --> france
    siblings --> france
    france --> swap[directional category-swap filter]
    swap --> positional[optional final-france positional filter]
    known --> export[external ids and matching tsv]
    positional --> export
    union --> candidates[complete candidate tsv]
    export --> verify[expected hashes and submission checks]
    candidates --> verify
```

the graph has two distinct roles in this diagram
the earlier graph/sibling branch supplies a complementary score and an expanded candidate pool
the later collective graph supplies a label-free representation to a new pair scorer without removing or inventing reference-target candidates

## 3. record model and identity contract

| field | meaning | use |
| --- | --- | --- |
| `eid` | original challenge entity id | restored only for output and identity checks |
| `rid` | stable row number within a prepared record table | compact array and join index |
| `qid` | source1 reference row number | candidate owner id |
| `tid` | source2 row number followed by offset source3 rows | globally unique target id within a split |
| `nm`, `ad`, `co` | original name, address and normalized country key | text views, features and routing |
| `own` | labeled target's true reference, or orphan marker | training and evaluation only |
| `deg` | complete true-alias count for a reference | evaluation denominator only |
| `fold` | original entity-grouped split | stage-specific training/evaluation selection |

`qid` is a historical column name for the reference side
a retrieval query is a target record, identified by `tid`
this distinction matters when interpreting candidate caps and per-reference candidate statistics

source3 target ids start after the source2 row count
that boundary is also used to build cross-source graph edges
changing input order without regenerating the row-id mapping would join the wrong records even if every column name still matched

the implementation therefore binds prepared data to original file hashes and verifies country, fold and target-owner joins when importing old score tables
labels, ids, truth degrees and folds are excluded from learned feature lists

## 4. why a staged design was necessary

the test set has 1,732,544 references and 9,969,589 targets
an unrestricted pairwise join is on the order of seventeen trillion comparisons
most of those comparisons cannot plausibly be matches

retrieval reduces that space; the gate allocates expensive pair scoring; the later models resolve ambiguity within the retained pool
each stage has a different failure mode:

| stage | failure | consequence |
| --- | --- | --- |
| retrieval | true reference absent | later classifiers cannot recover the link |
| gate | true candidate pruned | a high-quality cross-encoder never sees it |
| pair scoring | similar rival gets a higher score | target assigned to the wrong business |
| calibration/decision | useful candidate rejected or decoy accepted | recall or precision loss |
| export | wrong id mapping, missing row or invented candidate | invalid or irreproducible submission |

we measured those losses separately
an early perfect-matcher oracle on the saved post-gate pool reached 0.995551 on the audited fold, making candidate survival a concrete bottleneck
the lesson was to improve retrieval and its features before repeatedly changing the final tree algorithm

## 5. text representation and normalization

raw text is retained throughout
normalization produces additional comparison views rather than replacing the original challenge values

the earlier and learned feature stacks include:

- unicode normalization, case normalization and punctuation/space handling
- legal-form-separated names, compact names, token-sorted views and character skeletons
- alternate-name markers such as `dba`, `formerly`, `aka` and trading-as forms
- url/domain-like names and initials/acronym checks
- canonical address tokens, numeric tokens, state/department forms and missing-address indicators
- local transliteration and a fold-safe learned token mapping for cross-script variants

[`norm2.py`](../src/neural_v2/src/norm2.py) binds the learned normalization mapping to its fitting population
[`tfeat.py`](../src/neural_v2/src/tfeat.py) and [`gfeat.py`](../src/neural_v2/src/gfeat.py) implement the comparison views and generator-aware evidence

normalization is deliberately not a universal equality rule
two distinct businesses can share a normalized name, address or house number
we keep multiplicity, lexical detail, model disagreement and competing owners so that normalization does not automatically become a merge

## 6. retrieval lanes

### 6.1 earlier lexical and multilingual lanes

the earlier implementation combines name/address lexical similarity with pinned multilingual encoders
its independent lanes contribute complementary candidates and score/rank evidence
country-local reference pools reduce irrelevant comparisons

the original upgraded gate used 54 lexical/string/context features and declared no dense feature inputs
computing an embedding similarity during retrieval did not mean that the checkpoint learned how to use it
the richer gate required both new feature construction and a newly trained, matching feature contract

### 6.2 learned small retriever

the selected task retriever starts from `intfloat/multilingual-e5-small`
it uses symmetric `query: {name} | {address}` serialization, masked mean pooling, normalized embeddings and a 96-token input limit
training uses fold-2 identities, a recorded 64,000-pair warm start and a subsequent requested 1.5-million-pair pass
the completed pass processed 1,499,968 pairs

the retained recipe records batch 64, learning rate 0.00005, seed 42, bf16 and activation checkpointing
the latter made the local low-memory training run practical
see [`configs/training/retriever.json`](../configs/training/retriever.json)

complete-reference top-50 comparisons measured:

| population | earlier retrieval | task-trained retrieval |
| --- | ---: | ---: |
| india | 98.43% | 99.73% |
| us | 99.39% | 99.67% |
| blank-address india subset | 79.30% | 94.53% |

these are true-link retrieval recalls, not final matching or leaderboard scores

### 6.3 reverse evidence

[`reverse.py`](../src/neural_v2/src/reverse.py) also retrieves targets from reference records
the production reverse bank is built over the configured country population, with the blank-address-only scope recorded explicitly
the builder uses a faiss ivf-flat index and retains reverse-rank evidence
the gate-building workflow requests top 5 reverse candidates

the learned forward pass uses lexical top 10 and dense top 50
forward and reverse evidence are joined by the same stable `(qid, tid)` keys
index identity includes the dataset, encoder, country/scope and implementation fingerprints

## 7. the dense and name-aware gate

the final learned gate's `hybrid-v3` contract has 103 inputs:

| group | inputs | role |
| --- | ---: | --- |
| earlier lexical/string/context contract | 54 | name/address similarities, target competition, missingness and numeric overlap |
| corpus and numeric enrichment | 17 | name multiplicities, idf overlap and detailed house-number comparisons |
| generator-aware name/number evidence | 26 | additions, omissions, substitutions, initials, domains, alias markers and number edits |
| configured dense/reverse evidence | 6 | score, within-target rank and gap for the selected evidence columns |

the corpus enrichments are computed over the complete raw population for the split
for example, idf uses a smoothed `log((population + 1)/(document frequency + 1)) + 1` form
these values are not recomputed from a sample conditioned on selected reference owners

the name-change group records whether a difference looks like an insertion, deletion, typo, moved substitution, common-word replacement or rare brand-like token
numeric features distinguish exact equality, first-number membership, one-digit changes, dropped digits and last-digit differences
they are model inputs, not blanket rules that reject every address disagreement

the gate is fitted on encoder-unseen owner groups
its metadata declares the exact backend and ordered feature list
loaders reject an incompatible checkpoint rather than silently adding columns

### 7.1 adaptive candidate budget

the learned scoring run uses:

- minimum one candidate for a target that has retrieved candidates
- gate floor `0.001`
- maximum 50 candidates per target
- neural-scoring floor `0.001`

the neural floor was lowered together with the gate floor so recovered candidates were actually evaluated by the pair models

on the owner-complete gate holdout, fixed top-3 filtering retained about 99.04% of true links
the selected adaptive policy retained 99.7743%, with 1.3544 candidates per sampled query and a candidate-only oracle macro f0.5 of 0.999308
these diagnostics do not establish full-corpus matching accuracy

## 8. cross-encoder pair scoring

the selected learned ensemble contains complementary e5-small, e5-base, e5-large-instruct and bge-reranker-v2-m3 runs
their exact selected member configurations are in [`configs/training/members.json`](../configs/training/members.json)

[`ceprep.py`](../src/neural_v2/src/ceprep.py) creates five fixed hard-pair samples using seeds 21, 41, 51, 71 and 131
each requested sample has three million pairs and roughly 2.08 million positives
positive links for selected entities are restored before the cap, and held-out pairs are isolated by reference and true-owner groups

[`ce.py`](../src/neural_v2/src/ce.py) implements:

- joint reference/target tokenization with names, addresses and country text
- masked mean pooling and a binary classification head
- frozen input embeddings, binary cross-entropy with logits and adamw
- weight decay 0.01, warm-up followed by cosine decay, and gradient clipping at 1
- float16 autocast and gradient scaling for the cross-encoder runs
- distributed data parallel training on multi-gpu runs
- length-bucketed, shuffled batches to reduce padding waste
- periodic model snapshots and a final manifest of model/tokenizer file hashes

main comparisons held global batch size at 80
we varied model family, seed, learning rate, sequence length and larger-batch alternatives
fifteen complete-epoch members were frozen for scoring; the interrupted `bge21` checkpoint was retained separately and excluded

member probabilities are retained as `np_m0` through `np_m14`
the ensemble score is the sigmoid of the mean member logit, not the arithmetic mean of probabilities
the implementation also records selective evaluation and any fallback to gate evidence

## 9. the rich learned pair stack

[`stack2.py`](../src/neural_v2/src/stack2.py) uses the raw text/corpus evidence, generator features, gate/neural logits and retained member logits
the selected learned stack has 101 inputs

the first 13 retrieval/context fields from the earlier feature contract are omitted from this cached-score stack because those original retrieval columns are not present in its input tables
the model does not manufacture those unavailable values
its feature list is derived from the available score contract and checked again during scoring

supervised fitting uses a deterministic portion of fold-0 reference identities and excludes calibration/development owner groups
all fitting positives and hard negatives are retained; low-score negative targets are sampled at one in twenty and carry weight 20
inner early stopping keeps reference and true-owner grouping aligned

the new-score search used 64 optuna trials with the previous selected settings queued as trial zero
the later 2,560-trial search is retained as a separate research experiment, not the released global replacement
see [results](results.md) for the population-specific comparisons

## 10. run-6 and learned sibling refinement

the independent run-6 implementation is under [`src/er/stack`](../src/er/stack)
it unions earlier neural and lexical candidates, constructs score/text/name-frequency features, and fits a first lightgbm round using reference-group cross-fitting

the second round adds confident-owner/sibling features and the first-round logit
this allows another alias of the same reference to supply useful evidence for an otherwise ambiguous target
the release uses the `p2` score column from this branch; the historical fusion input name is `friend`

the earlier graph branch also trains binary/ranking views and a learned sibling-refinement head
[`graph_resolution/refine.py`](../src/fusion/business_entity_resolution/graph_resolution/refine.py) fits grouped lightgbm models, compares head/parent blends and writes `head` predictions
the refinement uses a historical audit already consulted during development, which is recorded as a limitation

### 10.1 inherited scope caveat

the recovered run-6 stack computes base score aggregates before target-incidence projection, but constructs some target-normalization/frequency and later sibling context after that projection
a controlled miniature example showed a competing reference losing a confident sibling under this ordering

we preserve the exact supplied score artifacts and document this limitation
we do not claim that every historical sibling feature was built over an unprojected population or that the size of the resulting score bias is known
the newer fusion and collective stages explicitly construct their context over their complete candidate pool before selecting fitting/evaluation reference groups

## 11. bounded hybrid evidence

the hybrid branch is implemented in [`final_hybrid`](../src/fusion/business_entity_resolution/final_hybrid)
it adds targeted coverage to the graph/sibling parent through four retrieval lanes: lexical name, lexical address, dense name and dense address
the dense field encoder is pinned multilingual-e5-base; reciprocal-rank fusion uses offset 20

the candidate generation is bounded to its selected target set
it preserves parent candidates and separately records genuinely new pairs
existing neural scores are reused; only missing pair scores and the required dense comparisons are computed

the cpu head fits residual logits around the parent score
existing pairs use the parent logit as their initial score; genuinely new pairs start from `logit(0.01)`
the fitting protocol uses owner-group cross-fitting and explicit weights for the sampled negative population
it compares neural-only/full feature modes and residual strengths while retaining a parent fallback

its output probability `p` is an additional input to the final full union
it is not a post-hoc candidate file made only from accepted matches

## 12. full score union and residual fusion

[`latest_fusion/pipeline.py`](../src/fusion/business_entity_resolution/latest_fusion/pipeline.py) combines:

| source | input score | union column |
| --- | --- | --- |
| learned stack | `stack_prob` | `newest` |
| learned gate and ensemble | `gate_prob`, `neural_prob`, member columns | `gate`, `neural`, `np_m*` |
| bounded hybrid | `p` | `hybrid` |
| learned sibling refinement | `head` | `graph` |
| run-6 stack | `p2` | `friend` |

old complementary pools are filtered at probability 0.001 before the full outer join
the union retains missing-score indicators; a missing model value is not treated as an observed negative prediction

preparation verifies the data metadata, newest-score content hash, pair uniqueness, target ownership labels, country/fold alignment and membership in the record universe
training truth is reconstructed from the target owner mapping after joining the pools

### 12.1 fusion features

[`latest_fusion/features.py`](../src/fusion/business_entity_resolution/latest_fusion/features.py) builds:

- score-presence flags and clipped logits for each source and neural member
- model spread, older-model mean, newest/older disagreement and gate/neural disagreement
- full-pool target competitor counts, maximum/second scores, owner margins and distance behind the best owner
- name exact/core equality, token overlap and directional coverage
- missing/extra name tokens and position relative to legal suffixes
- name multiplicities from complete reference/target record tables
- address/street overlap, missing addresses, house equality/conflict and numeric gap

the learned list excludes ids, labels, true owners, truth degrees and folds
the final collective learner additionally excludes raw country and segment ids

### 12.2 residual model

the residual fusion fits lightgbm with the newest stack's logit as its initial score
it uses a reference-disjoint fit/tune split inside the original fold-0 development partition
calibration uses a separate recorded partition
the implementation compares residual strengths, source blends and decoder choices, with overall and country-level promotion checks

fold-1 figures from this stage are explicitly historical/reused audit measurements
the emitted score table contains the selected known-country score head; accepted france decisions may come from a separate transfer policy
that distinction is important when the later sprint reuses the underlying scores for a new france blend

## 13. the competitive cavity graph

this is the final collective representation, implemented in [`innovation_graph.py`](../src/fusion/business_entity_resolution/latest_fusion/innovation_graph.py)
it is deterministic message passing followed by a learned tree reranker

### 13.1 graph construction

nodes are target records represented in the full candidate union
edges connect source2 and source3 records within the same country
the independent graph is built from normalized names and strict addresses, not truth labels or model-score thresholds

candidate edges come from equal nonempty normalized-name or address blocks
blocks with more than 16 records are excluded to avoid dense common-name/address cliques
an edge survives only without a house conflict and with either an exact normalized name or an exact address plus name-token jaccard at least 0.4

the edge weight is:

```text
0.55 * name-token jaccard
  + 0.35 * exact-address indicator
  + 0.10 * exact-name indicator
```

each target keeps its top four neighbors by weight with deterministic id ties
only reciprocal neighbor relationships survive

### 13.2 ownership and abstention

each candidate starts with its incumbent `baseline_p` score
probabilities are clipped before conversion to odds
for a target with candidate-owner odds `o(q,t)`, the graph uses:

```text
denominator(t) = 1 + sum over candidate owners of o(q,t)
prior(q,t) = o(q,t) / denominator(t)
null(t) = 1 / denominator(t)
```

the unit null odds explicitly represent abstention
competing owners remain in the denominator, including owners that are not selected to send messages
this prevents a restricted message set from appearing artificially certain simply because rivals were dropped

### 13.3 message updates

the implementation uses three rounds, damping 0.35 and shift strength 0.8
messages use edge-weighted clipped log odds
a positive shift requires at least two distinct original confident neighbors; the original seed threshold is 0.8
newly reinforced nodes do not become new original seeds

for each directed message, the update subtracts the exact reverse-edge contribution before computing the cavity state
this avoids immediate a-to-b-to-a self-reinforcement
the mean cavity signal is clipped before changing odds, and node probabilities are renormalized with the null state
the implementation checks normalization error and finite features

### 13.4 bounded work

at most two owner hypotheses per target may send messages, above a 0.05 prior floor
the directed message budget is 40 million rows
if the projected budget is larger, the implementation reduces sending owners and deterministically prunes sending targets to the recorded budget
unsent competing odds are retained in the denominator

### 13.5 representation exported to the learner

the graph adds thirteen `ig_*` features: prior/posterior owner probability, posterior null probability, target entropy, owner margin, graph degree, neighbor count, original seed count, neighbor mean/weight, logit shift, iteration change and maximum cavity shift

it does not add reference-target pairs
an absent true pair remains absent, and an entirely candidate-free target is outside this graph
retrieval recall therefore remains a separate prerequisite

### 13.6 recorded test graph

the completed collective run recorded the following workload over the released test population:

| quantity | value |
| --- | ---: |
| represented target nodes | 9,969,589 |
| targets with reciprocal neighbors | 1,750,828 |
| directed reciprocal edges | 2,686,360 |
| directed owner messages | 2,628,882 |
| positively reinforced pairs | 338,554 |
| sending targets pruned by message budget | 0 |
| maximum ownership/null normalization error | approximately 4.44e-16 |

the recorded workload stayed below the message budget
the budget fallback is an implemented ceiling, not an explanation for lost candidates in this particular run
the machine-readable evidence is retained in [`configs/collective/evidence.json`](../configs/collective/evidence.json)

## 14. the collective pair reranker

[`model_innovation.py`](../src/fusion/business_entity_resolution/latest_fusion/model_innovation.py) verifies the frozen incumbent and feature schemas, adds graph features over the complete train/test pools, then fits a new pair scorer
the selected collective mode uses lightgbm; the xgboost/catboost contextual mixture is a different experimental mode

the completed collective fit used 105 features and 796,847 fitting pairs
its selected strength was 1.0, so the late sprint's collective input is the fitted collective model probability rather than a half-strength blend

the collective backend uses 320 boosting rounds, learning rate 0.04, 63 leaves, maximum depth 8, minimum leaf size 80, l2 regularization 25, feature fraction 0.9 and seed 27943
per-target candidate counts determine normalized inverse-square-root training weights

the structured fitting protocol separates reference/name groups and removes fitting/calibration target overlap with the check population
its tests are conditional on already-developed upstream scores, not an untouched end-to-end holdout

the policy search compares model strengths 0.5 and 1.0 in logit space and enforces country-level preservation checks
the resulting full score table is the collective input to the final sprint
its original native export kept a separate france decision path; the late sprint deliberately reused the available france scores in a new blend
the new-head check improved from 0.991336 to 0.992280 macro f0.5 on 65,831 references under the recorded conditional protocol
this is not the public leaderboard score

## 15. final country decisions

the final release no longer learns during export
all settings are frozen in [`release.json`](../configs/release.json) and [`final.json`](../configs/final.json)

### 15.1 india and us

each target selects the reference with the highest collective score
ties choose the smaller `qid`
the owner is accepted only above the corresponding cut:

| country | cut |
| --- | ---: |
| india | 0.9310117959976196 |
| us | 0.9249221086502075 |

these thresholds were selected from labeled-country development comparisons with complete truth degrees and candidate competition

### 15.2 france

france has no labeled training counterpart
the final score is an available-score weighted mean:

| input | weight |
| --- | ---: |
| learned stack, `newest` | 2 |
| run-6, `friend` | 1 |
| learned sibling graph, `graph` | 1 |
| collective score | 16 |

missing inputs are excluded from both numerator and denominator
when all inputs exist, the collective contribution is 80%
each target selects its best reference and uses cut 0.8345136046409607
this arithmetic probability blend is distinct from the neural ensemble's mean-logit aggregation

## 16. french rule layers

### 16.1 directional category-swap filter

the filter uses the original run-3 french candidate-score pool, not only the finally accepted pairs
it chooses the best run-3 candidate per target and counts normalized one-word substitutions in both directions

a selected pair is dropped when all these conditions hold:

1. exactly one core-name word is missing and one is added
2. the first house-number strings agree
3. the swap and reverse-swap counts total at least 20
4. the observed direction contributes less than 70% of the total

the hypothesis is that repeated two-way category changes are more suspicious than predominantly one-way alias noise
it is a candidate-derived signal, not french ground truth
address disagreement by itself is not a hard drop

### 16.2 final-france positional filter

the additional variant drops a pair when the only added core-name token is `groupe`, no reference core token is lost, and `groupe` immediately precedes a recognized legal suffix
the reference must also have a recognized legal suffix
after-suffix aliases remain eligible

this removes exactly three accepted pairs from sprint2
the isolated accuracy effect is not measured

## 17. candidate and output contract

the final pre-matcher union contains 20,177,322 pairs
it is identical for the two released variants and includes rejected pairs

| per-reference statistic | value |
| --- | ---: |
| mean candidates | 11.6461 |
| median | 10 |
| p95 | 23 |
| p99 | 48 |
| maximum | 9,807 |
| references with no candidates | 25 |

the adaptive gate cap is per target, so it is not a cap on how many targets can point to one reference
the larger per-reference tail is therefore not a violation of the gate's top-50 rule

matching rows follow original source1 order
candidate rows preserve the recorded country-merge order: non-france references, then france
target ids within each row are sorted and deduplicated
every required reference appears and each accepted target has a single owner

the replay uses reference chunks to bound id-list aggregation memory
it checks both output hashes after writing
those byte-level checks cover row order, target ordering, empty rows and the decision policy together

## 18. compute and caching architecture

the heavy computation is factored into reusable stages:

| stage | dominant work | reusable artifact |
| --- | --- | --- |
| record preparation | cpu parsing and normalization | parquet tables, row ids and metadata |
| retrieval training | gpu contrastive learning | retriever bundle and fitting manifest |
| reference/target encoding | gpu matrix work | embeddings and reverse indexes |
| gate features and fit | gpu retrieval plus cpu features/trees | gate bundle and corpus caches |
| cross-encoder fitting | gpu distributed training | independent member bundles |
| full pair scoring | gpu inference and sharded i/o | complete per-member score tables |
| tree/fusion/graph stages | cpu features and learned scoring | fitted heads, prepared tables and predictions |
| final tuning and export | cpu arrays, rules and grouped strings | frozen policies and exact tsvs |

training used single- and four-a100 node types; large cpu feature and fitting work used `Standard_E64ds_v4`; final policy passes used `Standard_E16ds_v4`
the [compute and execution guide](compute.md) records the allocation, environments, concurrency, sharding and recovery design

hashes bind caches to raw data, normalized views, ordered features, model files, source implementations and scope
portable cache relocation changes a verified data location while preserving its content identity
changing retrieval or neural weights requires new dependent scores; changing only a cpu head can reuse frozen neural evidence

## 19. why the architecture changed during development

| observation | architectural response |
| --- | --- |
| blank-address records dominated missing candidates and wrong owners | task-specific retrieval, reverse evidence and richer ambiguity features |
| a stronger tree over the same restricted features barely helped | add evidence before another broad tree sweep |
| fixed top-3 filtering lost useful true links | adaptive score-based gate with an explicit cap and fallback |
| pairwise tuning could favor the wrong operating point | evaluate complete-reference macro f0.5 and complete rival ownership |
| historically projected sibling banks could change competitors' context | explicit full-pool feature construction in newer stages and a documented inherited limitation |
| the early unseen-country route bypassed useful rich scores | evaluate country transfer and retain richer scores for the final france blend |
| neural scoring was expensive to repeat | retain member scores, features, fitted heads and policy inputs separately |
| unrestricted local full-table operations exceeded ram | projected columns, sequential replay and chunked export |

## 20. limits and interpretation

high score agreement is evidence, not proof of business identity
duplicate normalized businesses and weak or absent addresses can remain ambiguous
graph propagation can amplify correlations, so it is bounded, competitive, seed-constrained and treated as learned features rather than mandatory links

the final french policy is a domain-transfer decision with public-result evidence, not a labeled france experiment
historical audit reuse is disclosed in the relevant stage reports
no earlier oracle or development value is presented as the final public score

for artifact bindings, reconstruction commands and the experiment chronology, continue with [reproduction](reproduce.md), [full pipeline](pipeline.md) and [results](results.md)
