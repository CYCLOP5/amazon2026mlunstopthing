# entity resolution arch

> supplied records → small candidate sets → scored links → calibrated sets
>
> this describes the implemented baseline and upgraded runtime
> full-pool results and completed submission files remain separate delivery gates

## 1. what the system produces

each s1 ref represents one business
the system links matching aliases from s2 and s3 or returns an empty set

| term | meaning |
| --- | --- |
| ref / anchor | an s1 business record |
| target / query | one s2 or s3 record being resolved |
| candidate | a possible ref–target link retained by blocking |
| gate | the learned filter before the final neural matcher |
| matcher | the fine-tuned pair classifier |
| decoder | the rule turning pair scores into accepted link sets |
| calibration | choosing the decoder and cutoff from labeled held-out entities |
| test infer | predicting the unlabeled competition records |
| audit | the held-out entity fold reserved for final evaluation |

the required files are

```text
matching_results.tsv
  source1_entity_id    matched_entity_ids

candidate_pairs.tsv
  source1_entity_id    candidate_entity_ids
```

both contain every required s1 id exactly once
target lists are comma-separated and may be empty
every final match must belong to that ref's exported candidate set

## 2. actual dependencies

model fitting is complete for the two currently running variants
their labeled-pool scoring and test scoring can run concurrently

```mermaid
flowchart td
    raw[provided tsvs] --> prep[prep and fixed entity folds]
    prep --> fit[fit models on fold 2]
    fit --> weights[frozen weights and feature contracts]
    weights --> val[score complete labeled target pool]
    weights --> test[score complete competition test pool]
    prep --> val
    prep --> test
    val --> cal[choose cutoff on fold 0]
    cal --> audit[locked fold 1 audit]
    cal --> export[decode test scores and export]
    test --> export
    export --> checks[strict ids coverage subset and size checks]
    checks --> pack[package and upload]
```

test scores do not depend on the chosen cutoff
export depends on both complete test scores and the matching calibration
the old single-job sequence delayed the first upload by running test scoring after calibration
current execution uses separate validation and test partitions

test infer is necessary because it creates predictions for the competition's unknown records
validation is a quality-control step rather than a mathematical prerequisite for obtaining a public leaderboard score
this repo deliberately requires full-pool calibration before final export so a sampled optimistic cutoff cannot be submitted accidentally

## 3. data model and folds

`src/data.py` reads the supplied files and writes parquet plus `meta.json`

| field | role |
| --- | --- |
| `eid` | original external id used in final tsvs |
| `rid` | compact internal row id used for joins and coverage |
| `nm`, `ad` | original unicode name and address text |
| `nn`, `an` | normalized offline-transliterated comparison views |
| `co` | normalized country label |
| `sr` | source number |
| `own` | known training owner or unmatched marker |
| `fold` | ref-level fit / tune / audit assignment |
| `deg` | known training link count for metric accounting |

ids and training ownership are not predictive features
they are used for joins labels and verification

| fold | use |
| --- | --- |
| 2 | model fit and hard-negative construction |
| 0 | tuning and cutoff selection |
| 1 | locked audit |

seed 42 freezes entity-grouped folds
aliases of one known business stay together
strata include country link degree unicode aliases and blank aliases
`meta.json` hashes bind later artifacts to this prepared dataset

### text views

- preserve raw unicode for neural input and raw-text comparisons
- use nfkc casefold punctuation normalization and whitespace normalization
- use `anyascii` for the shared lexical comparison view
- use the pinned `unidecode` implementation inside the teammate feature backend because its checkpoint was trained with that exact transform
- do not replace names with external translations
- do not enrich records through business lookup geocoding or external gazetteers

this is partly cross-script retrieval
many india target names use scripts absent from their mostly latin-script s1 references
raw-unicode-only lexical matching does not bridge that gap

## 4. candidate generation

```mermaid
flowchart lr
    q[target text] --> lex[char tfidf and exact keys]
    q --> e5[e5-base retrieval]
    q --> qw[qwen retrieval]
    q --> lg[e5-large retrieval in upgrade]
    refs[country reference indexes] --> lex
    refs --> e5
    refs --> qw
    refs --> lg
    lex --> union[union and dedup]
    e5 --> union
    qw --> union
    lg --> union
    union --> fields[complete pair feature values]
    fields --> gate[learned gate]
    gate --> c[up to 3 candidates per target]
    c --> nn[final neural matcher]
    c --> cp[candidate pairs export]
```

### lexical lane

`src/block.py` uses independent name and address character retrieval plus exact normalized keys
exact-key collisions are retained rather than silently choosing one business
the current `k_lex` is 20
the union may contain more than 20 pairs per target because channels and exact-key ties are combined

candidate membership alone does not supply both field scores
the implementation computes complete name and address cosine values for every retained pair
the corrected semantics are recorded as block score version 2
old incompatible feature caches and models are rejected

### dense lanes

`src/embed.py` owns model-family serialization and pinned loading
`src/hybrid.py` owns reference caches retrieval and candidate union

| alias | role formatting | dim | max tokens |
| --- | --- | ---: | ---: |
| `e5` | `query:` for targets and `passage:` for refs | 768 | 256 |
| `qwen3` | identity instruction for targets and raw ref text | 1024 | 256 |
| `e5-large` | instructed target query and raw ref text | 1024 | 512 |

the shared instruction asks for the identical business despite transliteration typos and formatting changes
name address and country are field-labeled
qwen uses left padding
embedding vectors are normalized and stored as float16
cuda weights load directly in float16 to avoid a transient float32 allocation peak

each active dense lane retrieves up to 100 refs per target
the three-retriever upgrade adds a channel before the gate rather than enlarging the final neural shortlist

### countries

country values are discovered from data
france is handled even though it has no labeled training counterpart
ordinary retrieval is country-partitioned
missing or unmatched country labels use an explicitly recorded unpartitioned fallback

### scale ceiling

the lexical lane uses sparse indexes and avoids a full cartesian pair table
the current dense implementation performs exact chunked similarity search
its memory is bounded but its arithmetic still scans the reference pool for each query

this is not proof of billion-record readiness
an approximate index such as hnsw or ivf/pq is a future scaling change requiring new recall and timing measurements
final candidate size and upstream retrieval cost are different quantities and both must be reported honestly

## 5. two implemented gate backends

| version | feature backend | fitted model | dense lanes |
| --- | --- | --- | --- |
| baseline | native 44-feature contract | lightgbm / catboost mean | e5 and qwen |
| upgrade | `teammate-v1-nos1`, 54 features | safe teammate lightgbm, 248 trees | e5 qwen and e5-large |

### baseline

`src/feat.py` builds raw and normalized name/address similarities numeric agreement missingness and retrieval evidence
`src/train.py` loads the fitted native model ensemble and enforces feature order

### upgrade

`src/tm_rules.py` holds the supplied token maps
`src/tm_prep.py` constructs canonical compact skeleton legal-form alias address-number and state views
`src/tfeat.py` builds the exact 54-column checkpoint contract

the upgrade includes

- name edit token-sort token-set partial and jaro-winkler evidence
- compact and skeleton name agreement
- extracted alias and legal-form comparisons
- canonical address and skeleton comparisons
- soft number agreement conflict and first-number cues
- state agreement missingness and target-side retrieval ranks/gaps

the uploaded teammate code did not contain the house-number features referenced by an older diagnostic report
this implementation does not claim to have recovered that missing version

### removed sampling shortcut

four reference-side aggregates were omitted

```text
s1_ncand
s1_bmax
s1_ntop1
s1_gap
```

when computed from an owner-sampled query set they reveal which owners supplied the sampled aliases
that inflates validation accuracy without establishing full-corpus behavior
any later use must compute them over the complete target population

the retained gate was fitted on 1,858,304 fixed candidate pairs
its held-out candidate evaluation used 716,835 pairs and 13,827 selected positive targets
these are sampled diagnostics

the production port matched the supplied preprocessing feature arrays and checkpoint probabilities exactly on 2,613 pairs from 64 queries
see [runtime parity](../reports/teammate_runtime_parity.json)

## 6. final neural matching

`src/neural.py` implements a locally trained binary pair classifier initialized from multilingual e5-base
it is a separate model from the frozen e5 retrieval encoder

- field-labeled ref and target text
- binary classification head with bce-with-logits training
- 502,635 training pairs including 34,785 positives
- hard negatives from generated candidates plus bounded random negatives
- known positive injection only in fit data when needed
- no gold-positive injection into tuning candidates
- two training epochs and maximum pair length 384

`src/match.py` applies the gate before neural scoring
the current final shortlist is at most three refs per target

the final score blends gate and neural log odds

```text
logit(p) = 0.6 * logit(p_neural) + 0.4 * logit(p_gate)
```

the upgraded output parts retain the component probabilities as well as the blend
that supports measured later refinements without losing the original scores
it does not authorize removing candidates based on accepted final matches

## 7. calibration and metric

for one ref with true set `t` and predicted set `p`

```text
f0.5 = 1.25 * tp / (1.25 * tp + fp + 0.25 * fn)
```

the challenge averages over refs
correct empty sets score one
false merges into singleton or unmatched refs therefore matter strongly

`src/infer.py` scores the complete labeled target pool but selects the cutoff on fold 0
the locked audit uses fold 1
scoring the full pool does not mean using fit labels to choose the cutoff

the complete pool is needed because a target belonging to another business can be falsely assigned to an evaluation ref
sampling only aliases owned by evaluation refs would hide many such errors

the two decoder candidates are

1. accept every pair above the cutoff
2. select the highest-scoring ref per target then apply the cutoff

calibration records the selected decoder threshold data fingerprint model fingerprint numeric settings and exact coverage
the public leaderboard supplies feedback after an upload; it is not a replacement for these offline checks

## 8. shard and cache contracts

`src/run.py` splits work by country and optional global target-id intervals
production partitions use 250,000-target work units inside four larger train or test partitions
each single-a100 worker runs two processes with 12 cpu threads each

```text
prepared data + frozen weights
  ├─ validation quarters 0 1 2 3
  └─ test quarters       0 1 2 3
```

reference caches are warmed before workers sharing that local cache read them
cache identity includes record order and text hashes model revision role serialization length dtype and dimension
complete embedding caches are read-only during inference
sparse joblib maps use copy-on-write mode because the sparse top-n extension rejects read-only buffers

each country/range run records

| artifact | purpose |
| --- | --- |
| `manifest.json` | configuration provenance listed parts and hashes |
| `parts/*.parquet` | scored candidate ids probabilities and training labels where applicable |
| `parts/*.npy` | exact target-id coverage including queries with no retained pair |
| `runs.json` | parent ownership child outcomes warmups and completion |
| `runpaths.json` | relative paths used to assemble the complete pool |

atomic writes prevent a half-written file from becoming a valid manifest entry
coverage arrays prevent missing empty predictions from looking complete
resume checks current data models source fingerprints and listed part hashes
every target must be covered exactly once across the combined runs

completed children are revalidated when resuming
restored checkpoint counts are not counted as newly computed records
original runtime snapshots are preserved when resuming a prior model version after the main branch changes

## 9. cpu and gpu responsibilities

| work | implementation | suitable compute |
| --- | --- | --- |
| tsv prep eda folds | polars numpy stdlib | cpu |
| sparse lexical search | scipy and sparse top-n | cpu |
| canonical features | polars rapidfuzz local token maps | cpu |
| boosted-tree fit / predict | lightgbm catboost; xgboost evaluated separately | cpu |
| embedding encode | sentence transformers / transformers | gpu for this corpus |
| dense similarity scan | bounded gpu matrices or cpu exact search | gpu preferred |
| pair classifier | pytorch / transformers | gpu preferred |
| calibration export validation zip | polars stdlib | cpu |

cpu-only neural inference is possible but has not been benchmarked as a competitive deadline path for these roughly 20m train/test targets
more cpu quota does not create more gpu capacity
the current hybrid pipeline uses both cpu and gpu within each worker

## 10. cloud lifecycle and budget

`src/cloud.py` reserves a conservative ceiling before creating task-tagged compute

```mermaid
flowchart lr
    reserve[reserve budget] --> create[create min 0 max 1 compute]
    create --> submit[submit bounded job]
    submit --> work[execute and persist output]
    work --> complete[verify job result]
    work --> fail[failed canceled or interrupted]
    complete --> cleanup[verify task compute deletion]
    fail --> cleanup
    cleanup --> ledger[close reservation]
```

the authorization is $1,000 total
two disjoint $500 ledgers preserve compatibility with already-running controllers

```text
artifacts/budget.json          baseline allocation
artifacts/upgrade_budget.json  upgrade allocation
```

their allocated caps sum to $1,000
reported accrual and worst-case reservations are conservative estimates rather than an azure invoice
pre-existing workspaces and unrelated resources are preserved

regional low-priority quota was verified at 600 cores during this session
quota and available physical capacity are separate
the original four-a100 worker lost its allocation at 9,161,442 scored validation targets
its 41 manifests had no missing or unlisted checkpoint files
remaining validation was moved to a smaller a100 using the immutable original runtime and copied checkpoints

shared input assets are uploaded once and hash-verified before parallel jobs receive their datastore uri
concurrent sdk uploads of the same local model folder previously collided with `BlobAlreadyExists`
artifact reads use the datastore's actual credential type and skip zero-byte directory markers

submitted azure jobs retain their deadlines output persistence and automatic compute cleanup

## 11. candidate-budget evidence

the organizer now reviews candidate generation code and candidate count per s1 alongside matching score
`src/validate.py` reports total pairs empty rows mean nearest-rank p50/p95/p99 maximum and the complete size histogram

three candidates per target is not a cap of three per s1
many target records may retrieve the same ref
the actual source1 distribution is measured from the exported file

the candidate file contains the complete final pre-neural shortlist
rejected neural pairs remain candidates
it must not be reconstructed from accepted matches or padded with unrelated pairs

## 12. reproducibility and evidence gates

- python version and dependencies are pinned with uv
- pretrained ids revisions licenses and parameter bounds are in [model sources](../reports/model_sources.json)
- the upgraded neural stack is about 1.712b parameters including the separately trained matcher
- original unicode and offline comparison views are separate
- fold ownership prevents alias leakage
- corrected lexical scores have an explicit version
- numeric precision and blend weight are part of inference identity
- checkpoint paths are portable but content fingerprints remain strict
- final export requires complete labeled-pool calibration and complete test coverage
- strict validation checks ids headers duplicates coverage and match-subset membership
- packaging includes the selected local snapshots weights tokenizer files calibration and upstream notices

see [evidence index](../reports/README.md), [ops](ops.md), [delivery status](status.md), and [research / eda](../plan.md)

## 13. remaining delivery work

1. finish baseline recovery and baseline test partitions
2. finish upgraded validation and test partitions
3. calibrate each configuration against the full labeled target pool
4. export and validate both tsvs for the selected first upload
5. use the first leaderboard result to choose the next refinement
6. complete all three intended uploads before sunday 2026-09-27 at 08:00 ist / 02:30 utc
7. finish artifact packaging budget audit and task-resource cleanup

no full-corpus score leaderboard score or completed submission is inferred from a passed smoke test
