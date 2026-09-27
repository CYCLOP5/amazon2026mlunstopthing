# ml challenge 2026 business entity resolution

- **team:** amazites
- **members:** varun jhaveri, shivsharan sanjawad, raj mathuria, aastha singh
- **doc revision:** 2026-09-27
- **submission date:** pending actual upload
- **submission planning cutoff:** 2026-09-27 21:00 ist / 15:30 utc

> implementation-method draft
> final variant cutoff full-pool score candidate count and archive hash must be populated from completed artifacts

## 1. executive summary

the system retrieves candidate businesses from complementary lexical and multilingual views
a learned gate reduces the shortlist before a locally fine-tuned pair classifier scores each retained link
full-pool calibration selects the decoder and acceptance threshold

the learned pipeline uses a task-trained multilingual-e5-small retriever with lexical and reverse-rank evidence
a 103-feature lightgbm gate feeds 15 fine-tuned cross-encoders and a cpu-trained rich lightgbm stack
the cross-encoder pool contains e5-small e5-base e5-large-instruct and bge-reranker-v2-m3 variants
the final submitted decoder and rule variant are selected from completed artifacts

## 2. data and problem analysis

- 2,206,821 training refs and 1,732,544 test refs
- 10,320,219 labeled training targets and 9,969,589 competition test targets
- repeated names shared addresses blank fields and script changes create ambiguous matches
- france is about 15% of test refs but has no labeled training counterpart
- macro per-ref f0.5 rewards correct empty sets and penalizes false merges strongly

raw unicode text is retained
offline `anyascii` comparison views bridge scripts for lexical search
the teammate backend uses its pinned `unidecode` transform to preserve checkpoint semantics
no external business lookup geocoding translation service or external reference enrichment feeds predictions

## 3. split and training strategy

entity-grouped folds use seed 42; aliases of one known business remain together
fold 2 supplies encoder and cross-encoder fitting data
the task retriever trained on 1,499,968 pairs with 96-token symmetric `query: {name} | {address}` serialization
the gate uses encoder-unseen fitting queries and fold-0 validation queries

cross-encoder runs use entity-isolated three-million-pair hard-pair samples
each model adds a binary head over mean-pooled name/address representations with bce-with-logits loss
the comparison varies family seed learning rate sequence length and batch size
15 complete-epoch models are selected: 6 large-instruct, 4 base, 2 small, 3 bge
member settings and weight fingerprints are recorded in their bundled metadata

the rich stack uses three deterministic reference partitions within fold 0: fit, calibration/search, development
an inner owner/reference-disjoint split chooses boosting rounds before refitting the fitting partition
calibration and development owners are excluded from stack fitting targets
full-target competitors and full-corpus frequency features are retained
fold-0 references participated in prior base-model development; these are development measurements, not a fresh blind audit

## 4. blocking and candidate generation

| setting | learned pipeline |
| --- | --- |
| lexical width | 10 |
| dense width | 50 |
| dense encoder | task-trained multilingual-e5-small |
| reverse lane | cached full-population reverse ranks |
| learned gate | 103-feature lightgbm, including dense cosine and reverse rank |
| retained candidates | gate probability >=0.001, maximum 50, minimum one fallback |

the lexical lane unions independent name/address searches and exact normalized keys
exact-key collisions are retained
both field similarities are computed completely for every retained pair
country labels are discovered from data and an explicit unpartitioned fallback handles missing/unmatched labels

the trained small encoder uses the same serialization for query and reference
its reverse index uses the same pinned model bundle and full target population
all model revisions are pinned in [model sources](reports/model_sources.json)

dense search is exact and chunked: memory is bounded, but the reference pool is still scanned
billion-record approximate indexing is not claimed
the final candidate file is the complete post-gate input to the neural matcher, including links later rejected

**actual candidate pairs:** pending completed export

**per-s1 mean / p50 / p95 / p99 / max:** pending strict validation report

per-target limits are distinct from candidate counts per source1 business
the final source1 distribution is measured directly from the output

## 5. matching and decoding

features cover normalized names addresses legal forms state numeric evidence full-corpus ambiguity and generator-aware name changes
the neural members score field-labeled ref/target pairs at gate probability >=0.001
large and bge members use a recorded upper gate bound of 0.995; skipped member scores fall back to the gate probability
the effective member probabilities are combined by mean logits and retained as separate stack inputs

the rich lightgbm stack learns from raw-text features gate evidence and all 15 member columns
cpu optuna starts from the previous selected parameters and compares 64 trials on cached features
selection uses per-reference macro f0.5 with complete truth degrees and full-target competition
country/house-segment calibration supports density transfer and unseen-country routing
one reference owner is selected per target before per-reference expected-f0.5 set decoding
normal-sized sets use exact expectations; oversized groups use a bounded approximation recorded in the output metrics
bounded france rules cover verified name transformations and soft address-number evidence; a plain export is retained for comparison

**final decoder / cutoff:** pending full-pool calibration

**final submitted variant:** pending complete variant comparison

## 6. results and limitations

| observation | scope |
| --- | --- |
| learned retriever top-50 recall: india 99.73%, us 99.67% | complete reference pools with sampled held-out queries |
| blank-address india retrieval recall: 79.30% to 94.53% | same retrieval comparison |
| adaptive gate true-link retention: about 99.77% versus 99.04% for fixed top 3 | gate holdout |
| mean retained candidates: about 1.35 versus 3 per query | same gate holdout; final per-source1 counts measured separately |
| prior optuna stack: macro f0.5 0.9845466096 on 73,752 businesses | earlier full-target development comparison |
| live india/us/france member-score and mean-logit checks passed | functional scoring checks |

these figures are not an official or full-pool matching score
repeated generic names shared addresses and near-copy records cause false merges
script changes shortened names weak addresses and edited numbers cause missed links
number conflicts remain soft evidence because true pairs can contain number noise

**full-pool macro f0.5:** pending

**selected stack development:** pending

**leaderboard feedback:** learned submission not yet uploaded; upgraded v1 previously reported public f0.5 of 0.969

## 7. execution and reproducibility

full train/test scoring uses 16 disjoint country/target-id assignments with 200,000-target internal work units
workers use explicit gpu affinity and share warmed reference indexes
score manifests bind data models feature order numeric policy and source hashes; coverage must be complete and non-overlapping
every selected member score is saved for cpu-only stack and calibration experiments
full-corpus cpu caches fit/evaluation matrices trial models and study journals are retained separately from the submission

uv pins python and dependencies; model metadata pins revisions licenses feature order parameter counts and precision
selected cross-encoders contain 6,397,997,583 parameters; the learned retriever adds 117,653,760
the deployed total is 6,515,651,343, below the eight-billion limit
source models have mit or apache-2.0 metadata

azure ml uses `Standard_NC96ads_A100_v4`, `Standard_NC48ads_A100_v4` and `Standard_NC24ads_A100_v4` a100 80 gb workers
cpu feature preparation tuning and export use `Standard_E64ds_v4` with 64 vcpus

## 8. submission contents and verification

the required outputs are `matching_results.tsv` and `candidate_pairs.tsv`
strict validation checks headers ids duplicates all-ref coverage and final-match membership in candidates
it also reports candidate count distributions for the organizer's additional ranking criterion

the final archive includes source uv lock local selected model snapshots tokenizer files calibration model provenance and upstream notices
raw datasets credentials cloud caches and training feature matrices are excluded

**upgraded v1 output checksums:** recorded in `reports/submission_v1.json`; both output validators passed with id checks enabled

**final archive hash:** emitted in the external packaging receipt; the archive contains a per-file hash manifest

implementation detail: [arch](docs/arch.md)

reproduction and operations: [ops](docs/ops.md) and [training](docs/training.md)

research and evidence: [plan](plan.md) and [report index](reports/README.md)
