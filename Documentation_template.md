# ml challenge 2026 business entity resolution

- **team:** amazites
- **members:** varun jhaveri, shivsharan sanjawad, raj mathuria, aastha singh
- **doc revision:** 2026-09-25
- **submission date:** pending actual upload
- **deadline:** 2026-09-27 08:00 ist / 02:30 utc

> implementation-method draft
> final variant cutoff full-pool score candidate count and archive hash must be populated from completed artifacts

## 1. executive summary

the system retrieves candidate businesses from complementary lexical and multilingual views
a learned gate reduces the shortlist before a locally fine-tuned pair classifier scores each retained link
full-pool calibration selects the decoder and acceptance threshold

the baseline uses e5-base and qwen retrieval with the original lightgbm/catboost gate
the implemented upgrade uses the verified teammate lightgbm gate plus e5-base qwen and e5-large retrieval
both use the fine-tuned e5 matcher
the final submitted variant is selected from complete results rather than assumed here

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

entity-grouped folds use seed 42
fold 2 fits models fold 0 tunes decisions and fold 1 is reserved for locked audit
aliases of one known business remain together

the neural matcher was trained on 502,635 pairs including 34,785 positives
it uses an e5-base initialization a binary classification head and bce-with-logits loss
the recorded fit uses two epochs batch 128 and maximum pair length 384
hard negatives come from generated candidates with additional bounded random negatives
missed known positives may be inserted into fit data only

the upgraded lightgbm uses 54 features and 248 fitted trees
its fit used 1,858,304 fixed candidate pairs
four sampled reference aggregates were removed because they exposed owner-sampling membership
the production transform and checkpoint probabilities passed exact parity checks

## 4. blocking and candidate generation

| setting | baseline | upgrade |
| --- | --- | --- |
| lexical width | 20 | 20 |
| dense width per retriever | 100 | 100 |
| dense lanes | e5-base and qwen | e5-base qwen and e5-large |
| learned gate | native lightgbm/catboost | teammate lightgbm without sampled s1 aggregates |
| final candidates per target | at most 3 | at most 3 |

the lexical lane unions independent name/address searches and exact normalized keys
exact-key collisions are retained
both field similarities are computed completely for every retained pair
country labels are discovered from data and an explicit unpartitioned fallback handles missing/unmatched labels

e5-base uses its query/passage prefixes
qwen uses an identity-retrieval instruction and left padding
e5-large uses an instructed query and raw ref text
all model revisions are pinned in [model sources](reports/model_sources.json)

dense search is exact and chunked: memory is bounded, but the reference pool is still scanned
billion-record approximate indexing is not claimed
the final candidate file is the complete post-gate input to the neural matcher, including links later rejected

**actual candidate pairs:** pending completed export

**per-s1 mean / p50 / p95 / p99 / max:** pending strict validation report

three candidates per target is not three candidates per s1
the final source1 distribution is measured directly from the output

## 5. matching and decoding

the baseline gate uses the native 44-feature representation
the upgrade includes canonical compact skeleton alias legal-form address state and soft numeric evidence
target-side ranks and gaps are retained while the four biased sampled s1 aggregates are excluded

the final neural classifier scores field-labeled ref/target pairs
the current blend is weighted log odds with neural weight 0.6
the upgraded parts retain gate and neural probabilities for later measured refinements

calibration compares plain thresholding with selecting one best ref per target before thresholding
it scores every labeled target so false assignments from other businesses remain visible
selection uses fold 0; fold 1 remains the locked audit

**final decoder / cutoff:** pending full-pool calibration

**final submitted variant:** pending complete variant comparison

## 6. results and limitations

| observation | scope |
| --- | --- |
| baseline india lexical/e5/qwen recall 0.995379 at dense width 100 | selected query retrieval diagnostic |
| adding large-instruct recalled 0.996101 and five extra positive queries | same selected query population |
| safer teammate gate improved paired high-precision recall with unchanged neural scores | fixed candidate diagnostic |
| 54-feature arrays and checkpoint predictions matched exactly on 2,613 pairs | runtime parity check |
| upgraded real-weight smoke covered 12 targets and 36 candidates | functional integration check |

these figures are not an official or full-pool matching score
repeated generic names shared addresses and near-copy records cause false merges
script changes shortened names weak addresses and edited numbers cause missed links
number conflicts remain soft evidence because true pairs can contain number noise

**full-pool macro f0.5:** pending

**locked audit:** pending

**leaderboard feedback:** upgraded v1 submitted with reported public leaderboard f0.5 of 0.969

## 7. execution and reproducibility

validation scoring and test scoring run as independent jobs
each is divided into four outer target-id partitions with 250,000-record internal work units
single-a100 workers use explicit gpu affinity and two 12-thread processes
cpu handles prep features calibration export and validation; gpu handles the selected neural workloads

the original four-a100 worker lost allocation at 9,161,442 scored validation targets
verified checkpoints and the original runtime snapshot were retained for smaller-worker recovery
resumed artifacts must match data model feature numeric and source fingerprints
the complete target pool must be covered exactly once

uv pins python and dependencies
model metadata pins revisions licenses feature order parameter counts and precision
the baseline uses about 1.152b neural parameters and the upgrade about 1.712b including the separately trained matcher
deployed source models have mit or apache-2.0 metadata and remain below the challenge parameter ceiling

azure ml training used `Standard_NC24ads_A100_v4` with 24 vcpus and one a100 80 gb
the initial multi-gpu pass used `Standard_NC96ads_A100_v4` with 96 vcpus and four a100 80 gb
accelerated test scoring used 16 disjoint single-a100 assignments with shared reference caches and reusable score batches
cpu aggregation/export used `Standard_E16ds_v4` with 16 vcpus

## 8. submission contents and verification

the required outputs are `matching_results.tsv` and `candidate_pairs.tsv`
strict validation checks headers ids duplicates all-ref coverage and final-match membership in candidates
it also reports candidate count distributions for the organizer's additional ranking criterion

the final archive includes source uv lock local selected model snapshots tokenizer files calibration model provenance and upstream notices
raw datasets credentials cloud caches and training feature matrices are excluded

**upgraded v1 output checksums:** recorded in `reports/submission_v1.json`; both output validators passed with id checks enabled

**final archive hash:** pending final model selection

implementation detail: [arch](docs/arch.md)

reproduction and operations: [ops](docs/ops.md) and [training](docs/training.md)

research and evidence: [plan](plan.md) and [report index](reports/README.md)
