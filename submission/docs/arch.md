# architecture

## 1. records and identifiers

source1 contains reference businesses
sources2 and 3 contain records to link or reject
internal `qid` is the zero-based source1 row number
internal `tid` is the source2 row number followed by source3 rows with the source2 offset
external entity ids are restored for export

raw names and addresses are retained
comparison views normalize unicode, punctuation and locally available transliteration
country keys are stripped and lowercased

## 2. candidate generation

our team combines lexical name/address retrieval with multilingual dense retrieval
the learned retriever uses task-specific training pairs and country-local reference pools
the earlier retrieval lanes provide additional candidates and independent score evidence

the fusion preparation joins the scored pools by `(qid, tid)`
missing model scores remain explicit rather than being treated as observed negatives
population-dependent frequency, rival-owner and sibling features are built from the full population before reference selection

the final union contains 20,177,322 pairs for 1,732,544 reference businesses
candidate counts per reference have mean 11.6461, median 10, p95 23 and p99 48
the maximum is 9,807; 25 references have empty candidate lists

## 3. pair scores

the learned retrieval pipeline applies a lightgbm gate before cross-encoder scoring
its pair ensemble combines member logits and feeds a richer tree stack
our run-6 stack, graph sibling model and bounded hybrid provide complementary evidence
the residual fusion and collective-graph head learn from these frozen scores and full-population context

the final collective prediction table retains every candidate pair
the replay uses its probabilities directly for india and the us

## 4. final india/us decisions

each target first chooses its highest-scoring reference across the complete candidate population
ties choose the smaller `qid`
the final acceptance cuts are:

| country | cut |
| --- | --- |
| india | 0.9310117959976196 |
| us | 0.9249221086502075 |

these cuts were selected on labeled reference groups using complete truth degrees and competing owners
the calibration comparison is development evidence, not an independent end-to-end accuracy estimate

## 5. final france decisions

france is absent from labeled training data
its final pair score is an available-score weighted mean:

| score | weight |
| --- | --- |
| learned neural stack | 2 |
| run-6 stack | 1 |
| graph sibling model | 1 |
| collective graph | 16 |

weights are normalized over the models that actually scored each pair
the nominal collective share is 80% when all components are present
targets choose one owner, then pairs below 0.8345136046409607 are rejected

## 6. french category-swap filter

names are normalized with accent and punctuation removal and a fixed legal-form vocabulary
the rule finds pairs with exactly one record-only word and one reference-only word
swap directions are counted over the original run-3 french best-candidate pool
the rule removes a selected pair when:

- the first house-number strings agree
- the swap and its reverse occur at least 20 times in total
- the observed direction contributes less than 70% of that total

this distinguishes frequent two-way category substitutions from predominantly one-way alias noise
address-number disagreement alone is not a rejection rule

## 7. final-france positional variant

the final-france archive additionally removes a pair if its only added core-name token is `groupe`, it loses no reference core token, and `groupe` appears immediately before a recognized french legal suffix
the reference must also contain a recognized legal suffix
after-suffix aliases are retained

this rule removes three pairs from sprint2
its isolated accuracy effect has not been measured

## 8. export

matching rows follow the original source1 row order
candidate rows preserve the submitted country-merge order: non-france references first, then france
target ids within each row are sorted and deduplicated
the replay hashes both files after generation and rejects any difference from the submitted artifacts

## 9. execution platform

| machine | use |
| --- | --- |
| `Standard_NC24ads_A100_v4` | a100 pair-model training and scoring workers |
| `Standard_NC96ads_A100_v4` | multi-gpu scoring and training pilots |
| `Standard_E64ds_v4` | full-population CPU feature, stack and calibration work |
| `Standard_E16ds_v4` | final country-policy and france-blend passes |

recorded-score replay needs CPU resources only
the release exporter writes reference chunks to bound string-aggregation memory
