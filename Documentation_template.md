# ml challenge 2026 business entity resolution

- **team:** amazites
- **members:** varun jhaveri, shivsharan sanjawad, raj mathuria, aastha singh
- **archive variant:** sprint2
- **matching pairs:** 5,866,303
- **candidate pairs:** 20,177,322
- **recorded public result:** 0.990284

## 1. executive summary

our team combines multilingual and lexical retrieval with learned pair scoring and full-population graph evidence
the final decision layer uses collective probabilities for india/us and a weighted blend for france
country-specific decoding is followed by a french category-swap filter
the final-france variant adds a narrow positional name rule

the archive contains the exact submitted tsvs, complete source implementations, pinned environments, model-source records and compact score assets for deterministic replay

## 2. problem analysis

the challenge contains 2,206,821 training reference businesses and 1,732,544 test references
there are 10,320,219 labeled training targets and 9,969,589 test targets
training covers india and the us; france is unlabeled

repeated names, missing addresses, transliteration, legal-form changes and near-copy decoys make ownership ambiguous
the macro f0.5 metric rewards correct empty predictions and strongly penalizes false merges
raw business identities are not enriched through external lookup services

## 3. candidate generation

lexical name/address retrieval is combined with multilingual dense retrieval
a task-trained retriever improves coverage for difficult aliases
independent earlier retrieval and scoring lanes add complementary candidates
the fusion stage joins these scored pools by reference and target ids

the final candidate tsv contains the complete pre-matcher union, including pairs the final decision layer rejects
candidate counts per reference have mean 11.6461, median 10, p95 23 and p99 48
the maximum is 9,807

## 4. matching and graph evidence

the learned gate feeds a cross-encoder ensemble and a richer lightgbm stack
our run-6 stack, graph sibling refinement and bounded hybrid provide independent evidence
residual and collective-graph models combine these signals with rival-owner, name-ambiguity, address and source-occupancy features

population-dependent features are computed over the full population before reference selection
initial training uses entity-grouped folds; later fusion development uses its recorded reference/name-group protocol
those later checks are conditional on previously developed upstream models

## 5. final decisions

targets first choose one reference owner with deterministic tie-breaking
india and us use frozen collective-score cuts of 0.9310117959976196 and 0.9249221086502075 respectively

france uses a weighted mean of available learned-stack, run-6, graph-sibling and collective scores with weights 2, 1, 1 and 16
its cut is 0.8345136046409607
the category-swap rule removes same-house, one-word substitutions whose observed direction is sufficiently balanced with its reverse in the original french candidate pool

the final-france variant additionally removes a selected pair whose only added core token is `groupe` immediately before a recognized legal suffix
after-suffix aliases are retained
this changes three pairs relative to sprint2; its isolated accuracy effect is unmeasured

## 6. results

sprint2's team-reported public macro f0.5 is 0.990284
its output contains 5,866,303 matches
the final-france file contains 5,866,300 matches; a separate public result was not recorded

the collective-heavy france blend scored development macro f0.5 0.9924115550 on labeled india/us records
this is a decision-selection measurement rather than france accuracy
the score history and evaluation scope are documented in `code/business_entity_resolution/docs/results.md`

## 7. reproduction and checks

the top-level source README gives the exact replay command
provided raw test files are checked against recorded fingerprints
the replay verifies the cached score inputs, recreates matching and candidate tsvs and requires their hashes to equal the submitted files

checks cover reference coverage, target ownership, valid ids and candidate membership
the full training, retrieval, scoring and fusion workflow is documented separately from cached replay
model identities, licenses and configuration provenance remain with the source

## 8. artifacts

```text
output/matching_results.tsv
output/candidate_pairs.tsv
code/business_entity_resolution/src/
code/business_entity_resolution/configs/
code/business_entity_resolution/assets/
code/business_entity_resolution/docs/
code/business_entity_resolution/README.md
code/business_entity_resolution/requirements.txt
manifest.json
```
