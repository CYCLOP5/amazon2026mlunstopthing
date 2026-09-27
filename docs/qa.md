# final submission q&a and code map

## 1. which code should reviewers start with

start with [`submission/README.md`](../submission/README.md)
the integrated project contains the combined team pipeline and the final sprint's selection code
the branch's root `src/` also retains earlier experiments and operational helpers

| reviewer topic | actual source |
| --- | --- |
| task retriever and pair-model fitting | [`neural_v2/src`](../submission/src/neural_v2/src) |
| recorded selected-member training settings | [`configs/training`](../submission/configs/training) |
| lexical and run-6 stack | [`er/stack`](../submission/src/er/stack) |
| graph sibling model | [`graph_resolution`](../submission/src/fusion/business_entity_resolution/graph_resolution) |
| complementary hybrid | [`final_hybrid`](../submission/src/fusion/business_entity_resolution/final_hybrid) |
| score union and residual fusion | [`latest_fusion/pipeline.py`](../submission/src/fusion/business_entity_resolution/latest_fusion/pipeline.py) |
| collective-graph training and scoring | [`latest_fusion/model_innovation.py`](../submission/src/fusion/business_entity_resolution/latest_fusion/model_innovation.py) |
| final cut and france-blend selection | [`final_tuning`](../submission/src/final_tuning) |
| frozen final decisions and tsv export | [`finish.py`](../submission/src/finish.py) |

`finish.py` is the consolidated release replay implementation
the actual late-run selection scripts are retained alongside it
the replay was checked against the exact uploaded files, including the final three-pair positional edit

## 2. what was the final pipeline

```text
supplied records
  -> lexical and multilingual retrieval
  -> learned gate, cross-encoder ensemble and rich stack
  -> complementary run-6, sibling-graph and hybrid scores
  -> full-population residual fusion and collective graph
  -> country-specific final decisions
  -> french category-swap filter
  -> optional three-pair positional filter
  -> matching and complete candidate tsvs
```

the [architecture guide](../submission/docs/arch.md) explains the stages
the [full-pipeline guide](../submission/docs/pipeline.md) gives the source paths, environments and reconstruction commands
later stages build on earlier scores; the final system is broader than the initial lexical/gate pipeline

## 3. what moved the result above 0.99

the final work combined stronger retrieval and neural evidence with reference/target ambiguity, sibling support and full-population graph context
the last sprint reused these scores and adjusted country cuts and the france blend

| submitted version | recorded public macro f0.5 |
| --- | --- |
| combined input baseline | 0.989926 |
| sprint1 | 0.990108 |
| sprint2 | 0.990284 |

the final-france file removes three pairs from sprint2; a separate public result was not recorded
individual model contributions are not inferred from these whole-pipeline scores
the separate 2,560-trial research winner was not promoted as the released global stack

## 4. what is special about france

france has no labeled training counterpart
its final score combines the learned stack, run-6, sibling graph and collective graph with weights 2, 1, 1 and 16
the nominal collective share is 80% when all component scores are present
the available-score denominator handles missing components explicitly

the frozen france cut is 0.8345136046409607
the category-swap filter uses same-house evidence and swap/reverse counts from the original french candidate pool
the final positional rule removes only an added `groupe` before a legal suffix while retaining after-suffix variants
its isolated accuracy effect is unmeasured

the complete policy is in [`release.json`](../submission/configs/release.json) and [`final.json`](../submission/configs/final.json)

## 5. how were leakage and evaluation handled

initial encoder and pair-model fitting uses entity-grouped data
all known aliases of a reference stay in the same initial fold
population-dependent frequency, rival-owner and sibling features are computed before reference selection
metric computation retains complete truth degrees and full target competition

later fusion and collective models have their own recorded reference/name-group protocols over already-developed upstream scores
historically consulted folds are not presented as pristine end-to-end holdouts
france policy choices are not labeled france validation measurements
see [results and scope](../submission/docs/results.md)

## 6. what exactly is in the candidate file

both final releases use the same 20,177,322-pair pre-matcher union
rejected matches remain in this file
each of the 1,732,544 reference businesses has a row, including empty rows
each accepted target has one reference owner and every accepted pair is a candidate

the candidate hash is `344f114f8c81d4c806cf7019a2ebefc6cb312e16ca7ec04797701ee36a9e5e71`
the [release receipts](../reports/final-packages.json) bind each ZIP to its exact matching and candidate files

## 7. how do we prove this is the submitted output

- sprint2 replay matches its uploaded matching and candidate files byte-for-byte
- final-france replay from the original raw test files matches both expected hashes
- all archive members were checked against their manifest hashes and CRCs
- code-style changes preserved the operations, literal values and public signatures of the imported implementation
- the fusion suite passed 395 tests with one skipped

see [replay evidence](../reports/final-package-replay.json) and [exact reproduction](../submission/docs/reproduce.md)
fresh neural retraining can have numerical variation; the packaged recorded-score replay is the deterministic release check

## 8. where are the retained files

| local path | purpose |
| --- | --- |
| `artifacts/final-packages/` | verified copies of both final ZIPs |
| `artifacts/release-inputs/` | exact matching TSVs and shared candidate TSV for packaging |
| `artifacts/package-assets/` | hash-bound final replay score inputs |
| `artifacts/downloads-backup/` | original supplied source, outputs and download snapshots |
| `student_resource/dataset/`, `cache/data/` | original and prepared challenge records |

these large local files are intentionally outside git
the published branch contains the complete source, frozen policies, dependency records and compact verification receipts
model identities and licenses are listed in [the model guide](../submission/docs/models.md)
