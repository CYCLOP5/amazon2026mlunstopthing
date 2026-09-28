# team method integration

the final trained assets are hosted on [Kaggle version 1](https://www.kaggle.com/datasets/cycl0p5/amazites-ml-2026-reproduction-assets/versions/1) and restored by the [verified fetch/reproduction commands](../submission/docs/reproduce.md)

this report follows the integration from the earlier feature-limited matcher to the released combined pipeline
the first sections preserve the original rich-stack development comparison; later sections explain the learned retrieval, full scoring, graph/hybrid fusion and final decision work
the [current code map](../submission/docs/qa.md), [architecture](../submission/docs/arch.md) and [compute plan](../submission/docs/compute.md) are the canonical final implementation guides

## implemented paths

- `src/retr.py`: compact e5-small contrastive fitting on fold2 identities, symmetric masked loss, grouped hard negatives, saved pair populations and resumable optimizer checkpoints. low-memory gpu training uses activation checkpointing.
- `src/norm2.py`: native-token learning from fold2 pairs only; french suffix and department normalization. department replacements apply to address components, preserving street names such as `rue du nord`.
- `src/rfeat.py`: a versioned rich gate contract combining existing string features with dense cosines/ranks/gaps, full-corpus name multiplicities, word idf overlap and numeric-distance evidence.
- `src/reverse.py`: optional full-population source1-to-target retrieval for the trained compact encoder. a faiss ivf-flat extra lane augments target-to-reference candidates; `rr_e5_small` records reciprocal reverse rank. each country bank spans its whole raw target population, including an optional blank-address-only scope.
- `src/stack2.py`: a pairwise tree stack over 60 string/corpus and gate/neural-logit features. cached scores lack retrieval columns, so those unavailable columns are excluded.
- `src/post.py`: country/house-stratum posterior calibration and optional gate fallback for countries without fitting labels.
- `src/decode.py`: target-owner selection followed by source1 prefix decisions. expected f0.5 is exact under independent calibrated pair probabilities for groups up to 64; larger groups use an explicit ratio-of-expectations approximation.
- `src/opt.py`: parallel optuna over cached features and source1 macro f0.5. it retains every candidate of every target touching a search reference, preserving every possible false merge into that reference set.

new matching runs record initial retrieval misses separately from true candidates removed by the gate. optional selective neural scoring retains the actual postgate candidate set and records which pairs received neural scores.

learned retriever manifests bind the model files to reference-cache identity. packaging includes selected learned weights and normalization assets, and rejects missing or mismatched model components.

reverse indexes bind their data, encoder, implementation and country coverage. cached postprocessing in the normal runner is checked against current run-manifest hashes and the selected base-score parent. the corpus-statistics cache includes its own builder implementation in invalidation.

target-only country labels use the existing global reference fallback with zero reciprocal-rank evidence. the reverse builder accepts `--device cpu` or `--device cuda`.

## evaluation boundaries

base models use the existing fold2 fitting boundary. the new pairwise stack splits fold0 reference identities into three deterministic partitions: fitting, calibration/search, and development comparison. known aliases follow their owner for supervised fitting. full-target negative competition is retained during evaluation.

fold0 was used during earlier base-model development. these results are development measurements, not a new blind audit. fold1 is excluded from this early rich-stack fitting/search path. later fusion/collective protocols have their own recorded scope, described below. test labels are never used for density correction.

the search cache contains 73,522 reference businesses, 978,271 incident targets and all 2,934,813 candidate pairs for those targets. truth labels outside the search reference set are removed from that cache. the source1 truth denominator still includes every alias of each search business, including retrieval misses.

## measured development result

comparison on the same 73,752 fold0 development businesses, with the complete target competition:

| method | source1 macro f0.5 | pair precision | pair recall |
| --- | ---: | ---: | ---: |
| submitted v1 configuration | 0.976006 | 0.994191 | 0.941895 |
| rich pairwise stack, empirical calibration and set decoding | 0.983963 | 0.996034 | 0.961168 |

the extra-orphan stress test duplicates each orphan target once under a distinct synthetic id. v1 scored 0.974118; the rich stack with density correction scored 0.982302. this is a controlled stress test, not a measurement of test labels.

the first real-data optuna smoke trial scored 0.984374 on the separate search partition and completed in 36.1 seconds on two cpu threads. this is not directly comparable with the development table because the reference populations differ.

the completed 64-trial search selected trial 47 with search macro f0.5 0.985203, versus 0.984374 for its queued baseline. parallel study execution took 250.9 seconds. [selected search parameters](optuna-search.json).

on the separate 73,752-reference development partition, the selected model scored 0.984547 with empirical calibration, versus 0.983963 for the original rich stack. precision was 0.996217 and recall 0.962220. the density-corrected extra-orphan stress score improved to 0.983149. [development comparison](optuna-development.json). these checks support a tuned test export; they do not establish a leaderboard score.

full measurements: [stack development](stack2-development.json) and [initial postprocessor development](post-development.json).

## runtime

local work used 12 logical cpu cores, 15 gib ram and an rtx 2060 with 6 gib vram. the compact encoder used 64-pair batches at 96 tokens with activation checkpointing. its requested fitting population was 1.5m pairs, starting from a recorded 64k-pair checkpoint; the completed pass processed 1,499,968 pairs.

the azure optuna run uses `standard_e64ds_v4`, with 8 trial processes and 8 lightgbm threads per process. the configured search is 64 trials. features are staged once, and each trial evaluates source1 macro f0.5 after empirical calibration and set decoding.

cached-score experiments still use the submitted retrieval and neural outputs. their measured gains come from the new pairwise stack, calibration and decoding. learned-retriever quality and its regenerated candidates require separate evaluation after encoder fitting.

## commands

these examples preserve the earlier rich-stack integration workflow and its feature version
the final `hybrid-v3`/ensemble/fusion reconstruction is documented in the [full pipeline](../submission/docs/pipeline.md)

prepare the environment with `uv sync --group neural --group cloud`.

```bash
.venv/bin/python src/norm2.py --out artifacts/norm2.json
.venv/bin/python src/retr.py --base cache/models/e5-small-source --out artifacts/retr-new --pairs 1500000 --batch 64 --threads 4
```

a retriever configuration file accepts a list of model/revision/checkpoint objects. checkpoint paths are relative to that file. the model must have a completed, hash-verified `retriever.json`.

```bash
.venv/bin/python src/hybrid.py --run cache/runs/v2_tr_india --out cache/runs/small_tr_india --retrievers-file artifacts/retrievers.json
```

generate corresponding train and development runs for each country, then train the new feature contract with `src/train.py --backend hybrid-v2 --normalizer artifacts/norm2.json`. matching accepts the same retriever configuration through `src/run.py --retrievers-file`; `--neural-floor` enables selective neural evaluation.

to enable the reverse lane, run `src/reverse.py --checkpoint artifacts/retr-new --out cache/reverse-new --split train --country us` for every train/test reference country. add `"reverse_root": "../cache/reverse-new"` to the compact model entry in `artifacts/retrievers.json` and regenerate gate-fitting candidates. package selected reverse indexes with `--reverse-root`. the lane's builder, reciprocal-rank lookup, reverse-only candidate recovery and package coverage are checked on a miniature corpus; large-corpus retrieval results still require the completed encoder and index builds.

the cached-score path is independently reproducible:

```bash
.venv/bin/python src/stack2.py fit --scores artifacts/post-train.parquet --model artifacts/stack-new
.venv/bin/python src/stack2.py score --scores artifacts/post-train.parquet --model artifacts/stack-new --out artifacts/stack-new-train.parquet
.venv/bin/python src/stack2.py score --scores artifacts/post-test.parquet --model artifacts/stack-new --out artifacts/stack-new-test.parquet
.venv/bin/python src/post.py fit --train artifacts/stack-new-train.parquet --test artifacts/stack-new-test.parquet --out artifacts/stack-new-recipe.json
.venv/bin/python src/post.py export --scores artifacts/stack-new-test.parquet --recipe artifacts/stack-new-recipe.json --out artifacts/submission-stack-new/output
```

new output paths are required. original submission checkpoints remain separate. `src/validate.py validate` and the official validator with `--check-ids` verify final tsv files.

```bash
.venv/bin/python src/stack2.py cache --scores artifacts/post-train.parquet --model artifacts/search-cache/fit
.venv/bin/python src/opt.py prepare --root artifacts/search-cache
.venv/bin/python src/opt.py run --root artifacts/search-cache --out artifacts/search-results --trials 64 --workers 8 --threads 8
```

the last command requires 64 available logical cpu cores. use a smaller worker/thread product on the local host. parallel trials use optuna journal storage with file locking. the output contains the selected model, parameters, individual trial checkpoints, trial records and study journal. search results require a separate development comparison and fresh test scoring before promotion.

## what was missing from the original matcher

the earlier upgraded gate declared an empty dense-feature list and used 54 lexical/string/context features
embeddings helped it retrieve candidates, but their scores and reciprocal evidence were not learned by that checkpoint
the richer implementation therefore required a new feature contract and a new fit

| gap | implemented response | preserved boundary |
| --- | --- | --- |
| generic retrieval missed difficult aliases | task-trained small encoder and reverse retrieval | fixed fitting groups and explicit index scope |
| name-only records lacked ambiguity evidence | full-reference/target name counts and token idf | corpus statistics computed over the declared population |
| similar names could hide important token changes | generator-aware additions, omissions, substitutions, initials and domain features | features do not create new labels |
| numeric overlap was too coarse | exact/first-number, one-digit and dropped-digit evidence | number disagreement is not a universal hard rejection |
| a fixed small gate list hid useful links | adaptive floor/cap and matching neural floor | actual retained candidates are exported |
| aggregate neural evidence concealed diversity | individual member probabilities and ordered feature contracts | model order and aggregation are fingerprinted |
| repeated gpu passes slowed policy iteration | persistent scored pools and cpu feature matrices | changed neural inputs require fresh dependent scores |

## learned retrieval and gate integration

the completed task retriever was evaluated against full country reference banks
top-50 true-link recall improved from 98.43% to 99.73% in india and from 99.39% to 99.67% in the us
blank-address india recall improved from 79.30% to 94.53%

the old gate was also replayed over fixed learned candidate unions
that diagnostic showed that a better retrieval representation could still lose useful candidates at a gate trained on different evidence
we therefore built the compatible rich gate and separately measured candidate-budget choices

the selected `hybrid-v3` gate uses 103 inputs, minimum one candidate, maximum 50 and floor 0.001
its owner-complete holdout retained 99.7743% of true links, averaging 1.3544 candidates per target query
the neural floor was lowered to the same 0.001 so recovered pairs were actually scored

[candidate selection](learned-candidate-selection.json) records the model identity, policy and scope
these are candidate-survival figures, not a final public score

## hard-pair and neural ensemble integration

five fixed three-million-pair datasets covered different hard-example samples
positive links were restored before the cap, and grouped validation separated reference and target-owner identities

the training matrix varied e5-small, e5-base, e5-large-instruct and bge-reranker-v2-m3, learning rates, token limits and batch sizes
most runs held global batch at 80
selected member configurations and the exact ordering are in the integrated project's training records

the cross-encoder driver owns its multi-gpu launch
it uses one process per visible gpu, an isolated rendezvous, masked mean pooling, frozen input embeddings, mixed precision, length-bucketed batches and periodic snapshots
fifteen complete-epoch members were selected; the interrupted bge checkpoint remained outside the scoring ensemble

the ensemble's aggregate is mean logit followed by sigmoid
individual probabilities remain available as `np_m0` through `np_m14`
the rich learned stack has 101 features after excluding unavailable earlier retrieval columns and adding the recorded name-change/member evidence

## full scoring and cpu handoff

the learned source inputs were frozen before the complete gpu pass
the scoring plan used sixteen disjoint assignments and within-country target ownership, rather than assuming one useful worker per country
retry/resume checks bind each output part to its ids and configuration

all 10,320,219 train targets and 9,969,589 test targets were verified
the corresponding learned-only test candidate set contains 14,146,782 pairs
the retained cpu fitting matrix is 322,009 by 101 and the active search matrix is 2,055,155 by 101

the new 64-trial search queued the prior selected settings as trial zero
on the separate 73,752-reference development comparison, the learned system scored 0.990353893 versus 0.984546610 for the preceding tuned stack
the first learned public artifact was reported at 0.986416, which motivated inspection of its actual unseen-country route

## later graph, hybrid and residual integration

the final pipeline also uses independent run-6 and graph/sibling evidence
the run-6 `p2` probability retains the historical column name `friend` in the fusion tables
the bounded hybrid augments a selected target population through lexical and dense field lanes, preserves parent pairs, and fits residual corrections with explicit negative-population weights

full residual fusion joins the learned stack, gate, neural members, run-6, sibling graph and hybrid scores
it keeps score-presence flags, disagreement, complete target competition, lexical/corpus features and numeric evidence
the newest stack's logit is the residual starting point

the collective stage then builds an independent target-similarity graph and a competitive owner/null representation over that full pool
its thirteen graph features are passed to a lightgbm pair model with 105 inputs
the recorded structured protocol separates new-head fit/calibration/check groups, while explicitly acknowledging upstream historical exposure

the conditional new-head check improved from 0.991336302 to 0.992279682 on 65,831 references
the [collective protocol and evidence](../submission/configs/collective/) preserve the selected strength, fitting rows and graph diagnostics

## final policy integration

the late sprint reused the combined model probabilities
it did not start another neural training round
india/us select the best collective owner and apply their frozen country cuts
france uses an available-score blend of learned stack, run-6, sibling graph and collective probabilities with weights 2, 1, 1 and 16

the final french category-swap filter uses the original run-3 french candidate pool to preserve its directional observation population
the additional positional variant removes three `groupe`-before-legal-suffix pairs
both variants keep the same complete 20,177,322-pair candidate file

reported public progression for the late artifacts was 0.989926 to 0.990108 to 0.990284
the last value belongs to sprint2; the final three-pair derivative was subsequently team-reported at 0.990285

the actual final selection code is retained under `submission/src/final_tuning/`
`submission/src/finish.py` is the consolidated replay entry point, checked against the exact submitted output hashes

## lessons and remaining interpretation limits

adding more trials was useful only after the feature and candidate evidence improved
the deployment route, not just the best offline model, determined which evidence france actually used
complete target competition, candidate provenance and explicit schema/model identity were as important to reproducibility as the training code

the inherited run-6 sibling projection and historical audit reuse remain documented limitations
the newer full-pool context and structured new-head protocols address specific scope problems, but do not make all earlier development exposure disappear
see [additional findings](additional-workspace-findings.md), [results](../submission/docs/results.md), [research/eda](../submission/docs/research.md) and [compute](../submission/docs/compute.md)
