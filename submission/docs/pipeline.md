# full data-to-output pipeline

## 1. execution paths

the release can be reproduced in two ways

- exact replay uses the included, hash-bound score assets and the original raw test records
- a full rebuild regenerates retrieval, fitted models and scored pools from the original train/test data before applying the frozen final policy

the first path reproduces the exact submitted bytes
the second requires the appropriate cpu/gpu environments and can have numerical variation from fresh neural training
all final source implementations and configuration files are included

### working directories

inside an extracted release, the submission-project root is `code/business_entity_resolution/`
inside the git checkout it is `submission/`
all paths below are relative to that root unless a section explicitly changes into a nested environment
paths in angle brackets denote existing stage artifacts, not values that should be passed literally

### stage dependency map

| stage | main input | output consumed downstream | source |
| --- | --- | --- | --- |
| preparation | original records and training links | row-id tables, ownership, degrees, folds, metadata | `neural_v2/src/data.py` |
| earlier retrieval/scoring | prepared records and pinned earlier models | neural/lexical candidate scores | `fusion/upstream_neural/src/` |
| learned retriever | fold-2 positive groups and hard pairs | hash-bound encoder bundle | `neural_v2/src/retr.py` |
| learned gate | forward/reverse candidates and rich features | gate model, normalizer, reverse indexes | `neural_v2/src/full.py gate` |
| learned pair ensemble | fixed hard-pair files and base snapshots | selected member bundles | `neural_v2/src/ce.py` |
| full learned scoring | frozen retriever, gate and ensemble | train/test member-score tables | `neural_v2/src/full.py score` |
| rich stack/calibration | learned score tables and corpus features | newest stack scores and calibrated decisions | `neural_v2/src/full.py finish` |
| run-6 | earlier scored roots and lexical candidates | `p2` score tables | `er/stack/pipeline.py` |
| graph/sibling branch | run-6/equivalent earlier evidence | `head` score tables | `fusion/business_entity_resolution/graph_resolution/` |
| bounded hybrid | graph parent, selected targets and extra retrieval | complementary `p` tables | `fusion/business_entity_resolution/final_hybrid/` |
| residual fusion | complete union of all score sources | selected head plus incumbent decisions | `fusion/business_entity_resolution/latest_fusion/pipeline.py` |
| collective model | residual baseline and label-free graph representation | full collective train/test probabilities | `fusion/business_entity_resolution/latest_fusion/model_innovation.py` |
| final release | frozen scores, raw text and final policy | exact matching/candidate tsvs | `finish.py` |

the nested paths in this table are under `src/`
the [architecture](arch.md) explains the transformations and [compute guide](compute.md) explains the execution plan

## 2. prepare challenge records

```sh
cd src/neural_v2
uv sync --frozen
uv run python src/data.py --data /path/to/dataset --out cache/data
```

this retains raw text, maps external ids to row ids, derives normalized comparison fields and creates entity-grouped training folds
`meta.json` binds the prepared population to the original files

the other source stacks consume the same prepared records or their own documented lexical preparation
do not interchange row-id populations without checking their metadata

## 3. earlier retrieval and pair scores

the earlier neural implementation is under `src/fusion/upstream_neural/`
its environment is pinned by its own `pyproject.toml` and `uv.lock`
its `src/data.py`, `src/block.py`, `src/train.py`, `src/neural.py`, `src/run.py` and `src/infer.py` implement preparation, retrieval, gate fitting, pair training, full scoring and calibration

model ids and immutable revisions are in its `reports/model_sources.json`
these earlier scored pools supply the run-6, graph and hybrid stages
`azure/_run6.yml` records the exact producer input locations used by the release's run-6 stack

## 4. task retriever and learned pair ensemble

the learned implementation is under `src/neural_v2/`
download the pinned base snapshots into the paths recorded by the model-source configuration
for example, the task retriever starts from:

```sh
hf download intfloat/multilingual-e5-small \
  --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --local-dir cache/models/e5-small-source
```

the task retriever uses symmetric `query: {name} | {address}` text, a 96-token limit and fold-2 pairs
the recorded recipe has a 64,000-pair warm start followed by the requested 1.5-million-pair pass
its exact configuration and checkpoint fingerprints are in `configs/training/retriever.json` at the submission-project root

```sh
uv run python src/retr.py --data cache/data --base cache/models/e5-small-source \
  --out artifacts/retr-warm --pairs 64000 --batch 64 --length 96
uv run python src/retr.py --data cache/data --base cache/models/e5-small-source \
  --out artifacts/retr-small --pairs 1500000 --batch 64 --length 96 \
  --resume artifacts/retr-warm
```

hard-pair preparation retains selected-entity positives and high-scoring negatives from the earlier score pool
it keeps fitting and held-out owner/reference groups separate

```sh
uv run python src/ceprep.py --scores /path/to/earlier/train.parquet \
  --out artifacts/ce-hard --seeds 21 41 51 71 131 --max-pairs 3000000
```

`configs/training/members.json` records each selected member's base model, revision, batch size, learning rate, sequence length, seed and hard-pair fingerprint
one member's training command has this form:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python src/ce.py \
  --data artifacts/ce-hard --pairs artifacts/ce-hard/pairs<seed>.parquet \
  --base /path/to/pinned/base --source /path/to/base/source.json \
  --out artifacts/member --batch <batch_per_gpu> --length <maxlen> \
  --lr <lr> --seed <seed> --max-seconds <max_seconds>
```

the source descriptor contains the model identity, revision and license
the recorded text format is `name: {name}\naddress: {address}\ncountry: {country}`
selected member probabilities are aggregated through mean logits
`ce.py` discovers the visible gpus and launches distributed workers internally with `torch.multiprocessing.spawn`
an additional outer multi-process launcher is not part of the recorded training command
the batch option is per gpu; consult the [training matrix](compute.md#51-training-matrix) for the global-batch comparisons
the shown device list is the four-gpu example; use the visible devices for the selected run
hard-pair files are named `pairs21.parquet`, `pairs41.parquet` and so on, matching the staged manifest

## 5. learned gate and full scoring

`src/fullprep.py` stages the prepared records, task retriever, hard pairs, normalizer and pinned base snapshots
`src/full.py` exposes the gate, ensemble, score and finish stages

```sh
uv run python src/fullprep.py --data cache/data --encoder artifacts/retr-small \
  --hard artifacts/ce-hard --normalizer /path/to/normalizer.json --out artifacts/assets
uv run python src/full.py gate --assets artifacts/assets --out artifacts/gate
uv run python src/full.py ensemble --members <selected-member-directories> --out artifacts/ensemble
uv run python src/full.py score --assets artifacts/assets --gate artifacts/gate \
  --neural artifacts/ensemble --split train --k-gate 50 --gate-floor 0.001 \
  --neural-floor 0.001 --out artifacts/scored-train
uv run python src/full.py score --assets artifacts/assets --gate artifacts/gate \
  --neural artifacts/ensemble --split test --k-gate 50 --gate-floor 0.001 \
  --neural-floor 0.001 --out artifacts/scored-test
uv run python src/full.py finish --assets artifacts/assets --gate artifacts/gate \
  --train-runs artifacts/scored-train --test-runs artifacts/scored-test --out artifacts/learned
```

the production scoring pass was sharded across disjoint country/target ranges
`src/fullplan.py` implements that partitioning
the saved stack tables contain the learned score plus the individual evidence columns required by fusion

## 6. run-6, graph and hybrid evidence

use the run-6 environment from `requirements/run6.txt`
the complete lexical and stack configurations are `configs/big_machine.toml` and `configs/stack.toml`
run this section from the submission-project root

```sh
python src/run.py -c configs/big_machine.toml --name lexical --only prep,blocking_train,blocking_test \
  --set 'paths.data="/path/to/dataset"'
python src/run.py stack -c configs/stack.toml --name stack_run6_swap_only \
  --set stack.swap_features=true --set stack.rules=false \
  --set 'stack.data="<prepared-data>"' \
  --set 'stack.train_roots=[<earlier-train-roots>]' \
  --set 'stack.test_roots=[<earlier-test-roots>]' \
  --set 'stack.lexical_train=["<lexical-train>"]' \
  --set 'stack.lexical_test=["<lexical-test>"]'
```

`azure/lexical_job.yml` and `azure/_run6.yml` record the release's exact lexical and run-6 jobs
the graph and hybrid implementations are under `src/fusion/business_entity_resolution/`
their executable definitions are `azure/graph_refine_job.yml` and `azure/final_hybrid_pipeline.yml` within that directory
the scripts `run_graph.py` and `run_final_hybrid.py` implement their preparation, fitting and scoring steps

## 7. full-population fusion and collective graph

run this section from `code/business_entity_resolution/src/fusion/` with that directory's pinned environment:

```sh
uv run python business_entity_resolution/scripts/run_latest_fusion.py prepare \
  --data <prepared-data> --newest <learned-output> --hybrid <hybrid-output> \
  --graph <graph-output> --friend <run6-stack-directory> --split train \
  --output <fusion-train> --expected-meta business_entity_resolution/configs/latest_fusion_data_meta.json
```

repeat preparation for `--split test`
the legacy flag `--friend` names the run-6 score input
preparation verifies data identity, forms the full score union and computes population-dependent features before selecting reference groups

```sh
uv run python business_entity_resolution/scripts/run_latest_fusion.py fit \
  --data <prepared-data> --train <fusion-train>/prepared-train.parquet \
  --test <fusion-test>/prepared-test.parquet \
  --metadata business_entity_resolution/configs/latest_fusion.json \
  --calibration business_entity_resolution/configs/latest_fusion_calibration.json --output <residual-fusion>
uv run python business_entity_resolution/scripts/run_model_innovation.py \
  --mode collective_graph --data <prepared-data> \
  --train <fusion-train>/prepared-train.parquet --test <fusion-test>/prepared-test.parquet \
  --incumbent <residual-fusion> --output <collective-graph>
```

the collective run writes full `validation_predictions.parquet` and `test_predictions.parquet` tables
its original selection and isolation protocol are preserved with the release evidence
the final score producer is `amazites-model-collective-graph-20260927-04`

the final policy-selection code is retained under `src/final_tuning/` at the submission-project root
`sprintcal.py` compares country cuts and calibrated decoding on the collective scores
`sprintfr.py` compares france blends using labeled-country development data and applies the category-swap rule
`sprintfixed.py` replays fixed comparison policies
their pinned cpu environment is `requirements/tuning.txt`
the selection scripts' `--help` lists the prepared-data and scored-pool inputs; exact release reproduction uses the frozen configuration below

## 8. final release policy

`src/finish.py` implements the final, frozen decision layer described in [architecture](arch.md)
the score inputs are:

- collective-graph test predictions
- france rows from the prepared fusion table
- france rows from the run-3 score pool used to estimate word-swap direction

the packaged asset manifest binds those inputs to their original producer objects and projected columns
the selected country cuts and france blend are in `configs/release.json`
the exact-release command in [reproduction](reproduce.md) restores external ids, writes the complete candidate pool and checks the submitted output hashes

## 9. fitting and evaluation contracts by stage

| stage | fitting boundary | selection/check boundary | interpretation |
| --- | --- | --- | --- |
| task retriever and learned pair models | original fold-2 business groups | encoder-unseen/owner-group controls | tests representation and pair-model generalization within labeled countries |
| rich learned stack | fold-0 deterministic partition 0, with eligible owner groups | partition 1 search/calibration and partition 2 development | later-stage development on previously studied upstream models |
| run-6 and sibling refinements | configured reference-group cross-fitting | documented tune split and historical fold-1 audit | historical audit reuse and projection caveats apply |
| residual fusion | fit/tune split within fold-0 partition 2 | separate calibration partition; reused fold-1 reports | conditional comparison of heads on frozen upstream scores |
| collective graph head | normalized-name groups from original fold 1 | isolated new-head calibration/check groups | new-head isolation, not pristine end-to-end holdout |
| final country cuts and france blend | labeled-country development comparisons | recorded public submissions; no labeled france counterpart | frozen release policy, with france uncertainty disclosed |

### collective protocol details

the recorded structured protocol hashes country and normalized full name with seed 2709, then assigns modulo-10 groups to fit buckets 0–4, calibration 5–6 and check 7–9
it assigns 110,186 references to fitting, 44,640 to the initial calibration population and 65,831 to checking

fitting removes candidates and true-owner groups protected by calibration/check targets
the recorded fitting table contains 796,847 pairs: 367,241 positive and 429,606 negative
calibration references sharing a candidate target with the check population are removed, leaving 20,043 clean calibration references
the recorded calibration/check target overlap is zero

the protocol still records historical upstream exposure
these isolation checks protect the new head; they do not erase earlier development use of its upstream scores or features

## 10. artifacts required for each reconstruction depth

### exact final replay

- original raw test files, or the exact prepared test tables
- `assets/manifest.json` and the three packaged score assets
- the active `configs/release.json`
- the pinned replay environment

this is the release's byte-identical reproduction route

### cpu reconstruction from full scored pools

- full train/test member and complementary score tables, not only the compact france projections
- raw/prepared training records, complete truth degrees and target owners
- full feature schemas, normalization state and existing head metadata
- the recorded fitting/group partition settings
- the relevant cpu environment lock

this route can rebuild features, fit tree/fusion heads, compare calibration and reproduce final policy selection without refitting neural models
changing a feature definition changes the cache identity and requires rebuilding that feature table

### complete neural reconstruction

- every pinned base-model and tokenizer source
- retained trained checkpoints, or the exact training-pair populations and settings needed to refit them
- lexical/reverse-index specifications and full reference populations
- full inference shard plans and output coverage checks

the large trained bundles are retained separately from the compact replay zips
the compact zips include the code and recorded scores needed to reproduce the actual submitted bytes

## 11. cache and schema boundaries

a parquet file with columns named `qid`, `tid` and `p` is not sufficient provenance by itself
the same row numbers can refer to different records after a changed preparation
each stage checks the relevant combination of:

1. original data hashes and prepared-data metadata
2. model/tokenizer file hashes and model identity
3. ordered feature names and backend version
4. normalization implementation and learned mapping
5. score configuration, candidate keys and target coverage
6. selection/calibration settings and the chosen incumbent artifact

source-code hashes participate in several historical cache keys
styling or modifying a builder can invalidate those cache keys even when the transformation is semantically equivalent
the original supplied sources are preserved with the local provenance snapshot; the final recorded-score replay validates the resulting score contents directly

## 12. why the score tables and accepted tables differ

upstream stages can emit probabilities for the full candidate union while preserving a separate accepted set for an unseen country
for example, the native collective-model export changes india/us decisions while retaining its incumbent france accepted set
its score parquet still contains collective probabilities for france candidates

the final sprint reuses that full score table and applies the explicitly recorded france blend
the final matching file must therefore be regenerated through the frozen release policy rather than copied from an intermediate model's `accepted_test.parquet`

likewise, `candidate_pairs.tsv` comes from the full final scored union
it is independent of whether a pair survives the final probability threshold or rule layer

## 13. verification sequence

1. verify model/data/feature metadata before joining or fitting
2. verify full target ownership and score coverage before calibration/export
3. select settings only within the stage's documented selection population
4. freeze a release configuration and preserve the input hashes
5. regenerate matching and candidate outputs through the final replay entry point
6. compare both output hashes with the submitted-file records
7. run strict schema, id, uniqueness and candidate-membership checks
8. package source, environments, score assets and methodology under the required layout
9. verify every zip member hash and crc

the [replay guide](reproduce.md) gives the release commands, [results](results.md) separates measured outcomes by scope, and [compute](compute.md) records the resource and reliability choices
