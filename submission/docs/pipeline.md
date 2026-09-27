# full data-to-output pipeline

## 1. execution paths

the release can be reproduced in two ways

- exact replay uses the included, hash-bound score assets and the original raw test records
- a full rebuild regenerates retrieval, fitted models and scored pools from the original train/test data before applying the frozen final policy

the first path reproduces the exact submitted bytes
the second requires the appropriate CPU/GPU environments and can have numerical variation from fresh neural training
all final source implementations and configuration files are included

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
uv run torchrun --nproc_per_node <world> src/ce.py \
  --data artifacts/ce-hard --pairs artifacts/ce-hard/pairs-<seed>.parquet \
  --base /path/to/pinned/base --source /path/to/base/source.json \
  --out artifacts/member --batch <batch_per_gpu> --length <maxlen> \
  --lr <lr> --seed <seed> --max-seconds <max_seconds>
```

the source descriptor contains the model identity, revision and license
the recorded text format is `name: {name}\naddress: {address}\ncountry: {country}`
selected member probabilities are aggregated through mean logits

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

## 8. final release policy

`src/finish.py` implements the final, frozen decision layer described in [architecture](arch.md)
the score inputs are:

- collective-graph test predictions
- france rows from the prepared fusion table
- france rows from the run-3 score pool used to estimate word-swap direction

the packaged asset manifest binds those inputs to their original producer objects and projected columns
the selected country cuts and france blend are in `configs/release.json`
the exact-release command in [reproduction](reproduce.md) restores external ids, writes the complete candidate pool and checks the submitted output hashes
