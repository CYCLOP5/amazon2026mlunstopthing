# complete trained-model reproduction

## 1. what the full archive contains

the final-france archive includes the trained neural weights, tokenizers, upstream snapshots, learned gates, cross-fitted tree heads, calibrations, graph configuration, fitted france rules and original hard-pair training inputs
it also includes complete stage checkpoints and the compact final-score assets

these are different starting points for the same fixed final policy:

| command | starting point | work performed |
| --- | --- | --- |
| `verify` | archive contents | validates the complete required model inventory and every payload/frozen-runtime hash |
| `cold` | original raw train/test records and bundled fitted models | lexical/dense retrieval, neural scoring, run-6, graph/sibling and hybrid predictions, learned stack, full fusion features, collective graph/model and final export |
| `predict` | original raw records and the complete upstream feature checkpoint | reloads the fitted final model, recomputes complete country graph context, predicts every candidate and exports the final policy |
| `replay` | original raw test records and compact final scores | reapplies the exact frozen final decisions and writes the submitted bytes |

`cold` does not consume the `resume/` prediction/feature checkpoints
its france direction filter uses the fitted word-pair parameters in `models/france/rules.json`, just as the other fitted stages use their saved weights/calibrations
the original direction-fitting pool and its fitting code remain available for audit

the full inference pipeline does not require a live cloud workspace or model downloads
dependency installation may access the package indexes named by the pinned environment files

## 2. setup and archive integrity

work from `code/business_entity_resolution/` in the extracted zip:

```sh
uv venv --python 3.11.16 .venv
uv pip sync --python .venv/bin/python requirements.txt
.venv/bin/python src/reproduce.py verify
```

`reproduction_manifest.json` lists all model, resume and training-pair files with their sizes, hashes and retained origins
the verifier explicitly requires the selected neural members, old/new gates, retrieval checkpoint, run-6 models, graph/rank models, hybrid heads, residual/collective models and fitted rules
a score-only payload does not pass this complete-model check

the zip-root `manifest.json` additionally covers the source, documentation, environment files and submitted outputs
zip integrity verification reads every member completely, checking its crc, size and sha256

the archive uses zip64 for large members
extract it with a zip64-capable tool and leave room for both the extracted models and the chosen working directory

## 3. original challenge data

the dataset is supplied separately by the organizer:

```text
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

`cold` and `predict` validate all original file fingerprints recorded in `configs/reproduction-data.json`, then recreate the original row-id/fold preparation
the `replay` route needs only the three raw test files and can also use the exact previously prepared test tables

the main populations are 2,206,821 training references, 10,320,219 training targets, 1,732,544 test references and 9,969,589 test targets
source3 target row ids start after the complete source2 population
these row ids are never interchangeable with a filtered or differently ordered dataset

## 4. full raw-data inference from fitted weights

```sh
.venv/bin/python src/reproduce.py cold \
  --data /path/to/dataset \
  --bundle . \
  --work /scratch/amazites-cold \
  --out reproduced-cold \
  --gpus 4 --threads 48
```

use a new output directory
the original full hybrid gpu stage uses exactly four independently assigned workers
the recorded environment was a four-a100 gpu node with large host memory, with e64-class cpu capacity for full-population feature construction
the cold graph/feature stages require substantially more memory than compact replay; the original large-memory execution plan is detailed in [compute](compute.md)

### executed stages

1. verify every supplied model artifact and recreate prepared challenge records
2. create an offline hugging face cache from the included pinned snapshots
3. run the original earlier scoring runtime with its recorded retrieval, gate and pair-model configuration
4. generate the lexical candidate source
5. predict both run-6 rounds from their saved cross-fitted models
6. rebuild graph candidates/features and predict the saved binary/ranking and sibling models
7. select the original bounded hybrid target population from the regenerated graph scores
8. regenerate hybrid lexical/dense candidates and score every pair with the included base/expert checkpoints
9. apply the saved weighted hybrid residual heads
10. score the learned-retrieval candidate pool with all selected neural members and the saved rich stack
11. rebuild the complete final score union and fusion features
12. reconstruct competitive graph context and predict the saved collective model
13. apply the frozen country cuts, fitted direction rules and final positional rule
14. write both output files and require the recorded submitted hashes

this path uses the original frozen earlier/learned scorer sources under `src/frozen/`
`configs/runtime-sources.json` binds their original file hashes
the runner copies these small source trees into the working directory and attaches the offline model cache there; the supplied checkpoints are not modified

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set during reconstruction
the original azure artifact locations in historical manifests serve as provenance, not as live input dependencies

### resource and numerical contract

the recorded retrieval uses fp16 and the original a100 neural scoring uses bf16 where supported
batching, padding, selected checkpoints and worker ownership are explicit in the original configurations
the output guard detects any changed candidate pool or final decisions rather than silently calling a different result the submitted one

the entire four-gpu cold pass was not rerun during local packaging validation
the package's validation evidence separately records complete checkpoint/hash verification, offline neural loading, exact full-population final-model prediction and exact output reproduction

## 5. trained final-model reconstruction from the full feature checkpoint

```sh
.venv/bin/python src/reproduce.py predict \
  --data /path/to/dataset \
  --bundle . \
  --work /scratch/amazites-predict \
  --out reproduced-model \
  --threads 8
```

this route reads `resume/fusion/test/prepared-test.parquet`
that is the full upstream feature/candidate checkpoint, not a table of final collective probabilities
the saved residual baseline and collective reranker are loaded from `models/`
the graph representation is recomputed before prediction

countries are processed independently because graph edges are country-local and the recorded run did not activate its global message-budget pruning fallback
each country retains its complete candidate and target context
prediction is chunked after graph construction, so a small prediction batch does not remove competing owners or sibling evidence

packaging verification regenerated all 20,177,322 candidate probabilities:

| country | candidate pairs | maximum probability difference |
| --- | ---: | ---: |
| france | 4,840,519 | 0 |
| us | 6,898,320 | 0 |
| india | 8,438,483 | 0 |

the regenerated scores and fitted rule parameters reproduced both submitted tsv hashes exactly

## 6. original exact-score replay

```sh
.venv/bin/python src/reproduce.py replay \
  --data /path/to/dataset --out reproduced-replay
```

or invoke the smaller final-policy implementation directly:

```sh
.venv/bin/python src/finish.py \
  --data /path/to/dataset --config configs/final.json --out reproduced-replay
```

this reads the three compact assets under `assets/`: complete collective probabilities, france component scores and the original rule-development pool
the asset manifest checks their hashes before use
the final candidate file is exported from the complete scored union, including every rejected pair

## 7. reproduce the fitted france direction parameters

the fitted rule can be regenerated directly from the original raw test records:

```sh
.venv/bin/python src/fit_rules.py \
  --data /path/to/dataset \
  --pool assets/swap-pool.parquet \
  --config configs/final.json \
  --out rebuilt-rules.json
```

the file records 1,869 directional word pairs, the original data identity, fitting-pool hash, minimum support 20 and maximum forward share 0.7
these are word-direction parameters, not a list of accepted or rejected entity ids
runtime application still checks the current pair's normalized words and matching house evidence

the separate positional `groupe` rule remains an explicit deterministic part of the final-france policy

## 8. environments and model fitting settings

| artifact | contents |
| --- | --- |
| `requirements.txt` | exact replay/orchestration versions |
| `src/frozen/earlier/pyproject.toml`, `uv.lock` | original earlier scoring environment |
| `src/frozen/learned/pyproject.toml`, `uv.lock` | original learned scoring environment |
| `src/fusion/pyproject.toml`, `uv.lock` | run-6, graph, hybrid and final model environment |
| `models/*/` | checkpoint-specific metadata, model files, feature order and decisions |
| `configs/training/` | selected retriever and pair-model training settings |
| `training/pairs/` | actual grouped hard-pair populations and associated text tables |
| `configs/earlier-scoring.json`, `learned-scoring.json` | exact original inference contracts |

neural metadata retains learning rate, seed, sequence length, batch/world configuration, pooling/head definition, weight decay, precision and training-pair hashes
lightgbm files and their sidecars retain the fitted trees, feature order and selected settings
the broader training history remains in [pipeline](pipeline.md) and [compute](compute.md)

## 9. resumption and output verification

working-stage receipts bind command arguments, the model-manifest hash, release configuration and resulting file hashes
a changed completed stage is rejected instead of silently reused
stage logs are written under `work/stages/`
use a new working directory when changing a model or preprocessing contract

the final-france output has 5,866,300 matches and 20,177,322 candidates
every source1 record has a row in both files
accepted targets have one owner and every accepted pair belongs to the true candidate pool
the complete expected output hashes are in [`configs/final.json`](../configs/final.json)

the official validator can be run from the organizer's resource directory:

```sh
python3 utils/validate_submission.py \
  --matching /path/to/reproduced-model/matching_results.tsv \
  --candidate /path/to/reproduced-model/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

format/identity verification is distinct from matching accuracy
the team-reported public macro f0.5 is 0.990285 for this final-france artifact; no france-only accuracy figure was reported
