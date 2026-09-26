# teammate method integration

## implemented paths

- `src/retr.py`: compact e5-small contrastive fitting on fold2 identities, symmetric masked loss, grouped hard negatives, saved pair populations and resumable optimizer checkpoints. low-memory gpu training uses activation checkpointing.
- `src/norm2.py`: native-token learning from fold2 pairs only; french suffix and department normalization. department replacements apply to address components, preserving street names such as `rue du nord`.
- `src/rfeat.py`: a versioned rich gate contract combining existing string features with dense cosines/ranks/gaps, full-corpus name multiplicities, word idf overlap and numeric-distance evidence.
- `src/stack2.py`: a pairwise tree stack over 60 string/corpus and gate/neural-logit features. cached scores lack retrieval columns, so those unavailable columns are excluded.
- `src/post.py`: country/house-stratum posterior calibration and optional gate fallback for countries without fitting labels.
- `src/decode.py`: target-owner selection followed by source1 prefix decisions. expected f0.5 is exact under independent calibrated pair probabilities for groups up to 64; larger groups use an explicit ratio-of-expectations approximation.
- `src/opt.py`: parallel optuna over cached features and source1 macro f0.5. it retains every candidate of every target touching a search reference, preserving every possible false merge into that reference set.

new matching runs record initial retrieval misses separately from true candidates removed by the gate. optional selective neural scoring retains the actual postgate candidate set and records which pairs received neural scores.

learned retriever manifests bind the model files to reference-cache identity. packaging includes selected learned weights and normalization assets, and rejects missing or mismatched model components.

## evaluation boundaries

base models use the existing fold2 fitting boundary. the new pairwise stack splits fold0 reference identities into three deterministic partitions: fitting, calibration/search, and development comparison. known aliases follow their owner for supervised fitting. full-target negative competition is retained during evaluation.

fold0 was used during earlier base-model development. these results are development measurements, not a new blind audit. fold1 is excluded from the new fitting/search paths. test labels are never used for density correction.

the search cache contains 73,522 reference businesses, 978,271 incident targets and all 2,934,813 candidate pairs for those targets. truth labels outside the search reference set are removed from that cache. the source1 truth denominator still includes every alias of each search business, including retrieval misses.

## measured development result

comparison on the same 73,752 fold0 development businesses, with the complete target competition:

| method | source1 macro f0.5 | pair precision | pair recall |
| --- | ---: | ---: | ---: |
| submitted v1 configuration | 0.976006 | 0.994191 | 0.941895 |
| rich pairwise stack, empirical calibration and set decoding | 0.983963 | 0.996034 | 0.961168 |

the extra-orphan stress test duplicates each orphan target once under a distinct synthetic id. v1 scored 0.974118; the rich stack with density correction scored 0.982302. this is a controlled stress test, not a measurement of test labels.

the first real-data optuna smoke trial scored 0.984374 on the separate search partition and completed in 36.1 seconds on two cpu threads. this is not directly comparable with the development table because the reference populations differ.

full measurements: [stack development](stack2-development.json) and [initial postprocessor development](post-development.json).

## runtime

local work uses 12 logical cpu cores, 15 gib ram and an rtx 2060 with 6 gib vram. the compact encoder uses 64-pair batches at 96 tokens with activation checkpointing. its fitting population is 1.5m pairs. the current run starts from a recorded 64k-pair local checkpoint; the exact next-stage population is saved before training.

the azure optuna run uses `standard_e64ds_v4`, with 8 trial processes and 8 lightgbm threads per process. the configured search is 64 trials. features are staged once, and each trial evaluates source1 macro f0.5 after empirical calibration and set decoding.

cached-score experiments still use the submitted retrieval and neural outputs. their measured gains come from the new pairwise stack, calibration and decoding. learned-retriever quality and its regenerated candidates require separate evaluation after encoder fitting.

## commands

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
