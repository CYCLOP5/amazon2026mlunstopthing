# reproduce the releases

## 1. exact cached replay

the package includes compact recorded score inputs under `assets/`
the original raw test files are required for id mapping and the french text rules

```sh
cd code/business_entity_resolution
uv venv --python 3.11.16 .venv
uv pip sync --python .venv/bin/python requirements.txt
.venv/bin/python src/finish.py --data /path/to/dataset --out reproduced
```

the replay:

1. verifies every score asset against `assets/manifest.json`
2. verifies the raw challenge files against `configs/release.json`
3. prepares compact test records with the original row identifiers
4. decodes india/us from collective scores and france from the frozen blend
5. applies the category-swap rule and, for the final-france variant, the positional rule
6. generates the complete candidate tsv independently of match acceptance
7. requires both output hashes to equal the submitted hashes

the two output files are written to the requested new directory
`replay.json` records successful hash equality

## 2. prepared records

an existing verified prepared dataset can be used instead of raw tsv files:

```sh
.venv/bin/python src/finish.py --data /path/to/cache/data --out reproduced
```

the directory must contain `test/ref.parquet`, `test/s2.parquet` and `test/s3.parquet`
their fingerprints must match the recorded prepared-data version
the loader does not silently reinterpret row ids from another preparation

## 3. score inputs

| file | retained population | role |
| --- | --- | --- |
| `collective.parquet` | complete test candidate pool | collective probability and candidate export |
| `france.parquet` | all france candidates from fusion preparation | neural-stack, run-6 and graph probabilities |
| `swap-pool.parquet` | all france candidates from the run-3 pool | original best-candidate swap statistics |

france projection occurs after the population-dependent model features and scores have been computed
these assets do not contain an accepted-match-only replacement for candidate generation
their producer locations, byte lengths, schemas and hashes are recorded in the asset manifest

## 4. frozen variants

| release | matching rows | accepted pairs | candidate pairs |
| --- | --- | --- | --- |
| sprint2 | 1,732,544 | 5,866,303 | 20,177,322 |
| final-france | 1,732,544 | 5,866,300 | 20,177,322 |

each zip provides its own active `configs/release.json`
the source tree also contains `configs/final.json` for the positional variant
runtime settings are frozen; exact replay does not rerun hyperparameter search

## 5. rebuilding scores

[the full-pipeline guide](pipeline.md) covers generation of the score assets from the provided train/test data
that route includes retrieval, training and GPU inference
cached replay is the deterministic release check; fresh neural retraining can introduce numerical differences even with the recorded seeds

## 6. resource use

replay is CPU-only and processes the complete candidate population
use a machine with sufficient RAM for the parquet joins and string aggregation
on a memory-constrained workstation, run one replay at a time
the archive builder records the environment and replay receipts used to verify each release
