# reproduce the releases

## 1. exact cached replay

the package includes compact recorded score inputs under `assets/`
the original raw test files are required for id mapping and the french text rules

### choose the correct archive

each variant is a separate `Amazites_submission.zip` with `output/`, `code/business_entity_resolution/`, `Documentation_template.md` and `manifest.json` at the expected locations
use the archive for the matching file that was submitted
an outer directory bundle containing both variants is a distribution bundle, not the single-variant layout

extract into a fresh directory
the commands below start at the extracted zip root

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

### raw input layout

`--data` can point to the challenge dataset directory containing `test/`, or directly to the directory containing these files:

```text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

the required columns are `entity_id`, `business_name`, `business_address` and `country`
the original bytes must match the recorded input hashes
rewriting a tsv through another tool can change quoting, row order or empty values and is therefore not accepted merely because its column names look correct

the replay assigns source1 row ids in input order and offsets source3 targets by the complete source2 count
it creates minimal working record tables under the new output directory; it does not modify the supplied raw files

## 2. prepared records

an existing verified prepared dataset can be used instead of raw tsv files:

```sh
.venv/bin/python src/finish.py --data /path/to/cache/data --out reproduced
```

the directory must contain `test/ref.parquet`, `test/s2.parquet` and `test/s3.parquet`
their fingerprints must match the recorded prepared-data version
the loader does not silently reinterpret row ids from another preparation
prepared mode expects the recorded standard tables, not an arbitrary parquet conversion or the minimal working records produced by a prior raw replay

## 3. score inputs

| file | rows | columns | role |
| --- | ---: | --- | --- |
| `collective.parquet` | 20,177,322 | `qid`, `tid`, `p` | collective probability and the complete final candidate export |
| `france.parquet` | 4,840,519 | `qid`, `tid`, `co`, `newest`, `friend`, `graph` | all france candidates with their component probabilities |
| `swap-pool.parquet` | 7,269,734 | `qid`, `tid`, `p2` | original run-3 french candidate pool for directional swap statistics |

france projection occurs after the population-dependent model features and scores have been computed
these assets do not contain an accepted-match-only replacement for candidate generation
their producer locations, byte lengths, schemas and hashes are recorded in the asset manifest
the three parquet files occupy 311,724,746 bytes before zip packaging
their azure provenance locations identify the original producers; live azure access is not required to use the packaged copies

the run-3 pool is intentionally larger than the final france pool
it supplies the original observation population for the swap/reverse counts
replacing it with only final accepted pairs would change the rule's statistical context

## 4. frozen variants

| release | matching rows | accepted pairs | candidate pairs |
| --- | --- | --- | --- |
| sprint2 | 1,732,544 | 5,866,303 | 20,177,322 |
| final-france | 1,732,544 | 5,866,300 | 20,177,322 |

each zip provides its own active `configs/release.json`
the source tree also contains `configs/final.json` for the positional variant
runtime settings are frozen; exact replay does not rerun hyperparameter search

### expected output hashes

| file | sha256 |
| --- | --- |
| sprint2 matching | `b05d40e6c914e1741ebb6c17e3c11edd7b04d90d6c9a4d1c770c409ed25ec2dc` |
| final-france matching | `e0565d95ecc02396992a8e56e9eec540438b8a255a477872a19ba25c8db9b7bd` |
| shared candidate file | `344f114f8c81d4c806cf7019a2ebefc6cb312e16ca7ec04797701ee36a9e5e71` |

the matching file uses the original source1 order
the candidate file uses the recorded country-merge row order
target ids within a row are sorted and deduplicated
these formatting choices are part of the byte-level replay contract, even though row order alone does not change the logical matching metric

## 5. rebuilding scores

[the full-pipeline guide](pipeline.md) covers generation of the score assets from the provided train/test data
that route includes retrieval, training and gpu inference
cached replay is the deterministic release check; fresh neural retraining can introduce numerical differences even with the recorded seeds

## 6. resource use

replay is cpu-only and processes the complete candidate population
use a machine with sufficient ram for the parquet joins and string aggregation
on a memory-constrained workstation, run one replay at a time
the archive builder records the environment and replay receipts used to verify each release

joins use projected score columns and id/text fields needed by the rule layer
large string-list exports are split into bounded reference chunks
the cpu process still reads the complete candidate population; a small owner sample would not be an equivalent replay

## 7. running from the git branch

the git checkout keeps code, policies and compact receipts in version control
the large artifacts are supplied separately and retained locally under `artifacts/`
from the repository root:

```sh
uv venv --python 3.11.16 .venv-replay
uv pip sync --python .venv-replay/bin/python submission/requirements.txt
.venv-replay/bin/python submission/src/finish.py \
  --data student_resource/dataset --assets artifacts/package-assets \
  --config submission/configs/release.json --out artifacts/replay-sprint2-new
.venv-replay/bin/python submission/src/finish.py \
  --data student_resource/dataset --assets artifacts/package-assets \
  --config submission/configs/final.json --out artifacts/replay-france-new
```

run the two replays sequentially on the local workstation
both output directories must be new
`artifacts/final-packages/` holds verified copies of the final zips, while `artifacts/release-inputs/` holds the exact tsv inputs used by the packager
the release code does not depend on files remaining in downloads

## 8. independent file validation

the strict validator is included at [`src/neural_v2/src/validate.py`](../src/neural_v2/src/validate.py)
it checks required reference coverage, duplicate rows, target ownership, candidate membership and external id validity
the original challenge validator is also preserved under `src/neural_v2/student_resource/utils/`

the final files were checked with the strict validator and the supplied validator with id checks enabled
the exported candidate statistics include the full histogram and nearest-rank quantiles

hash equality and format validation serve different purposes
hash equality proves replay of this specific release; validation proves compliance with the file/id/ownership contract
neither is a new accuracy measurement for unlabeled test data

## 9. packaging from verified tsvs

the repository's `src/release.py` reads the project-local release inputs and the compact score assets
for a new packaging destination:

```sh
.venv/bin/python src/release.py \
  --source submission --assets artifacts/package-assets \
  --out artifacts/rebuilt-packages
```

the builder:

1. checks each matching file and the shared candidate file against its frozen output hash
2. includes the full source tree, final selection scripts, dependency records and written methodology
3. installs the variant's active configuration under `configs/release.json`
4. adds the recorded score assets and their manifest
5. generates the zip member manifest
6. verifies crcs and every member's content hash before publishing the finished archive

compiled files, development cache directories and local environments are excluded
the exact archive hash can change when documentation or source formatting changes while the submitted tsv hashes remain fixed

## 10. interpreting failures

| failure | likely cause | correct response |
| --- | --- | --- |
| raw data hash mismatch | altered or different challenge files | restore the recorded original inputs |
| prepared hash mismatch | another preparation, row order or minimal working cache | use the recorded standard cache or raw input mode |
| asset checksum mismatch | partial transfer or wrong model-stage output | restore the file matching the manifest |
| absent score columns or invalid probabilities | incompatible score source | inspect the producer/configuration binding |
| output directory exists | attempted overwrite of a previous run | choose a new output directory |
| output hash mismatch | policy, tie, normalization, row-order or version drift | compare the frozen configuration and pinned environment; do not silently accept the new file |

## 11. scope of reproducibility

the checked release path reproduces the submitted bytes from recorded scores and original test records
the included full training/scoring source explains how those scores were produced and supports reconstruction from the original train/test data
fresh neural fitting can differ numerically across hardware or software environments

for review, follow the [architecture](arch.md), [compute plan](compute.md), [full reconstruction](pipeline.md) and [measured results](results.md) together
