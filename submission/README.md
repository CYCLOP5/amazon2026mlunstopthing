# amazites business entity resolution

our team links noisy business records to a reference business using multilingual retrieval, lexical evidence, learned pair scores and graph context
the release contains the complete source pipeline, pinned environments, compact replay inputs and the exact submitted output files

## 1. reproduce the submitted files

from `code/business_entity_resolution` in the extracted archive:

```sh
uv venv --python 3.11.16 .venv
uv pip sync --python .venv/bin/python requirements.txt
.venv/bin/python src/finish.py --data /path/to/dataset --out reproduced
```

`dataset/test/` must contain the original `test_source1.tsv`, `test_source2.tsv` and `test_source3.tsv`
the command checks input fingerprints, regenerates matching and candidate tsvs, and compares their sha256 values with the submitted files
the destination must be new

the archive's `configs/release.json` selects its frozen policy
the final-france variant applies the same policy followed by a narrow positional name rule

## 2. archive layout

```text
Amazites_submission.zip
  output/
    matching_results.tsv
    candidate_pairs.tsv
  code/business_entity_resolution/
    src/
    configs/
    assets/
    docs/
    requirements.txt
    README.md
  Documentation_template.md
  manifest.json
```

the score assets are derived from the supplied challenge records
they contain the complete final candidate pool and the score columns needed for exact replay
the raw challenge dataset is supplied separately

## 3. pipeline

```text
provided records
  -> lexical and multilingual candidate retrieval
  -> learned gate and cross-encoder scores
  -> run-6 stack, graph refinement and hybrid evidence
  -> residual fusion and collective graph
  -> country-specific final decisions
  -> french category-swap filter
  -> optional positional groupe filter
  -> matching_results.tsv and candidate_pairs.tsv
```

the candidate file contains all 20,177,322 final pre-matcher pairs, including rejected pairs
it is never reduced to accepted matches

## 4. documentation

| guide | contents |
| --- | --- |
| [architecture](docs/arch.md) | data flow, ownership, models and candidate semantics |
| [reproduction](docs/reproduce.md) | exact replay, input fingerprints and output conventions |
| [full pipeline](docs/pipeline.md) | rebuilding upstream retrieval, scores and graph models |
| [models and licenses](docs/models.md) | model identities, roles and upstream notices |
| [results](docs/results.md) | public results, measured scope and variant differences |

## 5. source and environments

| path | purpose | environment |
| --- | --- | --- |
| `src/finish.py` | frozen final replay | top-level `requirements.txt` |
| `src/er/` | lexical pipeline and run-6 stack | `requirements/run6.txt` |
| `src/neural_v2/` | learned retrieval, pair scoring and neural stack | its `pyproject.toml` and `uv.lock` |
| `src/fusion/` | hybrid, graph and collective fusion | its `pyproject.toml` and `uv.lock` |

source and configuration snapshots for earlier experiments are retained for traceability
the execution path for these two releases is specified in [the full-pipeline guide](docs/pipeline.md)

## 6. integrity

- ids retain their original challenge meaning
- each target has at most one accepted reference owner
- every reference has a matching row and a candidate row, including empty rows
- every accepted pair belongs to the complete candidate pool
- input and output fingerprints are checked during replay
- source business identities are never enriched through external lookup services
