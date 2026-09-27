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
| [q&a and code map](docs/qa.md) | reviewer questions, implemented stages, evidence and limitations |
| [compute and execution](docs/compute.md) | azure machine roles, training matrix, full scoring and reliability design |
| [reproduction](docs/reproduce.md) | exact replay, input fingerprints and output conventions |
| [full pipeline](docs/pipeline.md) | rebuilding upstream retrieval, scores and graph models |
| [models and licenses](docs/models.md) | model identities, roles and upstream notices |
| [results](docs/results.md) | public results, measured scope and variant differences |
| [challenge, eda and research](docs/research.md) | input census, empirical bottlenecks, literature and the decision ledger |

## 5. source and environments

| path | purpose | environment |
| --- | --- | --- |
| `src/finish.py` | frozen final replay | top-level `requirements.txt` |
| `src/final_tuning/` | country-cut and france-blend selection | `requirements/tuning.txt` |
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

## 7. how the stages fit together

the first expensive stage is candidate discovery
lexical views retain exact text/number evidence, while multilingual and task-trained retrieval recover noisier aliases
a feature-rich gate then allocates pair-model work without restricting every target to the same small fixed candidate count

independent pair models and stacks provide complementary judgments
the run-6, sibling and bounded hybrid branches retain useful evidence that is not identical to the learned ensemble's score
the full outer score union preserves those candidate sources and explicit missing-score information

the later collective graph is a representation over target-to-target similarities
its owner hypotheses compete with a null state, and bounded cavity updates produce additional features for the final learned pair head
the graph does not manufacture new reference-target pairs

the release policy is a separate, frozen layer
india/us use collective-score cuts; france uses a weighted available-score blend and the recorded rule layers
the separation makes it possible to inspect or reproduce a decision change without pretending a new neural model was trained

## 8. what is inside the package

| area | contents | review purpose |
| --- | --- | --- |
| `output/` | exact submitted matching and candidate tsvs | score and blocking audit |
| `src/finish.py` | consolidated deterministic final replay | reproduce the submitted bytes |
| `src/final_tuning/` | actual country-cut and blend-selection source | inspect how the late policy was chosen |
| integrated model/source trees | retrieval, features, neural training, stacks, graph and hybrid code | inspect and reconstruct score generation |
| `configs/training/` | selected retriever/member configurations | bind model recipes to their recorded identities |
| `configs/collective/` | fitting protocol, selected original policy and conditional model evidence | explain the final collective scorer's training/check scope |
| `configs/execution/` | training allocation and full-scoring coverage records | trace parallel execution and completed populations |
| `assets/` | complete final candidate probabilities and france replay inputs | deterministic cpu replay |
| environment files | stage-specific pinned dependencies | distinguish different historical runtimes |
| `docs/` and methodology | detailed design, experiments, results and commands | reviewer navigation |
| `manifest.json` at zip root | file sizes and hashes | package-integrity audit |

the original raw challenge data is provided separately
the large trained bundles and full score/feature histories are separate retained artifacts; the zip's compact score assets are enough for its exact final replay

## 9. which variant should be reviewed

| variant | accepted pairs | reported public macro f0.5 | difference |
| --- | ---: | ---: | --- |
| sprint2 | 5,866,303 | 0.990284 | recorded high-performing release |
| final-france | 5,866,300 | not separately recorded | three extra positional-rule removals |

both use the same 20,177,322-pair candidate file
each archive installs its own active `configs/release.json`
the original source checkout defaults to sprint2 and also includes `configs/final.json`

do not infer that a reported score for one artifact establishes the result for another
the [results guide](docs/results.md) records the public progression and distinguishes it from development, search, candidate-oracle and conditional new-head results

## 10. implementation and evaluation lessons

the early bottleneck was not only final tree tuning
missing-address candidates, insufficient matcher evidence and an overly restrictive gate all mattered
later, france exposed the difference between a strong labeled-country score and the score path actually used in production

we therefore retained complete candidate competition, separated heavy scoring from final decisions, preserved member probabilities, and made model/data/schema identity part of every reusable artifact
the inherited sibling-projection caveat and historical audit reuse are documented explicitly

the source is organized so a reviewer can move from a result to its policy, from the policy to its score input, and from that score to the exact implementation and recorded training/execution configuration
