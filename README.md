# business entity resolution

supplied business records → compact candidate sets → calibrated matching sets

> deadline: sunday 2026-09-27 at 08:00 ist / 02:30 utc
>
> attempts: three total, zero used at the last user confirmation
>
> azure authorization: $1,500 total across three disjoint $500 allocations

## current state

two implemented variants are scoring the complete datasets

| variant | retrieval | learned gate | final matcher |
| --- | --- | --- | --- |
| baseline | lexical + e5-base + qwen | original lightgbm/catboost mean | fine-tuned e5 pair classifier |
| upgrade | lexical + e5-base + qwen + e5-large | verified teammate lightgbm, 54 features | same fine-tuned e5 pair classifier |

both currently keep up to three candidates per target before final matching and use neural logit weight 0.6
three per target is not a cap of three per s1
the exported source1 candidate distribution is measured separately

the original baseline completed full-pool calibration and locked audit with offline macro f0.5 of 0.97550
upgraded calibration final test export and final output verification remain delivery gates
no official score or completed submission is claimed here

the first upgraded leaderboard upload may use an explicit provisional cutoff while full validation continues
complete test coverage and strict file checks still apply
see [leaderboard-first export](docs/ops.md#9-leaderboard-first-export)

## docs

| doc | contents |
| --- | --- |
| [arch](docs/arch.md) | data model modules retrieval gates matcher calibration caches and lifecycle |
| [ops](docs/ops.md) | parallel scoring cpu export validation packaging and recovery commands |
| [training](docs/training.md) | native candidate/gate fit neural training and checkpoint provenance |
| [status](docs/status.md) | timestamped progress dependencies and first-upload gates |
| [evidence](reports/README.md) | measured results and their evaluation scope |
| [research / eda](plan.md) | primary sources dataset analysis and decision history |
| [methodology](Documentation_template.md) | submission-method draft awaiting final measured results |

## why infer and validate

**test inference creates the predictions for the competition's unlabeled records**
without those predictions there is no matching file to upload

validation scores known records to choose the acceptance cutoff and check accuracy
it can run at the same time as test inference
export waits for complete test scores and the calibration for that exact model version

```mermaid
flowchart lr
    weights[frozen trained models] --> val[labeled-pool scoring]
    weights --> test[competition test scoring]
    val --> cal[cutoff and audit]
    test --> export[export]
    cal --> export
    export --> check[strict file checks]
    check --> upload[upload and leaderboard feedback]
```

## boundaries

- only supplied business records enter the matching pipeline
- no business lookup geocoding external translation services or external data augmentation
- preserve original unicode and use local transliteration as an additional comparison view
- deployed pretrained sources must meet the challenge's mit/apache-2.0 and 8b-parameter restrictions
- final output includes every required s1 row, including empty predictions
- every accepted match must occur in its exported candidate set
- export the actual final pre-matcher candidates, including rejected matches
- keep raw records credentials host-specific paths and model/cache binaries out of git

## setup and prep

python and dependencies are pinned by `.python-version`, `pyproject.toml`, and `uv.lock`
use the uv environment rather than global package installs

```sh
uv python install 3.11.16
uv sync --frozen --group neural --group cloud
uv run python src/data.py --data student_resource/dataset --out cache/data
```

the prepared dataset preserves original ids and text, adds compact row ids and comparison views, and freezes entity-grouped folds
fold 2 fits models, fold 0 tunes decisions, and fold 1 is the locked audit

## checks

```sh
uv run python src/data.py --check
uv run python src/block.py --check
uv run python src/train.py --check
uv run python src/tfeat.py --check
uv run --group neural python src/embed.py --check
uv run --group neural python src/neural.py check
uv run python src/hybrid.py --check
uv run python src/match.py --check
uv run python src/run.py --check
uv run python src/infer.py --check
uv run python src/validate.py --check
uv run python src/package.py --check
```

these runnable checks are separate from the expensive labeled-pool inference job
real-model smoke tests and exact teammate-feature parity are linked in the evidence index

## entry points

| module | responsibility |
| --- | --- |
| `src/eda.py`, `src/probe.py` | supplied-data analysis and diagnostic probes |
| `src/data.py` | parquet prep and entity-grouped folds |
| `src/block.py` | lexical candidates and complete field similarity scores |
| `src/feat.py`, `src/train.py` | native feature contract and tree models |
| `src/tm_rules.py`, `src/tm_prep.py`, `src/tfeat.py` | verified teammate feature backend |
| `src/embed.py`, `src/hybrid.py` | pinned multilingual encoders and hybrid retrieval |
| `src/neural.py` | hard-negative pair data fine-tuning and pair scoring |
| `src/match.py`, `src/run.py` | learned filtering neural inference shards and resume |
| `src/infer.py` | full-pool calibration and final tsv export |
| `src/validate.py` | ids coverage duplicates candidate membership and size stats |
| `src/package.py` | offline reproducibility archive |
| `src/cloud.py`, `src/budget.py` | bounded compute lifecycle and reservation accounting |

see the [runbook](docs/ops.md) for exact commands
validation and test partitions are separate jobs; do not put the test workload behind the full validation workload
cpu is appropriate for prep tree models calibration export and checks
gpu is preferred for this corpus's embedding and neural matching workload

## model and evidence summary

- 2,206,821 training refs and 1,732,544 test refs
- 10,320,219 labeled targets and 9,969,589 competition test targets
- france is about 15% of test refs and has no labeled training counterpart
- frozen baseline retrieval and its separately trained matcher total about 1.152b neural parameters
- the three-retriever upgrade and matcher total about 1.712b
- the teammate port matched 54 feature values and checkpoint predictions exactly on the parity sample
- the real upgraded smoke run covered 12 targets and 36 final candidates
- these smoke and sampled comparison results are not full-corpus or leaderboard scores

immutable model revisions licenses and parameter evidence are in [model sources](reports/model_sources.json)
upstream notices are in [licenses](licenses/readme.md)

## output and packaging

```text
output/
  matching_results.tsv
  candidate_pairs.tsv
  calibration.json
```

the final archive additionally carries source dependency locks selected model snapshots tokenizer files calibration and the completed methodology document
raw datasets must be supplied separately when reproducing the run
runtime loaders use packaged hf snapshots locally when available

the strict validator reports candidate count mean nearest-rank p50/p95/p99 maximum and the complete histogram
the organizer's final ranking reviews both matching quality and candidate generation
the current exact dense scan is memory-bounded but is not claimed to be a billion-record approximate index

## operations

the original four-a100 worker lost allocation after 9,161,442 scored validation targets
its verified checkpoints and exact runtime were retained for recovery on a smaller a100
the test workers continued independently

all task compute has finite runtime ownership tags persistent outputs and cleanup
already-submitted azure jobs retain their runtime limits and automatic compute cleanup
budget reports are conservative estimates, not an azure invoice
see [status](docs/status.md) and [ops](docs/ops.md) before allocating or resuming work
