# business entity resolution

supplied business records → compact candidate sets → calibrated matching sets

> updated planning cutoff: sunday 2026-09-27 at 21:00 ist / 15:30 utc, based on 18 hours remaining at about 03:08 ist
>
> attempts: three total; recorded leaderboard feedback: baseline 0.964, upgraded v1 0.969

## current state

the optuna-selected rich stack has completed full training/test score replay, calibration, export and both submission validators

| variant | retrieval | learned gate | final matcher |
| --- | --- | --- | --- |
| baseline | lexical + e5-base + qwen | original lightgbm/catboost mean | fine-tuned e5 pair classifier |
| upgrade | lexical + e5-base + qwen + e5-large | verified teammate lightgbm, 54 features | same fine-tuned e5 pair classifier |
| selected cached-score upgrade | same upgraded candidate pool | same upstream gate | 60-feature optuna lightgbm stack, country/house posterior correction and source1 set decoding |

the upstream variants keep up to three candidates per target and use neural logit weight 0.6 before the additional stack
three per target is not a cap of three per s1
the exported source1 candidate distribution is measured separately

the selected stack scored 0.984547 on 73,752 separate development references, compared with 0.976006 for the uploaded v1 configuration on those same references. these are development results, not leaderboard feedback

validated files are under `artifacts/submission-optuna/output/`; the complete model/code archive is `artifacts/submission-optuna/Amazites_submission.zip`. it contains 5,786,357 matches and 29,908,767 authentic candidates. archive integrity and the original base-runtime source hashes were verified

the learned pipeline uses the task-trained compact retriever, adaptive 103-feature gate and 15 complete-epoch cross-encoders
full train/test scoring feeds a fresh cpu optuna search; its final matching and leaderboard results are not yet available

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
| [learned reproduction](docs/learned-submission.md) | offline model paths, exact inference options and cpu tuning workflow |
| [teammate integration](reports/teammate-integration.md) | implemented methods, measured development results and reproduction commands |
| [additional findings](reports/additional-workspace-findings.md) | recovered experiment evidence and the sibling-context projection pitfall |

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
| `src/retr.py`, `src/retr_eval.py` | grouped compact-encoder fitting and complete-reference validation |
| `src/rfeat.py`, `src/norm2.py`, `src/reverse.py` | corpus-aware features, fold-safe normalization and optional reverse retrieval |
| `src/stack2.py`, `src/post.py`, `src/decode.py` | pairwise stack, score-density calibration and expected-f0.5 sets |
| `src/opt.py` | parallel macro-f0.5 hyperparameter search with complete competitor incidence |
| `src/match.py`, `src/run.py` | learned filtering neural inference shards and resume |
| `src/infer.py` | full-pool calibration and final tsv export |
| `src/validate.py` | ids coverage duplicates candidate membership and size stats |
| `src/package.py` | offline reproducibility archive |
| `src/cloud.py` | azure ml execution and artifact transport |

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
- full scored coverage includes every labeled and test target
- 64 optuna trials selected a model at search macro f0.5 0.985203; its separate development result is 0.984547
- the selected checkpoint has no recorded leaderboard feedback

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

## execution platform

| machine | hardware | use |
| --- | --- | --- |
| local workstation | 12 cpu threads, rtx 2060 6 gib | eda retrieval experiments and runtime checks |
| `Standard_NC24ads_A100_v4` | 24 vcpus, one a100 80 gb | pair-model training and parallel scoring |
| `Standard_NC96ads_A100_v4` | 96 vcpus, four a100 80 gb | multi-gpu pilot and initial full-pool scoring |
| `Standard_E16ds_v4` | 16 vcpus | cpu aggregation and submission export |
| `Standard_E64ds_v4` | 64 vcpus | 8-process, 8-thread-per-process optuna search |

the pair classifier trained for two epochs on 502,635 hard-negative pair examples
the accelerated test pass used 16 disjoint single-a100 assignments with two 12-thread processes per worker
shared reference embeddings and completed score batches were reused
see [training](docs/training.md), [arch](docs/arch.md), and [reproduction commands](docs/ops.md)
