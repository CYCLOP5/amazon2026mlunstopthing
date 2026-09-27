# business entity resolution

our team built a multilingual business-matching pipeline from supplied records, compact candidate sets and calibrated entity-level decisions

**team amazites · best recorded public leaderboard f0.5: 0.990284**

varun jhaveri · shivsharan sanjawad · raj mathuria · aastha singh

## final overview

our final pipeline combines learned retrieval and neural pair scores with run-6, hybrid, residual and collective-graph evidence
the two release variants use frozen country decisions and a france-specific filter
the complete integrated source and reproducible archive layout are in [submission](submission/README.md)
we identified candidate loss, sampled-reference leakage, calibration drift and inconsistent tie-breaking through controlled comparisons
the measured run-level results and reproducibility records are indexed in [status](docs/status.md) and [evidence](reports/README.md)

| variant | retrieval | learned gate | final matcher |
| --- | --- | --- | --- |
| baseline | lexical + e5-base + qwen | original lightgbm/catboost mean | fine-tuned e5 pair classifier |
| upgrade | lexical + e5-base + qwen + e5-large | verified 54-feature lightgbm | fine-tuned e5 pair classifier |
| learned pipeline | lexical + task-trained e5-small + reverse ranks | adaptive 103-feature lightgbm | 15 cross-encoders, 101-feature stack and calibrated set decoding |
| final fusion | union of the learned and earlier score pools | preserved upstream gates | collective graph, country cuts and a frozen france blend |

the learned gate retains probabilities >=0.001, up to 50 candidates per target, with one fallback
the final releases contain **20,177,322 candidate pairs**, averaging **11.6461 per source1 business**
the earlier learned-only export contains 14,146,782 pairs and is retained as a separate checkpoint
all **20,289,808 train/test targets** have saved scores, including all 15 neural member columns
cpu retuning completed 2,560 trials; its selected finalist scored **0.990555257** on separate development references, versus **0.990353893** for the preceding learned checkpoint
development measurements and per-archive public results are recorded separately from the final team leaderboard result

## docs

| doc | contents |
| --- | --- |
| [arch](docs/arch.md) | data model modules retrieval gates matcher calibration caches and lifecycle |
| [ops](docs/ops.md) | parallel scoring cpu export validation packaging and recovery commands |
| [training](docs/training.md) | native candidate/gate fit neural training and checkpoint provenance |
| [status](docs/status.md) | final team result and checkpoint-specific evidence |
| [evidence](reports/README.md) | measured results and their evaluation scope |
| [research / eda](plan.md) | primary sources dataset analysis and decision history |
| [methodology](Documentation_template.md) | team methodology, training and measured results |
| [learned reproduction](docs/learned-submission.md) | offline model paths, exact inference options and cpu tuning workflow |
| [team method integration](reports/team-integration.md) | implemented methods, measured development results and reproduction commands |
| [additional findings](reports/additional-workspace-findings.md) | recovered experiment evidence and the sibling-context projection pitfall |
| [final packages](submission/README.md) | sprint2 and final-france archives, exact replay and full pipeline source |

## why infer and validate

**test inference creates the predictions for the competition's unlabeled records**
without those predictions there is no matching file to upload

validation scores known records to choose the acceptance cutoff and check accuracy
it can run at the same time as test inference
export waits for complete test scores and the calibration for that exact model version

```mermaid
flowchart LR
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
- deployed pretrained sources must meet the challenge's model and mit/apache-2.0 licensing requirements
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
real-model smoke tests and exact feature/checkpoint parity are linked in the evidence index

## entry points

| module | responsibility |
| --- | --- |
| `src/eda.py`, `src/probe.py` | supplied-data analysis and diagnostic probes |
| `src/data.py` | parquet prep and entity-grouped folds |
| `src/block.py` | lexical candidates and complete field similarity scores |
| `src/feat.py`, `src/train.py` | native feature contract and tree models |
| `src/tm_rules.py`, `src/tm_prep.py`, `src/tfeat.py` | verified lexical feature backend |
| `src/embed.py`, `src/hybrid.py` | pinned multilingual encoders and hybrid retrieval |
| `src/neural.py` | hard-negative pair data fine-tuning and pair scoring |
| `src/retr.py`, `src/retr_eval.py` | grouped compact-encoder fitting and complete-reference validation |
| `src/rfeat.py`, `src/norm2.py`, `src/reverse.py` | corpus-aware features, fold-safe normalization and optional reverse retrieval |
| `src/stack2.py`, `src/post.py`, `src/decode.py` | pairwise stack, score-density calibration and expected-f0.5 sets |
| `src/opt.py` | parallel macro-f0.5 hyperparameter search with complete competitor incidence |
| `src/xcal.py`, `src/fra.py` | country-transfer and france policy diagnostics |
| `src/rescore.py` | replay a selected stack over cached train/test scores |
| `src/match.py`, `src/run.py` | learned filtering neural inference shards and resume |
| `src/infer.py` | full-pool calibration and final tsv export |
| `src/validate.py` | ids coverage duplicates candidate membership and size stats |
| `src/package.py` | offline reproducibility archive |
| `src/cloud.py` | azure ml execution and artifact transport |
| `submission/src/finish.py` | frozen sprint2/final-france replay with exact output-hash checks |
| `src/release.py`, `src/release_assets.py` | complete submission ZIPs and compact replay inputs |
| `src/code_style.py` | scope-aware local-name and comment cleanup |

see the [runbook](docs/ops.md) for exact commands
validation and test partitions are separate jobs; do not put the test workload behind the full validation workload
cpu is appropriate for prep tree models calibration export and checks
gpu is preferred for this corpus's embedding and neural matching workload

## model and evidence summary

- 2,206,821 training refs and 1,732,544 test refs
- 10,320,219 labeled targets and 9,969,589 competition test targets
- france is about 15% of test refs and has no labeled training counterpart
- the final model inventory and stage roles are documented in [model sources and licenses](submission/docs/models.md)
- our feature implementation matched 54 values and checkpoint predictions exactly on the parity sample
- full scored coverage includes every labeled and test target
- the 2,560-trial cached search selected a finalist at development macro f0.5 0.990555257
- best recorded public leaderboard f0.5: 0.990284 for sprint2

immutable model-source revisions and licenses are in [model sources](reports/model_sources.json)
upstream notices are in [licenses](licenses/readme.md)

## output and packaging

the current two-release source project is under `submission/`
the binary score assets and ZIPs are retained outside git
use the `assets/` directory from an extracted release, or the locally collected `artifacts/package-assets/`, when rebuilding:

```sh
uv run python src/release.py --assets artifacts/package-assets --out artifacts/final-packages
```

the builder checks each submitted TSV's hash before packaging, writes a complete archive manifest, and verifies every member hash and CRC
the two output directories are `sprint2/` and `final-france/`, each containing `Amazites_submission.zip`

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

the learned cross-encoders trained on entity-isolated three-million-pair hard-pair datasets
full scoring used 16 disjoint assignments with explicit device and target ownership
our team reused shared reference embeddings, completed score batches and full-corpus cpu caches
see [training](docs/training.md), [arch](docs/arch.md), and [reproduction commands](docs/ops.md)
