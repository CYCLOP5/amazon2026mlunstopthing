# ops and submission runbook

> deadline: sunday 2026-09-27, 08:00 ist / 02:30 utc
>
> attempts: three total, zero used at the last user confirmation
>
> submitted jobs retain their runtime limits output persistence and cleanup

## 1. files and environment

run commands from the repo root
raw data checkpoints caches and output tsvs are intentionally gitignored

```sh
uv python install 3.11.16
uv sync --frozen --group neural --group cloud

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

the small checks use temporary data and fake models where appropriate
real trained-model and three-encoder smoke evidence is recorded separately in [upgraded runtime smoke](../reports/upgraded_runtime_smoke.json)

```text
student_resource/dataset/       supplied raw files
cache/data/                    prepared parquet and metadata
artifacts/gate-inference/      compact original tree checkpoint
artifacts/upgraded-gate/       safe teammate lightgbm checkpoint
artifacts/neural-e5/model/     trained pair matcher
artifacts/cloud/               private job journals and runtime snapshots
output/                        exported files and final archives
```

the upgraded checkpoint is not interchangeable with a native 44-feature checkpoint
its metadata must declare `teammate-v1-nos1` and the exact 54-feature order
`src/train.py` provides the native tree training path; the teammate checkpoint was produced by the documented controlled experiment and is loaded by its verified production backend
do not claim the native training example regenerates that different checkpoint

## 2. prepare data

```sh
uv run python src/data.py \
  --data student_resource/dataset \
  --out cache/data
```

keep the resulting metadata with all later stages
changing preparation or fold assignment invalidates dependent run identities

## 3. independent scoring jobs

these examples show one full-country worker configuration
launch train scoring and test scoring on separate allocated workers
choose worker count threads and batches for the actual machine
the examples assume one 80 gb a100 and 24 cpu cores

### baseline

```sh
uv run --group neural python src/run.py \
  --data cache/data --cache cache/baseline-train \
  --gate artifacts/gate-inference --neural artifacts/neural-e5/model \
  --out output/baseline/train --split train \
  --device cuda --gpu-ids 0 --workers 2 --threads 12 \
  --retrievers e5 qwen3 --k-lex 20 --k-dense 100 --k-gate 3 \
  --encoder-batch 128 --neural-batch 128 --query-batch 4096 \
  --neural-weight 0.6 --shard-size 250000
```

```sh
uv run --group neural python src/run.py \
  --data cache/data --cache cache/baseline-test \
  --gate artifacts/gate-inference --neural artifacts/neural-e5/model \
  --out output/baseline/test --split test \
  --device cuda --gpu-ids 0 --workers 2 --threads 12 \
  --retrievers e5 qwen3 --k-lex 20 --k-dense 100 --k-gate 3 \
  --encoder-batch 128 --neural-batch 128 --query-batch 4096 \
  --neural-weight 0.6 --shard-size 250000
```

### upgrade

use the same scoring commands with

```text
--gate artifacts/upgraded-gate
--retrievers e5 qwen3 e5-large
```

use separate upgrade cache and output paths
do not resume an original-model output directory with upgraded weights or source

### currently launched outer partitions

intervals are half-open global target-id ranges

| q | validation interval | test interval |
| --- | --- | --- |
| 0 | `[0, 2600000)` | `[0, 2500000)` |
| 1 | `[2600000, 5200000)` | `[2500000, 5000000)` |
| 2 | `[5200000, 7800000)` | `[5000000, 7500000)` |
| 3 | `[7800000, 10320219)` | `[7500000, 9969589)` |

each outer partition adds `--rid-start` and `--rid-stop`
its internal work units still use `--shard-size 250000`
the union must cover each target exactly once
partial outer partitions must not individually request final calibration or export

## 4. calibrate and export on cpu

once all scoring manifests are complete, use their actual country/range directories
the wildcard examples assume local copies of the four job outputs

```sh
uv run python src/infer.py calibrate \
  --data cache/data \
  --runs output/variant/train-q*/countries/* \
  --out output/variant/calibration.json \
  --audit
```

```sh
uv run python src/infer.py export \
  --data cache/data \
  --runs output/variant/test-q*/countries/* \
  --calibration output/variant/calibration.json \
  --out output/variant/submission
```

for a single full-pool `src/run.py` output use `output/variant/train/countries/*` or `test/countries/*`
the baseline recovery job already requests full-pool calibration after it finishes its remaining work

export rejects incomplete target coverage overlapping partitions stale data and a different model/configuration fingerprint
it reconstructs external ids from prepared data and preserves empty s1 rows
the candidate tsv includes every pair actually retained for final matching rather than only accepted matches

calibration and export do not need an allocated gpu
cpu quota is useful for these stages and for prep/features/tree fitting
it does not make the neural work already in progress finish instantly

## 5. validate both required files

```sh
uv run python src/validate.py validate \
  --matching output/variant/submission/matching_results.tsv \
  --candidate output/variant/submission/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

also run the supplied challenge checker from its actual resource path

```sh
uv run python student_resource/utils/validate_submission.py \
  --matching output/variant/submission/matching_results.tsv \
  --candidate output/variant/submission/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test \
  --check-ids
```

the strict project checker reports

- required s1 and target counts
- rows empty rows and listed target counts in both outputs
- mean nearest-rank p50/p95/p99 maximum and histogram of candidates per s1
- failures for bad headers malformed rows duplicates unknown ids missing coverage or final matches outside candidates

do not spend a submission attempt on a file that fails these checks
passing checks proves structural correctness, not a particular leaderboard score

## 6. package the selected variant

the gate neural checkpoint calibration retrievers and runtime source must describe the same variant
for the interrupted baseline preserve the original immutable runtime snapshot rather than silently substituting upgraded source

```sh
uv run python src/package.py \
  --matching output/variant/submission/matching_results.tsv \
  --candidate output/variant/submission/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test \
  --repo-root . --code-root '<matching-runtime-root>' \
  --readme '<variant-readme>' --methodology '<variant-methodology>' \
  --gate-model-dir '<matching-gate-dir>' \
  --neural-model-dir artifacts/neural-e5/model \
  --calibration output/variant/calibration.json \
  --hf-cache '<local-hf-cache>' \
  --team-name Amazites --output-zip output/variant/Amazites_submission.zip
```

case-sensitive archive names model ids paths and shell variables retain their required spelling
the prose and labels use lowercase

the package includes the two outputs source uv lock selected weights tokenizer artifacts calibration provenance and model license notices
selected retriever snapshots are required and loaded locally after extraction
raw competition datasets credentials cloud caches and training feature matrices are excluded

the methodology template remains explicitly provisional until full metrics cutoff candidate counts and the actual upload date can be filled from evidence

## 7. cloud budget and recovery

the authorized total is $1,000 split between two $500 ledgers

```sh
uv run python src/budget.py --ledger artifacts/budget.json status
uv run python src/budget.py --ledger artifacts/upgrade_budget.json status
```

add the two reported spent/committed values for the project view
do not interpret either ledger alone as the total project spend
`artifacts/budget_authorization.json` records the two allocations
fixed staging allowances and hourly ceilings are conservative accounting estimates

`src/cloud.py` accepts shared remote inputs and an explicit ledger
upload a shared local model once before parallel submission, verify its bytes, then use that datastore uri in every job
the sdk's concurrent upload path previously raised `BlobAlreadyExists` when eight jobs tried to upload one folder

every managed job has finite runtime zero minimum nodes one maximum node ownership tags persistent outputs and cleanup
only task-created compute may be removed
completed outputs and reusable checkpoints remain at their recorded datastore uri

### original baseline recovery

the four-a100 node lost allocation after 9,161,442 targets had been scored
all 41 manifests had their listed files and no unlisted checkpoint parts were found
the replacement worker uses a byte-verified archive of the original runtime and a copied checkpoint tree
only its process/gpu allocation changes; model data feature precision and shard identities remain fixed
recovery includes validation/calibration only, so it does not repeat the separate test jobs

the initial recovery count is restored work
new progress is the increase beyond 9,161,442 rather than the copied files' modification time

### interpreting status

| signal | what it establishes |
| --- | --- |
| local controller running | the local wait/orchestration process is alive |
| azure job queued | waiting for allocation/startup |
| azure job running | assigned execution is active; inspect artifacts for actual model progress |
| new manifest coverage | additional targets completed |
| complete exact-coverage index | all expected targets passed aggregation checks |
| full calibration file | labeled-pool scoring and threshold selection completed |
| validated tsvs | uploadable prediction files exist |

quota is permission to request cores, not a guarantee that a vm size is available
if a spot allocation disappears, preserve its outputs and resume with matching fingerprints on available compute
do not discard completed work or assume that `running` alone proves record throughput

## 8. three-upload sequence

1. finish and validate the first baseline; upload and record the returned score
2. compare the upgraded full-pool result and candidate counts; use the next upload for the justified improvement
3. use remaining validation/leaderboard evidence for a final cutoff blend or blocking refinement

the third configuration is not predetermined before the first feedback arrives
the upgraded run retains component scores to support later analysis
any changed candidate budget must still export the real pre-matcher candidate set

track the portal's attempt count after each upload
the last confirmed count was zero used out of three
all intended uploads must finish before sunday 2026-09-27 08:00 ist

## 9. after scoring completes

split-only scoring jobs produce score shards rather than final upload files
combine complete manifests, calibrate, export, validate, and package the chosen variant
record the portal result and remaining attempt count after each upload
verified compute cleanup preserves the completed artifacts at their recorded datastore uri
