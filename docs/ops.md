# ops and submission runbook

> updated planning cutoff: sunday 2026-09-27, 21:00 ist / 15:30 utc
>
> attempts: three total, zero used at the last user confirmation
>
> scored artifacts retain their model data and runtime provenance

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

do not use a submission attempt on a file that fails these checks
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

## 7. azure execution and checkpoint reuse

`src/cloud.py` accepts shared remote inputs
upload a shared local model once before parallel submission, verify its bytes, then use that datastore uri in every job
the sdk's concurrent upload path previously raised `BlobAlreadyExists` when eight jobs tried to upload one folder

outputs and reusable checkpoints are stored at explicit datastore uris

### portable checkpoint identity

the original multi-gpu pass produced 41 verified manifests covering 9,161,442 targets
resumed workers use a byte-verified runtime archive and a copied checkpoint tree
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

1. finish upgraded test scoring and strictly validate the first provisional export; upload and record the returned score
2. use upgraded full-pool validation and candidate-count evidence for the next justified improvement
3. use remaining validation/leaderboard evidence for a final cutoff blend or blocking refinement

the third configuration is not predetermined before the first feedback arrives
the upgraded run retains component scores to support later analysis
any changed candidate limit must still export the real pre-matcher candidate set

track the portal's attempt count after each upload
the last confirmed count was zero used out of three
all intended uploads must finish before the updated planning cutoff, sunday 2026-09-27 21:00 ist

## 9. leaderboard-first export

the live leaderboard requires `matching_results.tsv`
the full source/model/output archive is a separate final deliverable
full labeled-pool calibration may continue independently of an initial provisional upload

```sh
uv run python src/infer.py export-provisional \
  --data cache/data --runs artifacts/final_test/countries/* \
  --cutoff 0.8 --decoder target_top1_then_threshold --out output/v1

uv run python src/validate.py validate \
  --matching output/v1/matching_results.tsv \
  --candidate output/v1/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

`0.8` is an explicit provisional operating point, not a claim of optimal full-pool performance
its diagnostic comparison is recorded in [selection evidence](../reports/provisional_selection.json)
the command requires complete test coverage compatible configurations and valid scored parts
it records `provisional-export-selection` metadata instead of claiming completed calibration
the candidate tsv remains the full pre-neural candidate set regardless of the acceptance cutoff
run the supplied validator as well before using an attempt

## 10. redistributing unfinished units

`src/scatter.py` resumes an explicit assignment of existing country/rid work units
the plan must own every original unit exactly once across its workers
each worker verifies the original data/model owner and copies only manifest-listed checkpoint parts
completed targets are reused; missing targets are scored with the same runtime parameters
all assigned target ids must be covered exactly once before a worker reports completion

the real-model recovery check removed a persisted batch, recomputed it, and reproduced all 36 pair identities and probabilities with zero difference
the larger handoff preserved 4,418,033 completed test target scores
the revised plan uses up to 16 single-a100 workers with two 12-thread processes each
workers with no remaining computation reuse completed artifacts without allocating a gpu

the vm size is fixed while a job runs
increasing quota or a cluster node limit does not automatically distribute its existing python processes
checkpoint redistribution preserves useful work while making extra workers productive

## 11. after scoring completes

split-only scoring jobs produce score shards rather than final upload files
combine complete manifests, calibrate, export, validate, and package the chosen variant
record the portal result and remaining attempt count after each upload
completed artifacts retain their dataset model and runtime provenance
