# delivery status

> snapshot: 2026-09-25 16:27 utc / 21:57 ist
>
> deadline: sunday 2026-09-27 08:00 ist / 02:30 utc
>
> attempts used: 0 of 3 at the last confirmation

this is a timestamped operational snapshot rather than a live dashboard
no uploadable submission files or official leaderboard scores existed at this check

## completed work

- full supplied-data eda and label audit
- pinned uv runtime and offline transliteration
- native lexical/tree baseline and fine-tuned multilingual pair matcher
- verified cloud execution persistent checkpoints and task-only cleanup
- teammate code/diagnostic review and controlled lightgbm/xgboost comparisons
- exact 54-feature teammate runtime/checkpoint parity
- implemented e5-base qwen and e5-large combined retrieval
- real-model and sharded-launcher smoke checks
- strict submission validation and source1 candidate-size reporting

## live scoring snapshot

all 13 jobs reported running and had fresh output activity

| work | completed targets | required targets | progress |
| --- | ---: | ---: | ---: |
| baseline labeled pool | 9,481,218 | 10,320,219 | 91.9% |
| baseline test pool | 3,315,633 | 9,969,589 | 33.3% |
| upgraded labeled pool | 1,343,488 | 10,320,219 | 13.0% |
| upgraded test pool | 1,918,321 | 9,969,589 | 19.2% |

the baseline recovery has produced 319,776 new targets beyond its 9,161,442 restored checkpoints
completed shards were revalidated with zero new retrieval/feature/neural work before unfinished shards resumed
both reference-cache warmups succeeded

the current allocation is

- one smaller-a100 baseline recovery worker
- four baseline test workers
- four upgraded validation workers
- four upgraded test workers

the upgraded jobs use the actual teammate checkpoint and all three retrievers
the baseline uses its original frozen model/runtime

## first-submission estimate

the measured interval from 15:54 to 16:27 utc covered about 33 minutes
baseline test throughput was about 641 targets/second across four workers
the slowest quarter projected about 3.1 hours of scoring still remaining at that recent pace

the working first-file estimate is **saturday 2026-09-26 around 01:30–03:30 ist**
this includes time after scoring for calibration availability export download and strict checks
it assumes continued allocations and timely postprocessing
it is not a guarantee against preemption changing country-specific throughput or output transfer delays

| work | recent aggregate throughput | remaining scoring at that rate |
| --- | ---: | ---: |
| baseline test | about 641 targets/sec | about 2.9 h aggregate; 3.1 h slowest quarter |
| upgraded labeled pool | about 602 targets/sec | about 4.1 h |
| upgraded test | about 716 targets/sec | about 3.1 h |

these are workload estimates rather than completed results
final files do not appear merely because the scoring jobs finish
the complete manifests must still be calibrated exported and validated

## first-upload gates

| gate | state at this snapshot |
| --- | --- |
| trained baseline models | complete |
| complete baseline labeled target scores | running recovery |
| selected cutoff and locked audit | waiting on complete labeled scores |
| complete baseline test target scores | running in four partitions |
| compatible configuration and exact coverage checks | final aggregation pending |
| matching and candidate tsvs | not exported yet |
| strict id/subset/candidate-size checks | waiting for tsvs |
| first portal result | no upload yet |

## why both scoring paths exist

test inference produces predictions for the competition's unknown records
validation inference scores known records to choose the acceptance cutoff and evaluate errors
those jobs are independent after training and run concurrently
export depends on complete test scores and the calibration belonging to the same model version

cpu is sufficient for preparation lexical features tree fitting calibration export and file validation
the current embedding encoders and neural matcher benefit strongly from gpu execution
a new cpu-only neural run is not assumed to beat already-progressing gpu jobs

## budget snapshot

| measure | combined value |
| --- | ---: |
| authorized allocation caps | $1,000.00 |
| conservative accrued estimate | $471.81 |
| committed worst case including outstanding reservations | $835.36 |
| headroom against the allocated caps | $164.64 |

the two $500 ledgers cover baseline and upgrade work separately
these figures include conservative hourly ceilings and fixed allowances
they are not an azure billing invoice
task-created compute is deleted when its controlled job finishes or fails

## next three uploads

1. finish the validated first baseline and record its returned score
2. compare the upgraded full-pool results and candidate sizes before the second upload
3. use that evidence and leaderboard feedback for the final refinement

the third model/cutoff/candidate budget is not chosen in advance of feedback
all three intended uploads must occur before the sunday morning cutoff

see [arch](arch.md), [ops](ops.md), [training](training.md), and [evidence](../reports/README.md)
