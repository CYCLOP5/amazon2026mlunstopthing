# overnight width diagnostic

selected lexical fold-0 cached pairs only. this is not production hybrid/full-pool validation, policy promotion, or evidence for a .998 claim.

## gap cause

- 487 of 13,827 linked targets (3.52%) have no true pair in the fixed lexical universe; width cannot recover them.
- safe k3 retains 96.00% of all linked targets; it loses 66 true pairs after lexical retrieval, including 61 on 580 raw blank-address targets.

## efficiency tradeoff

| policy | target mean | true retention | recall @ >= .995 precision | source1 mean | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k1 | 1.000 | 95.39% | 93.40% | 0.0081 | 0 | 0 | 9 |
| k2 | 2.000 | 95.81% | 93.32% | 0.0162 | 0 | 0 | 18 |
| k3 | 3.000 | 96.00% | 93.32% | 0.0242 | 0 | 1 | 22 |
| k5 | 5.000 | 96.22% | 93.32% | 0.0404 | 0 | 1 | 37 |
| k10 | 10.000 | 96.35% | 93.32% | 0.0808 | 0 | 2 | 135 |
| k20 | 20.000 | 96.44% | 93.32% | 0.1616 | 1 | 4 | 410 |
| blank10 | 3.228 | 96.32% | 93.32% | 0.0261 | 0 | 1 | 22 |
| gap1_else3 | 2.999 | 95.98% | 93.30% | 0.0242 | 0 | 1 | 22 |
| gap1_else3_blank10 | 3.227 | 96.30% | 93.30% | 0.0261 | 0 | 1 | 22 |

## reading it

- compare fixed k3 with `blank10`, `gap1_else3`, and `gap1_else3_blank10`; their rows show the concrete cached candidate-cost versus retention tradeoff without fitting a threshold to labels.
- source1 distribution includes every reference in `train/ref.parquet`; zero-candidate references are retained in its mean and tail ranks.
- this is score-only cpu work: widening raw-blank targets can recover gate-pruned candidates while gap pruning removes easy rows, but actual neural cost also depends on batching, sequence lengths, and runtime overhead rather than pair count alone.

## limits

- every policy ranks the complete paired lexical universe directly; no policy is scored on a pair intersection.
- w=.6 uses the unchanged cached neural probabilities and safe gate scores. precision curves are descriptive same-fold0 scans, not chosen operating thresholds.
- no azure, new ml job, gpu run, or production/fullpool artifact was used.
