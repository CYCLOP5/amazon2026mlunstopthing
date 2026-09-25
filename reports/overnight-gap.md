# overnight gap diagnostic

all metrics below use only the supplied selected fold-0 cached rows. they are diagnostic evidence, not full-pool validation or a deployable threshold selection.

## reported public facts

- baseline public score: 0.964 (team-reported)
- upgraded public score: 0.969 (team-reported)
- stale local reports are not treated as contradicting those team-reported facts.

## causal gap

- 13,827 linked targets and 4,000 empty targets are represented.
- the fixed candidate universe contains the true pair for 96.4779% of linked targets; the remaining 487 cannot be recovered by any gate, blend, or cutoff.
- the safe gate retains 96.0006% at top-3; 66 otherwise recoverable true pairs are removed before the neural score can rank them.
- at the existing descriptive w=0.6 blend, 65 targets have a true pair in the safe top-3 but a wrong top-1. at cut 0.80, 324 correct top-1 pairs fall below the cut and 57 wrong top-1 pairs remain selected.

## score comparisons

the target-macro diagnostic averages one selected-row decision per target, including correct empty-target decisions. it is not source1 macro and cannot be compared to the leaderboard metric. pair f0.5 is reported separately from aggregate selected-pair tp, fp, and fn; it is also diagnostic-only.

| score | candidate scope | target-macro diagnostic @ .80 | pair f0.5 diagnostic @ .80 | precision @ .80 | recall @ .80 | recall at >= .995 precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cross_encoder_all_candidates | all fixed candidates | 0.9330 | 0.9750 | 0.9897 | 0.9203 | 0.8977 |
| safe_gate_top1 | all fixed candidates | 0.9340 | 0.9778 | 0.9937 | 0.9191 | 0.9121 |
| safe_top3_logit_blend_w_0.0 | safe-gate top-3 | 0.9340 | 0.9778 | 0.9937 | 0.9191 | 0.9121 |
| safe_top3_logit_blend_w_0.2 | safe-gate top-3 | 0.9417 | 0.9818 | 0.9966 | 0.9269 | 0.9330 |
| safe_top3_logit_blend_w_0.4 | safe-gate top-3 | 0.9452 | 0.9828 | 0.9965 | 0.9316 | 0.9368 |
| safe_top3_logit_blend_w_0.6 | safe-gate top-3 | 0.9447 | 0.9822 | 0.9956 | 0.9319 | 0.9332 |
| safe_top3_logit_blend_w_0.8 | safe-gate top-3 | 0.9408 | 0.9798 | 0.9936 | 0.9285 | 0.9191 |
| safe_top3_logit_blend_w_1.0 | safe-gate top-3 | 0.9331 | 0.9754 | 0.9902 | 0.9204 | 0.8985 |

## error strata

| country | targets | candidate miss | top-3 loss | wrong top-1 | cutoff miss @ .80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| india | 8,925 | 401 | 29 | 24 | 165 |
| us | 8,902 | 86 | 37 | 41 | 159 |

## field strata

| field | value | targets | candidate miss | top-3 loss | wrong top-1 | cutoff miss @ .80 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| empty_name | false | 17,827 | 487 | 66 | 65 | 324 |
| blank_address | true | 580 | 69 | 61 | 58 | 162 |
| blank_address | false | 17,247 | 418 | 5 | 7 | 162 |

no target had an empty name in this population, so an empty-name error rate is not estimable here.

among 65 wrong top-1 rows with a retained true pair, 25 share a normalized name and 1 share a normalized address with the wrong candidate. this is an exact-normalized-field hard-negative flag, not a semantic error taxonomy.

## limits

- no model, stack, blend weight, or threshold was fit or selected. all six weights and all cut points are same-label descriptive comparisons.
- labels are available only for this sampled/selected fold-0 population. no full-pool metric is claimed.
- the safe top-3 is upstream of the cross-encoder blend; a later score cannot recover a gate-pruned true pair.
