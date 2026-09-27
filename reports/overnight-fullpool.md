# overnight full-pool diagnosis

the replay covers all 10,320,219 train targets. saved-score hashes and exact target coverage were verified before evaluation. base models were not retrained and raw features were not regenerated. the target top1 is selected across all references before anchor filtering.

base models were fit on fold2 and developed on fold0. eight full-pool variants were compared: six weighted-logit blends, a logistic stack, and a nonlinear lightgbm model. fold0 held-out anchors selected the logistic stack. its fit used fold0 anchor and fold0-owned/orphan target groups with sampled negatives inverse-weighted by 50. the full-pool tune negatives reuse fit-side target groups, so selection is transductive rather than target-group-independent. fold1 remained locked until the one-time audit.

the logistic stack uses gate and neural logits. its coefficients are 0.41550174355506897 and 0.47509413957595825, with intercept 1.8613258600234985. the selected cutoff is 0.8649235367774963.

the nonlinear variant used one fixed 150-tree, 15-leaf lightgbm model over gate/neural scores and ranks, score gaps, target address missingness, and source context. it used no reference-truth features. the saved `nonlinear.txt` SHA-256 is `76c0b5d3c1c6fefb03dd7183a05e1752bcaa68810076f82097997125f7844210`.

the CPU full-pool tune took 134.8 seconds and the locked audit took 35.6 seconds on the 12-logical-CPU, 15 GiB host.

| variant | held-out fold0 source1 macro f0.5 | full fold0 source1 macro f0.5 | cutoff |
| --- | ---: | ---: | ---: |
| nonlinear | 0.9727237071234832 | 0.972785421350731 | 0.92666095495224 |
| logistic stack | 0.9784026779294054 | 0.9785152900449455 | 0.8649235367774963 |
| weighted logit, w=0.0 | 0.9672792089951712 | 0.967599783993783 | 0.5291096568107605 |
| weighted logit, w=0.2 | 0.9746860286985396 | 0.9748252532179328 | 0.44305726885795593 |
| weighted logit, w=0.4 | 0.9778626724966529 | 0.9779305827651429 | 0.4314977824687958 |
| weighted logit, w=0.6 | 0.9781291446144351 | 0.9782766540469559 | 0.5069302320480347 |
| weighted logit, w=0.8 | 0.9753846864424981 | 0.9755431048110915 | 0.6563835740089417 |
| weighted logit, w=1.0 | 0.9714471889698871 | 0.9715470316246011 | 0.6959583163261414 |

the simple weighted-logit blend accounts for most of the fold0 gain. recalibrating w=0.6 from frozen v1's 0.8 cutoff to 0.5069302320480347 raised its fold0 score from 0.9756860071259694 to 0.9781291446144351; the logistic stack scored 0.9784026779294054. the one-time fold1 locked audit scored the selected logistic stack at 0.9786888883553567 source1 macro f0.5 and frozen v1 at 0.9758741644794233. these are offline results, not public leaderboard scores.

## candidate reachability and error slices

the fold0 postgate perfect-matcher oracle scored 0.9955514140602323 source1 macro f0.5, below 0.998, with pair recall 0.9854924640683164. this oracle is limited to the saved postgate candidate pairs. it is not a global ceiling: changing retrieval or gate candidates can change the oracle.

the json `gate_lost` value (11,080 overall) counts linked targets missing before the matcher. it combines initial retrieval misses with gate pruning; the saved final-pair artifact cannot separate those causes. the corresponding `wrong_top1` count is 4,373, and `cutoff_lost` is 29,599.

| target address | linked targets | pre-matcher missing | wrong top1 |
| --- | ---: | ---: | ---: |
| blank | 33,646 | 8,081 | 3,664 |
| nonblank | 730,095 | 2,999 | 709 |
| total | 763,741 | 11,080 | 4,373 |

blank-address targets are 4.4% of linked targets, but account for 72.9% of pre-matcher missing targets and 83.8% of wrong-top1 outcomes. the ambiguity audit found zero exact raw-input groups with multiple owners. this does not prove that blank-address cases are intrinsically irresolvable. country, blank-alias, and singleton-anchor loss tables remain in the [json report](overnight-fullpool.json).
