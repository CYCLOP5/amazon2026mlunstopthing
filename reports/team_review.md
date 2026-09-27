# team findings and controlled comparisons

## inputs reviewed

our team reviewed the dataset eda, pair-feature tables, model diagnostics and earlier pipeline implementations
we retained the source snapshots and benchmark evidence and ran controlled comparisons with uv-isolated dependencies

## useful findings

- our full-data audit found that 38.31 percent of source1 records share a lowercase name with another source1 in their country
- transliteration and consonant-skeleton views complement literal spelling comparisons
- compact names, legal forms, alias markers, normalized address tokens and per-target candidate gaps provide useful evidence
- house-number edits overlap between true aliases and distractors. prefixes, suffixes, leading zeros and numeric distance should be soft evidence rather than rejection rules
- source1 has no blank names or addresses in the supplied data, while missing target addresses require a name-focused retrieval path
- the test set contains france without labeled french training records. cross-country validation is a stress test, not a guaranteed bound on french performance

## score and version caveats

the diagnostics report `0.9688` validation macro f0.5 for `lgbm_big`. the improvement note cites an older `0.958` leaderboard result and states that its new two-lightgbm plus xgboost implementation had not been run against the dataset by that note's author. these are different measurements and versions

the uploaded `features.py` does not define `hn_digit_diff`, `hn_log_absdiff`, `hn_prefix` or `hn_suffix`, although the diagnostics and conclusions discuss them. no missing feature was silently invented in the replication

the supplied training script filters candidate rows to validation source1 hash buckets before target-level assignment. this changes the competing-owner set relative to full inference. its score cannot be directly equated with our full-target calibration

## controlled fit

the adapter used 1858304 fit pairs and 716835 validation pairs from the existing entity-disjoint lexical runs. there were 13827 positive validation targets and 13340 retained true pairs. all alternatives for each target were kept before ranking

we compared our text preprocessing and feature functions with `bscore = ns + ads` from stored character scores. the token-skeleton blocker was evaluated separately. model fits were bounded at 800 rounds with early stopping and four cpu threads

### sampling trap caught

reference-side candidate counts computed only from owner-sampled queries reveal the sampling process. removing `s1_ncand`, `s1_bmax`, `s1_ntop1` and `s1_gap` reduced lightgbm precision at cut 0.5 from 99.77 to 98.36 percent. the stronger original number is not promotion evidence

these aggregates may be useful when computed over the full target corpus. the safer comparison below omits them

### matched-precision recall

recall denominator is all 13827 selected positive targets. this is not official full-target macro f0.5

| model | precision at least 0.990 | at least 0.995 | at least 0.999 |
| --- | ---: | ---: | ---: |
| current lightgbm | 86.21% | 79.21% | 40.01% |
| current catboost | 87.92% | 83.89% | 56.71% |
| lightgbm a without sampled s1 aggregates | 92.70% | 91.21% | 84.94% |
| lightgbm b without sampled s1 aggregates | 92.69% | 91.25% | 84.47% |
| xgboost without sampled s1 aggregates | 92.43% | 90.48% | 81.81% |
| safe three-model mean | 92.95% | 91.26% | 84.67% |

xgboost stopped at iteration 753. the mean's gain is small and operating-point dependent. the supplied representation and lightgbm configuration jointly help; their individual contributions were not isolated

## paired neural comparison

all 716835 lexical pairs joined cached neural scores by exact target/reference identity with no missing pairs or label mismatches. each gate selected its own top three; the neural probabilities and 0.6 log-odds blend were held fixed

| gate plus same neural model | recall at precision 0.990 | 0.995 | 0.999 |
| --- | ---: | ---: | ---: |
| current mean gate | 93.86% | 91.89% | 82.58% |
| validated lightgbm a | 94.29% | 93.32% | 88.44% |
| validated three-model mean | 94.35% | 93.27% | 87.72% |

this supports a broader comparison of the safe lightgbm representation. it does not establish a new full-corpus score

## third retriever correction

adding large-instruct at width 100 recovered five owner pairs on the india probe. all five survived gate top-three and became top-one matches. at final cut 0.8 there were five net additional true positives and unchanged total false positives; stricter cuts showed both gains and losses

the added retriever introduced 530145 extra first-stage pairs on 8925 queries. final learned filtering still retained three per target. small recall gains are real improvements, but final f0.5, computational work and candidate footprint must be compared together

## submission priority and updated ranking rule

our team implemented the validated gate and three-retriever combination and checked exact feature/checkpoint parity and real-runtime behavior
complete upgraded validation and test scoring run in parallel with the first baseline delivery
the remaining submission refinement will use complete validation and leaderboard feedback

the challenge update explicitly ranks candidate size per source1 as well as matching quality. current final matching receives at most three candidates per target, not a hard cap of three per source1. at full test coverage this implies approximately 29.91 million scored pairs and 17.26 pairs per source1 on average; exact distribution and tails must be measured from the completed candidate file

the exported candidate set must remain the actual input to the final matching model. post-hoc deletion based on accepted matches is not a blocking improvement

the current dense benchmark uses exact gpu search. billion-record scalability would require an approximate index and a measured candidate-count/recall tradeoff. this is a required scalability follow-up rather than a claim already established by the current million-record run
