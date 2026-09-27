# learned cached cpu retuning

best recorded public leaderboard f0.5: **0.990284** for the later sprint2 release
our team completed the 2,560-trial search and frozen development comparison below
the detailed run measurements are distinct from the [final team leaderboard result](final-result.json)

## measured starting point

- user-reported public score: 0.986416 for the first learned france-rule tsv
- uploaded matching sha256: `dd16e70f5baa7d0174d45deab1f92df1db84dd069550174a6c0ce348f42e3bce`
- fixed fitting cache: 322,009 pairs × 101 features; evaluation cache retains all competing owners for 420,456 touched targets
- [baseline diagnostics](learned-r2-baseline.json): 621 missing true candidates, 3,442 true candidates losing to another owner and 2,920 winning truths rejected; 340 retained false pairs
- these counts describe partition1, not unseen france labels or a new blind evaluation

## scorer correction

the previous search broke equal calibrated scores with the old base probability; deployment uses the candidate model's probability
a two-owner example reproduced search f0.5 = 1 while deployment chose the rival and scored 0
the search now uses the deployed tie-breaker; `src/opt.py check` covers the mismatch

each of two reference folds now receives calibration fitted on the other fold
all competing owners remain present during winner selection; full truth degrees include missing candidates
the unchanged selected model scores 0.9904283336 with this objective, versus 0.9905330809 with self-calibration and corrected ties
old objective values must not be mixed into the new studies

## search

- 2,560 trials: three independent 768-trial studies on 64-core workers and one 256-trial study on a 16-core worker
- 208 cpu cores, distinct sampler seeds; all inputs reuse verified staged score matrices
- seed each study with the current selected parameters and a conservative baseline
- vary tree size/depth, learning rate, round limit, leaf/hessian constraints, l1/l2, split gain, row/column sampling, class weighting, histogram resolution, path smoothing and extra trees
- native feature-gain masks compare all features, generator-free features, score-only features, large-model members and exclusion of bge members
- family masks are bound to the frozen 15-member score configuration; shared mean logits are excluded when a family is excluded
- retain trial models, metadata and study journals with the experiment outputs
- keep partition2 out of search; its prior checkpoint result has already been viewed, so later comparisons are separate development checks rather than a pristine blind test

## frozen finalist comparison

all 2,560 trials completed; one winner per study was frozen before the partition2 comparison
the cached evaluator exactly reproduced the existing full-population checkpoint result

| model | search macro f0.5 | development macro f0.5 | trees |
| --- | ---: | ---: | ---: |
| existing public checkpoint | 0.990428334 | 0.990353893 | 182 |
| run00 | 0.990554707 | 0.990390528 | 514 |
| run01, selected | 0.990584158 | 0.990555257 | 180 |
| run02 | 0.990632547 | 0.990369390 | 377 |
| run03 | 0.990556652 | 0.990351300 | 112 |

run01 adds 0.000201364 on development, or 0.0201364 percentage points
pair precision rises from 0.998561366 to 0.998570048; recall rises from 0.972649090 to 0.973080863
this is a small development gain, not a measured leaderboard increase
[complete comparison](learned-r2-finalists.json) · [cache parity](learned-development-cache-parity.json)

## france and calibration

the submitted recipe explicitly uses gate probabilities for france; changing lightgbm alone leaves that route unchanged
compare gate, neural and stack score routing with country-transfer calibration stress and unchanged full target competition
the rule-only score effect remains unidentified from one aggregate public result
country-transfer checks are proxies, not measurements of france accuracy

`src/xcal.py run` compares seven score policies and three calibration policies with full target competition
the current gate/density route scores 0.961734 on india and 0.960991 on us when calibration is borrowed from the other country
full-stack empirical calibration scores 0.992008 and 0.989351 respectively; the matcher itself was trained on both countries, so this is a calibration-transfer probe rather than a true unseen-country training experiment
[complete transfer results](learned-country-transfer.json)

`src/post.py fit --blend-unseen --empirical-unseen` exposes that alternative for unlabelled countries
its candidate tsv has 5,814,727 matches and the identical 14,146,782 candidate pairs; all india/us reference rows are byte-equivalent in match content
65,634 france reference rows change, with 35,694 added pairs and 44,062 removed pairs
strict and official id-enabled validation passed; public effect remains unmeasured: [variant evidence](learned-france-stack-variant.json)

development feature preparation now has an explicit partition2 mode, and search rejects development-purpose caches
this avoids recomputing the full feature corpus for every finalist while keeping development labels out of the searches

our team investigated france-specific candidates while preserving the uploaded india/us predictions
france has 259,452 of 1,732,544 test references, or 14.9752%
under that country mix, closing the public gap from 0.986416 to 0.99 through france alone would require about 0.023933 additional france macro f0.5
the actual public country mix and france labels are unknown, so this is a scenario calculation
`src/fra.py` audits gate/stack routing, density/empirical calibration and rules on/off, checks cross-country rival separation and reproduces existing full exports before comparing the changed pairs

## release gate

freeze a small finalist set before development comparison, calibrate on partition1 and validate full exports against original challenge ids
export the same actual pre-matcher candidate set and preserve the uploaded checkpoint
record the final team result separately from every checkpoint-specific receipt
