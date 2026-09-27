# learned cached cpu retuning

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
- retain every trial model, metadata and study journal locally
- keep partition2 out of search; its prior checkpoint result has already been viewed, so later comparisons are separate development checks rather than a pristine blind test

## france and calibration

the submitted recipe explicitly uses gate probabilities for france; changing lightgbm alone leaves that route unchanged
compare gate, neural and stack score routing with country-transfer calibration stress and unchanged full target competition
the rule-only score effect remains unidentified from one aggregate public result
country-transfer checks are proxies, not measurements of france accuracy

## release gate

freeze a small finalist set before development comparison, calibrate on partition1 and validate full exports against original challenge ids
export the same actual pre-matcher candidate set and preserve the uploaded checkpoint
finish selection and remaining public submissions before 21:00 ist on 2026-09-27
