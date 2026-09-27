# team pipeline comparison

## scope

reference: our archived multilingual matching implementation
team-reported public score at this stage: 0.984
comparison: submitted upgraded v1 and the frozen overnight experiments in this repository

this historical comparison records how our team identified missing retrieval, feature and calibration capabilities before building the learned pipeline
the final team-reported public score is [0.989](final-result.json)

the checkout contains source and experiment prose, but no trained models or scored outputs. its final 0.9930–0.9934 values are explicitly label-free projections, not leaderboard results. the 0.984 result is not tied to a checkpoint in the supplied experiment log. the implementation differences below are verified in source; their individual leaderboard contributions are not measured ablations against this pipeline.

reference paths below are relative to that checkout. local paths refer to this repository.

## highest-impact differences

| area | current implementation | reference implementation | implication |
| --- | --- | --- | --- |
| retrieval learning | frozen e5-base, qwen and e5-large retrieval; only the pair classifier is fine-tuned | e5-small bi-encoder fine-tuned on 1.5m matched pairs; hard in-batch negatives grouped by name prefix or city | a smaller task-trained retriever is a substantive missing experiment; larger frozen encoders are not equivalent |
| evidence reaching the gate | 54-feature safe gate declares `dense_features: []`; dense hits are rescored lexically before top-3 selection | learned embedding cosine, ranks, reverse-neighbour competition, token idf and corpus-wide name multiplicities enter stage 1 | retrieval can recover a difficult pair whose dense evidence is then unavailable to the current gate |
| test-distribution calibration | a global train-selected cutoff, or logistic reblend trained on train scores | test score histograms per country and house relation; extra mid-score density lowers the inferred posterior | train calibration alone does not address a changed prevalence of look-alike negatives |
| unseen-country handling | the same neural weight or logistic stack applies to every country | blend for labelled countries, stage-1 tree for countries absent from labels; each scorer calibrated separately | france can need a different scoring policy even when the encoder supports french |
| final decision | one owner per target, then a global probability cutoff | one owner per target, then a per-reference probability-ranked prefix chosen by an approximate expected f0.5 objective, including empty | the shared one-owner constraint is already implemented; the set-level decision rule is missing |
| pair training and blending | one e5-base classifier, 502,635 distinct training pairs; overnight nonlinear stack sees scores, ranks, gaps, blankness and source | e5-small classifiers on 3.2m and 4.8m hard pairs; four-fold stage-1 scores; final blend can also see the original pair features | the overnight eight-input stack is not equivalent to their feature-rich pairwise blend |

### supervised retrieval and shortlist direction

reference `src/amzn/embed.py:66-157` trains symmetric contrastive retrieval, masks aliases of the same business as false in-batch negatives, and groups difficult negatives. defaults are 1.5m pairs, batch 384 and temperature 0.05 (`embed.py:279-286`). `embed.py:232-273` retrieves source1 to targets; reverse target-to-source1 top-5 supplies competition features.

current `src/embed.py:133-159` loads frozen pretrained retrieval weights. `src/hybrid.py:299-332` retrieves targets to source1 with several encoders. changing direction alone is not established as a gain, but the reference combines both directional evidence and task-trained geometry.

reference `src/amzn/v3.py:138-188` unions key top-30 and learned top-40 candidates per source1. its documented 0.9974 pair recall is a selected training-sample, pre-matcher measurement. current 0.98549 recall and 0.995551 macro oracle are full-fold0 postgate measurements. these are not an apples-to-apples retrieval ablation.

### the dense-feature disconnect

`artifacts/upgraded-gate/metadata.json:60-76` declares no dense features and provenance from fixed lexical candidates. `src/tfeat.py:22-25` enforces that contract. `src/hybrid.py:326-327` nevertheless computes dense similarities, while `src/match.py:309-314` selects the final shortlist through the lexical-feature gate.

reference `src/amzn/v3.py:26-33,53-127,279-320` provides learned cosine, reverse rank, idf coverage, rare missing/shared tokens and normalized-name multiplicities. the counts are computed from the full raw split, unlike the owner-sampled reference aggregates removed from the current gate. the current pipeline already uses lexical tf-idf similarities and target-side competition; the missing elements are explicit dense evidence and richer full-corpus ambiguity signals, not all idf or all context.

the 17-variant overnight tree screen held the safe lexical feature set fixed. its negative result does not establish that tree models with these missing features cannot improve.

### test calibration and france

reference `src/amzn/calibrate.py:78-107` estimates a separate posterior for each country, house relation and score bin. schematically, `p_test(bin) = clip(a * positive_train_density(bin) / test_density(bin), 0, 1)`. the factor `a` comes from very-high-score mass. extra test pairs in a bin therefore reduce confidence.

this depends on stable positive score density and reliable top-score anchors; it is an assumption to stress-test, not knowledge of test labels. the reference log reports one useful public comparison: 0.960932 before per-bin correction and 0.968379 after it (`docs/experiments.md:56-66`). that is evidence for the method in their pipeline, not a promised gain here.

`src/amzn/hybrid.py:62-77` uses the blend for countries present in labelled data and the stage-1 score otherwise. current `src/match.py:265-320` applies the same blend across countries. france has 259,452 of 1,732,544 test references, about 15% of the macro denominator. a hypothetical ten-point france-only improvement would move overall macro by about 1.5 points; france accuracy itself is unavailable.

their country projections are generated by `src/amzn/estimate.py`, which samples labels from the assumed posteriors. those projections are not independent evidence that their france fallback, or the same fallback here, achieves the projected accuracy. the current gate also lacks their dense/ambiguity features, so copying the switch without measuring it is insufficient.

### entity-level decoding

reference `src/amzn/train_v3.py:132-154` ranks candidate probabilities within each source1 entity and compares candidate prefixes with the probability of an empty entity. it uses a ratio-of-expectations approximation, not exact expected f0.5. calibrated probabilities matter.

current `src/infer.py:367-390,498-505` searches scalar cutoffs for plain or target-top1 decoding. the source1 macro metric is evaluated correctly, but the available decision rules remain restricted. a per-reference decoder is a real cpu-only experiment on cached scores.

### training scale and feature-rich stacking

reference `src/amzn/crossenc.py:50-73` uses learned-retriever hard neighbours plus every positive of selected fitting entities, including positives missed by retrieval. current `src/neural.py:182-206` already injects missing positives and samples hard/random negatives; that mechanism is not missing. the differences are learned-retriever negatives, more distinct training pairs and the downstream blend's evidence.

reference `src/amzn/train_v4.py:216-258` produces four-fold out-of-fold scores and averages folds on test. `src/amzn/blend.py:56-72,125-154` combines cross-encoder logits with name ambiguity and optionally all stored stage-1 features. their first and second cross-encoder log records 3.2m and 4.8m hard pairs, versus the current classifier's 502,635 distinct pairs and two epochs.

the reference rejected its additional stage-2 score-aggregation model after observing worse test projections despite better holdout metrics. it still retains some retrieval/sibling context in stage 1; this is not evidence that every contextual feature should be deleted. the second cross-encoder gives only a small documented holdout gain. xgboost rather than lightgbm is not isolated as the cause of the public-score difference.

## normalization: genuinely missing versus already present

reference `src/amzn/translit.py:31-80` learns native-to-latin token mappings from matched training pairs. its documentation reports 1,363 entries and 97.4% native-token coverage. current gate normalization uses generic `unidecode` plus phonetic skeletons; no equivalent learned token dictionary is used.

reference `src/amzn/normalize.py:20-29,62-75,157-180` additionally removes french additions such as `fils`, `groupe` and `developpement`, and maps department names such as `nord` and `gironde` to region tokens. current `src/tm_rules.py` already handles many french legal forms, abbreviations and regions. illustrative cpu execution confirmed:

| input | current canonical name/state | reference canonical name/region |
| --- | --- | --- |
| `ciel ecole s.a.r.l.` with `nord` | `ciel ecole`, no recognized state | `ciel ecole`, `frhdf` |
| `ciel ecole & fils` with `hauts-de-france` | `ciel ecole fils`, `hdf` | `ciel ecole`, `frhdf` |
| `ciel ecole groupe developpement` with `gironde` | additions remain, no recognized state | `ciel ecole`, `frnaq` |

these are constructed behavior checks, not measured accuracy gains. current `src/tfeat.py:96-104` already checks whether the reference's first address number occurs anywhere in the target address. the reference's additional missing number features concern nearest numeric distance and differing-digit position (`src/amzn/train_v4.py:164-190`), not simply first-number membership.

## checks against actual submitted files

country counts below were recomputed from the saved matching tsvs joined to the full test reference table. they measure prediction volume, not accuracy or the number of decoys.

| country | references | baseline matches | baseline mean | v1 matches | v1 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| france | 259452 | 919282 | 3.5432 | 856565 | 3.3014 |
| india | 809986 | 2735276 | 3.3769 | 2644842 | 3.2653 |
| us | 663106 | 2236629 | 3.3730 | 2193552 | 3.3080 |

training truth has about 3.46 links per reference in both labelled countries. these totals cannot establish whether the missing test matches are correct rejections or false negatives. direct score/house-segment histograms are needed before asserting the same decoy-density shift in this scorer.

## evidence limits and next experiments

- reference `translit.learn` uses the entire training truth, including its later holdout. fit any adopted supervised dictionary on fitting entities only for honest local evaluation.
- the reference reuses its small holdout for early stopping and many comparisons. current fold1 is already consumed and must not become a new tuning set.
- the reference documents about 108.1m final candidate pairs, versus current 29.9m. source1 mean is about 62.4 versus 17.26. a wholesale wider union has a ranking tradeoff under the organizer's candidate-size criterion.
- its score-floor cross-encoder policy rescores about 8.8m candidate pairs per small cross-encoder, while current v1 scores all 29.9m top-3 pairs with one larger classifier. counts do not by themselves establish runtime, but selective neural scoring is a useful missing efficiency experiment.

priority:

1. compare country-specific score distributions and house-relation strata; test prior correction, unseen-country scorer routing and per-reference decoding with existing scores and explicit assumptions
2. train a gate/blend on actual hybrid candidates with dense cosine/rank, full-corpus name ambiguity, idf and numeric-distance features; preserve a new independent evaluation boundary
3. fine-tune one compact retriever on grouped hard negatives, then measure initial retrieval recall, postgate recall and final source1 macro separately
4. add fold-safe learned transliteration and the verified french normalization gaps, retraining affected feature consumers rather than changing checkpoint inputs silently

the main lesson is better task learning, evidence flow and test-distribution handling. additional frozen encoders and same-feature tree tuning did not test these capabilities.
