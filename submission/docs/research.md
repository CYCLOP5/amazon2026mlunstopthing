# challenge, eda, research and decision history

## 1. the actual challenge

source1 is a reference catalog of businesses
sources2 and 3 contain noisy records that may refer to one of those businesses, or to none of them
the system must emit a set of target ids for every reference business, including an empty set when appropriate

the inputs contain ids, business names, addresses and countries
the labeled training links define ownership, while test links are withheld
the task is therefore open-world entity resolution with many plausible negatives, not a balanced pair-classification exercise

the organizer evaluates per-reference macro f0.5
false positives carry four times the denominator weight of false negatives in the equivalent count formula
correctly empty references matter to the macro average
the candidate file is also part of the deliverable and must describe the real pre-matcher candidate pool

## 2. eda process

the initial analysis used the original tsvs and their sha256 identities
[`eda.py`](../src/neural_v2/src/eda.py) and the stored census checked:

1. record counts and file identity
2. duplicate/malformed ids and id whitespace
3. country populations by source and split
4. blank and missing-like text values
5. unicode/script and url-like name patterns
6. name/address length distributions
7. true links per reference and source
8. references with no aliases and targets with no owner

later diagnostics joined these data properties to the actual retrieval, gate, ownership and rejection errors
that second step was essential: a common data pattern is not necessarily the largest source of model loss

the [compact census](../configs/data-census.json), [full eda](../src/neural_v2/reports/eda.json) and [label audit](../src/neural_v2/reports/labels.json) preserve the counts

## 3. dataset sizes and identity checks

| file | records |
| --- | ---: |
| train source1 | 2,206,821 |
| train source2 | 5,034,616 |
| train source3 | 5,285,603 |
| test source1 | 1,732,544 |
| test source2 | 4,887,273 |
| test source3 | 5,082,316 |

there are 10,320,219 training targets and 9,969,589 test targets
the census found no duplicate ids, malformed ids or id-whitespace errors in those six input files
names were nonblank in the recorded census, but many target addresses were blank or missing-like

stable row ids make the large joins and feature arrays efficient
they are an implementation index, not a replacement for the original external ids
hash checks prevent a changed row order from silently changing what `qid` and `tid` mean

## 4. country distribution and domain shift

| country | train references | test references | train targets | test targets |
| --- | ---: | ---: | ---: | ---: |
| us | 1,323,633 | 663,106 | 6,186,873 | 3,817,031 |
| india | 883,188 | 809,986 | 4,133,346 | 4,717,565 |
| france | 0 | 259,452 | 0 | 1,434,993 |

france is roughly 15% of the full test reference catalog and has no labeled training counterpart
the relative us/india population also changes, and the target-to-reference ratio is larger in test
these are observable distribution shifts; they do not reveal the public subset's country mixture or the unknown test ownership rate

### design consequences

- train/evaluate known-country behavior separately from unseen-country policy
- retain country as a retrieval/routing boundary and make any model use of it explicit
- inspect the actual inference route used for france instead of assuming a stronger global stack is active there
- treat predicted french score distributions and match counts as diagnostics, not labels
- evaluate transfer behavior in labeled-country proxies while recording the proxy's limitations

## 5. missing addresses

| target population | blank addresses |
| --- | ---: |
| train source2 india | 57,846 |
| train source3 india | 64,948 |
| train source2 us | 111,121 |
| train source3 us | 110,968 |
| test source2 india | 52,764 |
| test source3 india | 59,240 |
| test source2 us | 55,107 |
| test source3 us | 55,317 |
| test source2 france | 21,537 |
| test source3 france | 21,541 |

all reference addresses were nonblank in the recorded census
target missingness therefore changes the evidence available for matching even when the reference side is well described
the missing-like-token counters are separate diagnostics and are not assumed to be disjoint from every other text category

the early full-pool error audit made the importance of this slice measurable
blank-address targets were only about 4.4% of linked targets in its evaluated slice but contributed 72.9% of missing pre-matcher links and 83.8% of wrong top-1 choices

### changes that followed

- task-specific name/address retrieval rather than relying only on a generic embedding
- blank-address reverse retrieval with an explicitly recorded scope
- full-reference name multiplicity and token rarity features
- pair-model and graph evidence from other aliases
- separate missing-address indicators rather than replacing absent evidence with a confident mismatch

the later blank-address india retrieval comparison improved from 79.30% to 94.53% true-link recall at top 50
the later gate audit was still needed to check whether those recovered links survived filtering

## 6. multilingual and name-form variation

the reference names in the labeled us/india catalog are predominantly ascii in the census
many target names are not
india source2 has 562,437 non-ascii target names and source3 has 391,172
the corresponding us counts are 202,171 and 215,565

non-ascii is a character property, not a language classifier
it includes accents and punctuation as well as script changes
the more detailed eda also records devanagari and other indic-script occurrences

url-like names are another distinct form:

| training population | url-like names |
| --- | ---: |
| india source2 | 68,573 |
| india source3 | 77,136 |
| us source2 | 132,693 |
| us source3 | 133,786 |

### changes that followed

we retained original text and added comparison views for domains, initials, alternate-name markers, compact/sorted names and local transliteration
the task-trained retriever and mean-pooled pair models supplied learned cross-form evidence
name-change features described substitutions, additions and omissions without treating every normalization match as proof of identity

## 7. length and truncation

median reference-name lengths are 22 characters in the us and 27 in india
median reference-address lengths are 34 and 76 respectively
the french test reference medians are 19 for names and 48 for addresses

joint pair serialization contains both records plus field labels and country text
wordpiece length can therefore be much larger than one raw name's character length, especially with long addresses and script changes

we compared sequence limits of 96, 128 and 160 in the full cross-encoder matrix
length-bucketed training reduced padding waste
the selected configuration records the actual limit for every member instead of treating all family variants as interchangeable

## 8. truth cardinality and genuine negatives

| country | true linked targets | orphan targets | references with no true aliases |
| --- | ---: | ---: | ---: |
| us | 4,578,522 | 1,608,351 | 73,896 |
| india | 3,059,843 | 1,073,503 | 49,351 |
| total | 7,638,365 | 2,681,854 | 123,247 |

an orphan target is a record with no true source1 owner
a no-match reference is a source1 business with no true source2/source3 aliases
these are different objects and different sources of false-positive loss

the label distribution supports multiple aliases per business in both target sources
we enforce one owner per target, not one target per reference or one match per target source

hard-pair training intentionally enriches positives and difficult negatives
that training distribution is not the deployment prior, so calibration and final macro evaluation must still include realistic wrong-owner and orphan competition

## 9. why candidate recall was audited separately

the early post-gate candidate oracle was 0.995551 macro f0.5 on its evaluated fold
no pair classifier can recover a true link that is missing from that frozen candidate set

the original missing-link counter combined retrieval misses with gate pruning
we then introduced explicit forward/reverse and gate-selection comparisons so the source of loss was observable

the resulting changes were not simply wider lists everywhere
we trained better retrieval representations, added evidence to the gate, and used a score-adaptive budget
the selected gate policy improved candidate survival while averaging about 1.35 pairs per sampled target query

## 10. evaluation lessons

### 10.1 the scoring unit is a business set

a pairwise f-score is not the same as averaging f0.5 over reference businesses
references with no true aliases and references with many aliases have different error patterns but both contribute to the macro denominator
the early provisional pairwise cut was therefore revisited using the exact challenge scorer

### 10.2 missing truth changes the problem

if a true alias is absent from a sampled candidate table, it remains a false negative for its reference
using only in-candidate positives as the denominator can make a weaker retrieval system appear better
the true degree must come from the full label table

### 10.3 competing owners must remain visible

the winner for a target must be selected against all its candidate references before restricting the reported reference population
removing rivals first can turn an incorrect owner into an apparently correct one

### 10.4 context must match its claimed population

reference counts, maximum scores and sibling banks can reveal the construction of an owner-selected sample
in one diagnostic, removing owner-conditioned reference aggregate features reduced apparent precision substantially
we therefore distinguish raw/full-population features from features built after a projection

the inherited run-6 sibling ordering has a recorded counterexample
the newer fusion and collective stages construct their context over the complete candidate pool before selecting new-head fitting/check references

### 10.5 a used holdout stays used

the early one-time fold-1 audit was locked for that comparison
later development consulted it and later heads used additional partitions from previously exposed records
we retain stage-specific scope labels rather than rebranding every later comparison as a new blind audit

## 11. research that informed the design

| primary source | useful idea | boundary in this project |
| --- | --- | --- |
| [ditto](https://arxiv.org/abs/2004.00584) | jointly encoded record-pair matching and task-specific hard examples | sampled pair benchmarks do not establish this open-world business-set score |
| [filtering benchmark](https://arxiv.org/abs/2202.12521) | evaluate pair completeness separately from classifier quality | the oracle belongs to one frozen candidate set and evaluation population |
| [dial](https://arxiv.org/abs/2104.03986) | separate high-recall blocking from precision-oriented matching | hard near-duplicate negatives can help the matcher without necessarily helping blocker recall |
| [e5 model card](https://huggingface.co/intfloat/multilingual-e5-base) | documented multilingual serialization and embedding use | a public retrieval score is not a calibrated business-identity probability |
| [faiss index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) | bounded exact/approximate similarity search | approximate reverse recall must be measured under its actual index/scope |
| [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) | keep dependent examples together | owner grouping alone does not remove every name-family or historical-exposure risk |
| [probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | separate ranking quality from probability calibration | sparse cells and changed country distributions can invalidate transfer assumptions |
| [decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html) | tune a decision rule for the actual utility | generic pair thresholds are not the full source1 macro objective |
| [f-score threshold analysis](https://arxiv.org/abs/1402.1892) | threshold selection can overfit and depends on the metric/population | an f1-specific formula is not a direct solution to this macro f0.5 problem |
| [transclean](https://arxiv.org/abs/2506.04006) | cross-source inconsistency can identify questionable links | consistency is evidence, not permission for unconditional transitive closure |

these sources informed experiments and design choices
they supplied no external business identities, geocoding results or entity labels to the challenge pipeline

## 12. experiment and decision ledger

| question | test | outcome or lesson |
| --- | --- | --- |
| would a different tree algorithm fix the original gap | cpu tree/ensemble comparisons on the same restricted features | no tested variant displaced the high-precision gate; improve evidence first |
| was the provisional decision rule calibrated correctly | full-target macro replay and a selected logistic gate/neural stack | improved the one-time audit from 0.975874 to 0.978689 over the saved pool |
| were missing candidates a hard bottleneck | post-gate perfect-matcher oracle and error slices | yes for that pool; blank addresses dominated loss |
| would task-specific retrieval help | frozen versus trained small retriever against complete country reference banks | strong measured retrieval gains, especially blank-address india |
| did the old gate preserve the new retrieval gains | old-gate replay, then a compatible rich-gate holdout | gate and candidate-budget work were necessary, not optional |
| did extra evidence improve the matcher | raw-text/corpus stack versus v1 on the same development references | a material macro improvement before larger optuna search |
| did the learned model generalize to the public mixture | learned development comparison and first learned public release | strong labeled-country development did not establish the actual france route |
| could cpu-only search keep improving the learned stack | 64-trial and later 2,560-trial searches on saved scores | measured but modest later development gains; the broad finalist was not the final submitted global head |
| did every additional verifier help | learned edit-channel and four-view comparisons | no demonstrated eligible tuning win; retained as experiments |
| could richer collective context help | complete-pool graph features and a separately scoped new-head check | improved the conditional collective check from 0.991336 to 0.992280 |
| what moved the final public score above 0.99 | combined score inputs, country decisions and france blends | recorded progression 0.989926 to 0.990108 to 0.990284 |
| did the final three-pair edit improve accuracy | deterministic positional-rule replay | file behavior verified; independent score effect unrecorded |

## 13. what we kept

we kept complementary retrieval, exact/rare lexical evidence, task-trained embeddings, learned pair models, full-population score context, bounded collective features and calibrated ownership decisions
we also kept the evidence needed to distinguish those contributions: candidate ids, member scores, data/model hashes, feature schemas, fitting partitions, selected policies and actual output files

## 14. what we learned about limits

model capacity alone did not remove data/decision errors
candidate coverage, feature scope, the deployed country route, calibration and output identity each had an independent role

the final public score is not a proof that every hypothesis or rule is correct
france remains unlabeled, historical audit reuse remains part of the development record, and the exact three-pair edit has no separately recorded public gain
the detailed [results ledger](results.md) and [architecture](arch.md) connect these limits to the actual source
