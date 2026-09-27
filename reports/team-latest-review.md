# team method review and evaluation findings

## scope

our team compared successive internal matching implementations on 2026-09-27
we reviewed the training recipes, calibration, france rules, composition features and validation logic against the competition's source1 macro f0.5 definition
the best recorded public result is [0.990284](final-result.json) for sprint2; the measurements below describe individual experiments

## methods we integrated

- task-trained multilingual retrieval and dense-aware candidate filtering
- cross-encoders trained on entity-isolated hard-pair datasets
- generator-aware name, acronym, domain and address-number features
- country/house calibration and explicit unseen-country score routing
- complete target-owner competition followed by expected-f0.5 set decoding
- cached per-model scores for cpu-only hyperparameter and calibration experiments

we evaluated bagging, learning-rate changes, richer pair features and additional cross-encoder families
the internal blend log reported cv 0.99121 to 0.99130 and composition cv 0.99130 to 0.99150
those restricted-candidate measurements were not treated as equivalent to a complete-population or public leaderboard result

## evaluation issues our team identified

### missing truth inflates the score

counting only true links that survived candidate generation removes false negatives from the denominator
one correct prediction with one missing true link scores 0.833333 under full truth, but incorrectly scores 1.0 when the missing truth is discarded
we retained complete truth degrees, including empty and zero-candidate references

### pruning competitors changes ownership

if target t scores 0.70 for evaluated reference a and 0.95 for another reference b, removing b changes the selected owner
we reproduced that behavior and retained every competing candidate for targets incident to evaluation references
population-dependent name, score and sibling context must be computed before any evaluation projection

### composition features need separate fitting and self-exclusion

a scored pair must not count itself as an already-established sibling
label-derived source2/source3 and missing-address priors must use fitting owners only
we kept these conditions separate from confidence features computed on the complete unlabelled population

### string rules need bounded ambiguity

our checks identified accented-acronym classification gaps and overly permissive single-letter initials
two plausible owners for one acronym must remain ambiguous
the production rule layer preserves hard-drop precedence, unique target ownership and a hashed normalization contract
generator flags remain evidence features rather than test labels

## france and calibration

france has no labelled training counterpart in the supplied data
we compared gate, neural and stack score routing using labelled-country transfer probes, then examined france's unlabelled score distributions
full-stack empirical calibration performed better in those probes than gate-only density transfer
the classifier had seen both labelled countries, so these probes do not measure france accuracy
our france-only calibration variant passed strict and official id validation and preserved india/us predictions and the actual candidate set

## reproducible evidence

- [metric and competitor counterexamples](team-latest-check.json)
- [full-population context issue](additional-workspace-findings.md)
- [integrated methods](team-integration.md)
- [country-transfer results](learned-country-transfer.json)
- [france-only output verification](learned-france-stack-variant.json)
- [2,560-trial finalist comparison](learned-r2-finalists.json)
- [candidate and validation contract](../Ml_Challenge.txt)

we measured matching quality and candidate footprint independently
accepted links must belong to the actual final pre-matcher candidate set; removing rejected candidates after matching is not a candidate-generation improvement
