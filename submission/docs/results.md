# results and measurement scope

## 1. submitted variants

| artifact | public macro f0.5 | scope |
| --- | --- | --- |
| supplied collective-graph / mix3 baseline | 0.989926 | recorded team submission |
| sprint1 | 0.990108 | corrected team-reported value |
| sprint2 | 0.990284 | team-reported submitted result |
| final-france | not separately recorded | sprint2 with three positional-rule removals |

sprint2 is the best confirmed result among these sprint variants
the earlier shorthand for sprint1 was corrected to 99.0108%, or 0.990108

## 2. decision-layer development

the india/us cut comparison used reference groups from fold 0 while retaining full target competition
the selected country cuts changed development macro f0.5 from 0.9924478552 to 0.9925301624 on that comparison

the final france blend and cut were tuned on labeled india/us records
the selected collective-heavy blend scored development macro f0.5 0.9924115550, with pair precision 0.9988200771 and recall 0.9786969666
these are development measurements for selecting a decision rule, not labeled france accuracy

## 3. upstream evaluation

entity grouping prevents aliases of the same reference from being split across initial training folds
later fusion experiments use additional reference/name-group protocols over cached scores
their isolation is conditional on the already-developed upstream models and features
historically consulted folds are not described as a pristine end-to-end holdout

## 4. output checks

both releases retain the same complete candidate pool
the exact uploaded matching files are stored unchanged in their corresponding archives
checks cover source1 coverage, id validity, duplicate ownership, candidate membership and output fingerprints
the final-france file removes three existing pairs and preserves every source1 row
