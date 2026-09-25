# ambiguity audit

## root cause

- saved full-pool postgate oracle macro f0.5 is 0.995551, below .998. gate reachability is therefore already insufficient for that target within the saved postgate universe.
- retained truth is 25,565/33,646 (0.75982) for blank addresses versus 727,096/730,095 (0.99589) for nonblank addresses.
- blank targets produce 8,081/11,080 (72.93%) gate losses and 3,664/4,373 (83.79%) wrong top-1 outcomes. their within-stratum rates are 24.02% before and 14.33% after the gate; nonblank rates are 0.41% and 0.10%.
- this supports blank address as a major full-pool failure stratum. it does not establish that missing address alone explains the public-score gap.

## exact target ambiguities

| source | country | exact multi-owner groups | target records | owner assignments |
| ---: | --- | ---: | ---: | ---: |
| none | none | 0 | 0 | 0 |

equal input features force a deterministic classifier to return an equal result for those features. these target-only groups do not by themselves show equal full candidate-pair features, so they are not used as a model-theoretic upper bound.

## blank-name collisions

| source | country | fold0 linked blank targets | colliding reference name | rate |
| ---: | --- | ---: | ---: | ---: |
| 2 | india | 5,696 | 530 | 9.30% |
| 2 | us | 10,809 | 1,463 | 13.54% |
| 3 | india | 6,311 | 707 | 11.20% |
| 3 | us | 10,830 | 1,315 | 12.14% |

## reference name collisions

| split | country | reference records | records in colliding normalized names | rate |
| --- | --- | ---: | ---: | ---: |
| train | india | 883,188 | 392,116 | 44.40% |
| train | us | 1,323,633 | 474,025 | 35.81% |
| test | france | 259,452 | 89,267 | 34.41% |
| test | india | 809,986 | 354,418 | 43.76% |
| test | us | 663,106 | 193,133 | 29.13% |

## corpus shift

| split | source | country | targets | blank address | normalized-name collision | blank target ref-name collision |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 2 | india | 2,017,799 | 2.87% | 31.62% | 9.25% |
| train | 2 | us | 3,016,817 | 3.68% | 29.33% | 13.61% |
| train | 3 | india | 2,115,547 | 3.07% | 29.32% | 10.64% |
| train | 3 | us | 3,170,056 | 3.50% | 28.97% | 11.92% |
| test | 2 | france | 703,378 | 3.06% | 35.69% | 19.79% |
| test | 2 | india | 2,312,565 | 2.28% | 28.59% | 8.91% |
| test | 2 | us | 1,871,330 | 2.94% | 23.52% | 10.35% |
| test | 3 | france | 731,615 | 2.94% | 35.63% | 16.30% |
| test | 3 | india | 2,405,000 | 2.46% | 26.49% | 10.40% |
| test | 3 | us | 1,945,701 | 2.84% | 23.44% | 8.88% |

france is absent from train and present in test, so no france training prevalence or labeled error estimate exists here. all corpus-shift counts use every supplied source record; fold1 accuracy was not inspected.
