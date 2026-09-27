# historical learned-only submission

this page covers the earlier learned-only archive
the submitted combined pipeline and its final variants are documented in [submission](../submission/README.md) and the [q&a guide](qa.md)

amazites: varun jhaveri, shivsharan sanjawad, raj mathuria, aastha singh

**best recorded public leaderboard f0.5: 0.990284 for the later sprint2 release**
our team developed the learned pipeline and reproducibility workflow described here
the archived learned checkpoint's own scores and hashes remain documented in its [release receipt](../reports/submission-learned.json)

the archive contains the matching and actual pre-matcher candidate tsvs, source, dependency lock, trained retriever, 103-feature gate, 15 cross-encoders, selected lightgbm stack, calibration, reverse indexes and methodology
`package_manifest.json` records member files, model provenance, hashes and retriever locations
the later combined pipeline and its two release archives are documented in [submission](../submission/README.md)

## reproduce inference

commands below run from `code/business_entity_resolution` inside the extracted archive
use the original unmodified dataset directory containing `train/` and `test/`
the recorded gpu configuration uses a bfloat16-capable cuda device; an a100 80 gb supports the shown two-worker setup
cpu-only retuning uses saved gpu score files and feature caches from the experiment outputs

```sh
uv python install 3.11.16
uv sync --frozen --group neural
export DATA=/absolute/path/to/dataset
uv run --frozen python src/data.py --data "$DATA" --out cache/data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
uv run --frozen python src/run.py \
  --data cache/data --cache cache/inference \
  --gate models/gate --neural models/neural \
  --retrievers-file ../../package_manifest.json \
  --split test --out cache/test-scores \
  --device cuda --gpu-ids 0 --workers 2 --threads 10 --shard-size 200000 \
  --k-lex 10 --k-dense 50 --k-gate 50 \
  --gate-floor 0.001 --neural-floor 0.001 --neural-weight 1 \
  --encoder-batch 512 --neural-batch 128 --query-batch 2048 \
  --stack-model-dir models/stack --calibration models/calibration.json \
  --export-out ../../reproduced/output
uv run --frozen python src/validate.py validate \
  --matching ../../reproduced/output/matching_results.tsv \
  --candidate ../../reproduced/output/candidate_pairs.tsv --test-dir "$DATA/test"
```

the selected calibration carries the country/house transforms, decoder and optional bounded france rules
the same command uses the bundled selection without retuning
score directories support checked resume; changed model/data/feature/numeric fingerprints are rejected
all target ids must be covered exactly once before export

## training and cpu tuning

`src/retr.py` fits the compact retriever; `src/ceprep.py` creates grouped hard-pair datasets; `src/ce.py` fits the pair classifiers
`src/full.py` orchestrates the gate, parallel score generation and final cpu stage
`src/stack2.py` creates the rich fit matrix; `src/opt.py` runs the journal-backed parallel macro-f0.5 search
`src/post.py` calibrates score distributions and `src/decode.py` selects per-reference match sets

training targets retain full owner/reference competition during evaluation
fitting, calibration/search and stack-development reference partitions are disjoint
the prior tuned parameters are included as the new search's first trial; the selected trial and boosting rounds are in stack metadata
cross-encoder member probabilities are cached separately; ensemble combination uses mean logits

after downloading experiment scores, bind their dataset location to a local prepared copy
relocation verifies score and dataset hashes before atomically updating each metadata file

```sh
export SCORES=/absolute/path/to/downloaded/scores
uv run --frozen python src/post.py rebase --data cache/data --scores \
  "$SCORES/train.parquet" "$SCORES/test.parquet" \
  "$SCORES/stack-train.parquet" "$SCORES/stack-test.parquet"
```

`train.parquet` and `test.parquet` are the base scores for a new stack; `stack-*` files preserve the previously selected stack outputs

## checks

```sh
uv run --frozen python src/run.py --check
uv run --frozen python src/opt.py check
uv run --frozen python src/package.py --check
```

the methodology contains the measured validation results and candidate-count statistics for this archive
the final team result is recorded separately in [final-result.json](../reports/final-result.json)

## this checkpoint's measured result

the learned-only archive was reported at **0.986416** on the public leaderboard
its labeled-country development comparison used 73,752 reference businesses and scored 0.990353893 macro f0.5, versus 0.984546610 for the preceding tuned stack
the evaluation populations and the actual france route are different, so the development value is not a substitute for the public result

| learned-only artifact property | value |
| --- | ---: |
| source1 rows | 1,732,544 |
| accepted matches with its bounded france rules | 5,823,095 |
| actual candidates | 14,146,782 |
| mean candidates per reference | 8.1653 |
| p50 / p95 / p99 | 7 / 15 / 30 |
| maximum candidates for one reference | 298 |

the plain learned export had 5,834,027 matches
the bounded-rule version changed 29,986 reference rows while preserving india/us outputs and the candidate file
its isolated france-rule accuracy effect was not measured

## why this was an important component

the task-trained retriever materially improved complete-reference recall, especially for blank-address india
the dense/name-aware gate and adaptive 0.001 floor preserved more of that retrieved evidence
the cross-encoder ensemble then supplied separate member probabilities, allowing the 101-feature stack to learn complementary behavior

the completed full-scoring pass covered 10,320,219 train targets and 9,969,589 test targets
all fifteen member columns were retained
that made the results useful beyond this one archive: later cpu and fusion work could reuse the learned probabilities without another neural pass

## its role in the final combined release

the final combined pipeline uses this component as a score source, together with run-6, graph/sibling and bounded hybrid evidence
its `stack_prob` becomes the `newest` score in the full fusion union
gate, aggregate neural and individual member scores also contribute features

the final score union contains 20,177,322 candidates, so it is a different pool from this learned-only archive
the final matching policy uses collective probabilities for india/us and a multi-source france blend
the later sprint2 public result is 0.990284

## what the france result taught us

this early learned release used gate-based handling for the unseen country
it therefore did not use all the strong rich-stack/neural evidence for france in the same way it did for the labeled countries

the subsequent investigation separated three questions:

1. are the needed candidates present?
2. which score route does the deployed code actually use for that country?
3. does its calibration/decision rule transfer under the new distribution?

labeled-country transfer probes supported richer score routing with empirical calibration, but did not measure france accuracy
the final french policy was evaluated through recorded public submissions and retained as a separate frozen decision layer

## reusable artifact contract

the learned component's local inventory covers 20,843 files and 30,936,099,215 logical bytes
its checked reusable set includes model bundles, base and stacked train/test scores, the fitting/evaluation matrices, the 64-trial study/model outputs, normalizers and reverse indexes
that is a component-specific inventory, not a claim that every artifact from every later branch is contained in this archive

changing only a later cpu head can reuse compatible member-score tables
changing retrieval, tokenization, model weights, member order or candidate selection requires regenerated dependent evidence
hash-checked rebasing changes an artifact's data location without redefining its identity

## review links

- [complete final architecture](../submission/docs/arch.md)
- [actual azure training/scoring matrix](../submission/docs/compute.md)
- [full reconstruction](../submission/docs/pipeline.md)
- [results and submission lessons](../submission/docs/results.md)
- [final replay](../submission/docs/reproduce.md)
- [candidate selection record](../reports/learned-candidate-selection.json)
- [development comparison](../reports/learned-development-comparison.json)
- [model selection](../reports/learned-model-selection.json)
