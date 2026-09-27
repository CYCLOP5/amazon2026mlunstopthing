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
