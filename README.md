# reproducible entity resolution workflow

this repository matches provided business records using only the supplied tsv files and eligible pretrained models

the selected stack is lexical retrieval plus `intfloat/multilingual-e5-base` and `Qwen/Qwen3-Embedding-0.6B` retrieval then a baseline tree gate and an e5 cross encoder

full corpus calibration, final test export, and locked audit are pending. no official score or completed submission is claimed here

## boundaries

- use only the provided data and mit or apache 2.0 pretrained assets at or below 8b parameters
- do not use business lookup geocoding translation or other external data or apis
- retain original unicode text and use the offline transliterated comparison view created by `anyascii`
- do not put raw business records entity ids credentials sas tokens api keys or host paths in public artifacts
- do not reuse a sampled threshold for final output. calibrate on the complete training target pool with the exact final configuration

selected sources

- `https://huggingface.co/intfloat/multilingual-e5-base`
- `https://huggingface.co/Qwen/Qwen3-Embedding-0.6B`
- `https://github.com/anyascii/anyascii`
- `https://arxiv.org/abs/2004.00584`

## setup

the committed `.python-version` pins python 3.11.16. `uv.lock` is the reproducibility lock and selects `torch 2.8.0+cu128` through the configured pytorch index

```sh
uv python install 3.11.16
uv sync --frozen
uv run python src/data.py --check
uv run python src/block.py --check
uv run python src/train.py --check
uv run --group neural python src/embed.py --check
uv run --group neural python src/neural.py check
uv run python src/run.py --check
uv run python src/package.py --check
```

these checks use temporary data. they do not download a corpus or make paid calls

run the pipeline commands below from the repository root. after extracting a final package, run them from `code/business_entity_resolution`; the same `src/...` paths work there, but the supplied raw dataset must be provided separately with `--data`

## prepare the supplied data

place the supplied files under `student_resource/dataset` in this repository, or pass the supplied dataset directory explicitly with `--data`. the final package does not include raw records

```sh
uv run python src/data.py \
  --data student_resource/dataset \
  --out cache/data
```

`src/data.py` reads explicit tsv fields with empty strings preserved. it stores raw unicode `nm` and `ad` plus normalized transliterated comparison fields `nn` and `an`. normalization is nfkc casefold punctuation-to-space whitespace collapse and offline `anyascii` only for the comparison view

the train split creates fixed seed 42 entity folds. fold 2 is fit data fold 0 is tuning data and fold 1 is the locked audit partition. strata include country link degree unicode aliases and blank aliases. keep the resulting `cache/data/meta.json` with every later artifact

## lexical candidates and tree gate

run each country and fold separately. these examples make sampled lexical train and tuning runs with bounded eight-thread retrieval

```sh
uv run python src/block.py \
  --data cache/data --cache cache/block --country us --fold 2 \
  --n 5000 --neg 1000 --k 20 --threads 8 \
  --out cache/runs/lex_train_us

uv run python src/block.py \
  --data cache/data --cache cache/block --country india --fold 2 \
  --n 5000 --neg 1000 --k 20 --threads 8 \
  --out cache/runs/lex_train_india

uv run python src/block.py \
  --data cache/data --cache cache/block --country us --fold 0 \
  --n 2000 --neg 1000 --k 20 --threads 8 \
  --out cache/runs/lex_val_us

uv run python src/block.py \
  --data cache/data --cache cache/block --country india --fold 0 \
  --n 2000 --neg 1000 --k 20 --threads 8 \
  --out cache/runs/lex_val_india
```

the lexical blocker keeps independent name address and exact-key candidates. it is not a full pairwise comparison

train the tree only after the selected train and validation runs share the same candidate score version and dense feature schema. `src/train.py` computes complete name and address features for every retained pair rather than trusting the channel that introduced it

```sh
uv run python src/train.py \
  --data cache/data \
  --train cache/runs/lex_train_us cache/runs/lex_train_india \
  --val cache/runs/lex_val_us cache/runs/lex_val_india \
  --out models/gate --model lgb --threads 8 --trees 800
```

`models/gate/metadata.json` records the feature contract score version input runs and sampled-query warning. sampled metrics are diagnostics not final calibration

## dense validation candidates

e5 uses its required `query:` and `passage:` prefixes. the validation command below writes `dense_candidates.parquet` beside the lexical run and records the pinned source revision

```sh
uv run --group neural python src/embed.py \
  --data cache/data --run cache/runs/lex_val_india --cache cache/embed \
  --model intfloat/multilingual-e5-base \
  --revision d128750597153bb5987e10b1c3493a34e5a4502a \
  --sources reports/model_sources.json \
  --device cuda --batch 64 --k 100 --maxlen 256
```

the observed india validation union of lexical e5 and qwen retrieval reached link recall `0.99538` at width 100 from each dense retriever. this is retrieval recall on a selected query diagnostic not a final score

## neural pair model

prepare fold 2 pairs with hard negatives. fold 2 can add a missed known positive before hard-negative selection. fold 0 validation keeps only its generated candidates and receives no gold-positive injection

```sh
uv run --group neural python src/neural.py prepare \
  --data cache/data --run cache/runs/lex_train_us \
  --out cache/neural_pairs/train_us --hard 8 --random 2

uv run --group neural python src/neural.py prepare \
  --data cache/data --run cache/runs/lex_train_india \
  --out cache/neural_pairs/train_india --hard 8 --random 2

uv run --group neural python src/neural.py prepare \
  --data cache/data --run cache/runs/lex_val_us \
  --out cache/neural_pairs/val_us

uv run --group neural python src/neural.py prepare \
  --data cache/data --run cache/runs/lex_val_india \
  --out cache/neural_pairs/val_india

uv run --group neural python src/neural.py train \
  --train cache/neural_pairs/train_us cache/neural_pairs/train_india \
  --val cache/neural_pairs/val_us cache/neural_pairs/val_india \
  --out models/neural \
  --model intfloat/multilingual-e5-base \
  --revision d128750597153bb5987e10b1c3493a34e5a4502a \
  --sources reports/model_sources.json \
  --batch 128 --epochs 2 --maxlen 384 --device cuda
```

the trained model is an e5 initialized cross encoder with a binary classification head and bce-with-logits loss. the e5 model-source parameter field is the conservative safetensors-element bound `278044162`. the measured training set had 502635 pairs including 34785 positives. two epochs took about 25 minutes including validation on the a100 stage

the final neural artifact is `models/neural` with `neural_metadata.json` tokenizer config and inference weights. use the provided local checkpoint for offline scoring when available. regenerate it only from the pinned data artifacts and command above. preserve its precision metadata

## selected final configuration

use this configuration unchanged for full training calibration and final test inference

| setting | selected value |
| --- | --- |
| retrievers | `e5 qwen3` |
| lexical candidates | `20` |
| dense candidates per retriever | `100` |
| upstream tree filter | `3` |
| neural blend weight | `0.6` |
| tree artifact | `models/gate` |
| neural artifact | `models/neural` |

the two dense retrievers plus the neural encoder total about 1.152b neural parameters. source revisions, licenses, and conservative parameter bounds are in `reports/model_sources.json`

`src/run.py` defaults to one worker, one thread, and query batches of 4096. run the full country-parallel train prediction first. on the verified a100 worker use three country workers with eight threads each. this occupies at most 24 cpu threads rather than all available workspace quota

```sh
uv run --group neural python src/run.py \
  --data cache/data --cache cache --gate models/gate --neural models/neural \
  --out artifacts/final_train --split train \
  --workers 3 --threads 8 \
  --k-lex 20 --k-dense 100 --k-gate 3 \
  --retrievers e5 qwen3 --device cuda \
  --encoder-batch 64 --neural-batch 128 --query-batch 4096 \
  --neural-weight 0.6 \
  --calibration-out models/calibration.json --audit
```

`src/run.py` refuses calibration unless every training country and target is covered. it writes country manifests source hashes coverage arrays timing and model configuration. use a lower encoder or neural batch if the selected device cannot hold the model and batches. do not change precision between calibration and test because the configuration fingerprint records it

after reviewing the complete calibration and required locked audit use the resulting calibration file for the complete test export

```sh
uv run --group neural python src/run.py \
  --data cache/data --cache cache --gate models/gate --neural models/neural \
  --out artifacts/final_test --split test \
  --workers 3 --threads 8 \
  --k-lex 20 --k-dense 100 --k-gate 3 \
  --retrievers e5 qwen3 --device cuda \
  --encoder-batch 64 --neural-batch 128 --query-batch 4096 \
  --neural-weight 0.6 \
  --calibration models/calibration.json --export-out output/final
```

the full-pool calibration and test outputs are pending. do not run the test command with a threshold copied from sampled validation

## compute notes

the verified remote training configuration was `Standard_NC24ads_A100_v4` with an 80 gb a100 and 24 cpu cores. it completed the cu128 path with `torch 2.8.0+cu128`

the local benchmark environment has 12 cpu threads 15.37 gib ram and a 6 gib gpu. e5 dense validation ran on that local gpu. do not assume the local encoder batch or precision transfers to the a100 or vice versa

keep workers times threads within available cores. use country sharding memory-mapped embedding arrays and finite batches. do not schedule all 350 workspace cpu cores

## cloud execution

`src/cloud.py` creates only task-tagged compute. it has a hard `500` usd ledger cap a finite job timeout zero minimum nodes one maximum node 120 second idle scale down and verified deletion. it does not delete existing workspace workloads

use current pricing and placeholders supplied by the account owner

```sh
uv run --group cloud python src/cloud.py run \
  --subscription '<subscription>' \
  --group '<resource-group>' \
  --workspace '<workspace>' \
  --profile gpu --hours '<hours-at-most-6>' --rate '<current-usd-per-hour>' \
  --fixed 10 --cap 500 \
  --input train=azureml://datastores/<datastore>/paths/<project>/<upstream-run>/out/ \
  --command 'uv run --frozen --group neural python src/neural.py train --train ${{inputs.train}}/train --val ${{inputs.train}}/val --out ${{outputs.out}}/models/neural --batch 128 --epochs 2 --maxlen 384 --device cuda' \
  --out artifacts/cloud/neural --no-download
```

the output uri is recorded in the job journal. pass that `azureml://datastores/...` uri as the next `--input` to chain jobs without a local download. use only task compute in the supplied workspace and never place subscription ids sas tokens or api keys in this file

azure references

- `https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series`
- `https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-optimize-cost?view=azureml-api-2`

## verify and package

verify the final outputs before packaging

```sh
python <challenge-resource-root>/utils/validate_submission.py \
  --matching output/final/matching_results.tsv \
  --candidate output/final/candidate_pairs.tsv \
  --test-dir <supplied-dataset>/test \
  --check-ids

uv run python src/package.py \
  --matching output/final/matching_results.tsv \
  --candidate output/final/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test \
  --repo-root . --code-root . \
  --readme README.md --methodology Documentation_template.md \
  --gate-model-dir models/gate --neural-model-dir models/neural \
  --calibration models/calibration.json \
  --output-zip output/submission.zip
```

`src/package.py` requires the final full-corpus `models/calibration.json`; it rejects sampled or partial calibration. it validates ids, candidate subset membership, model provenance, calibration completeness, and the selected source versions. it packages only the retrievers selected by that calibration and their sourced provenance. append `--hf-cache <safe-huggingface-cache>` to include their pinned retriever snapshots for offline inference

the validation helper belongs to the supplied challenge resources, not the final archive. the package command above runs from the source checkout. to repackage an extracted archive, run this from `code/business_entity_resolution`:

```sh
uv run python src/package.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir <supplied-dataset>/test \
  --repo-root . --code-root . \
  --readme README.md --methodology ../../Documentation_template.md \
  --gate-model-dir models/gate --neural-model-dir models/neural \
  --calibration models/calibration.json \
  --output-zip ../../repacked_submission.zip
```

## current evidence

- training anchors 2206821 and test anchors 1732544
- france is about 15 percent of test anchors and has no labeled training equivalent
- the metric is macro per anchor f0.5 with correct empty sets and singleton anchors included
- india e5 qwen lexical union recall is `0.99538` at dense width 100 each on the selected query diagnostic
- selected query neural diagnostics reached `0.934` link recall at `0.995` precision with weighted-logit neural weight `0.6`
- the tree top 3 filter reduced neural calls from about 1.5 million to 53481 in that selected query diagnostic
- a real local cli run completed 38 queries and scored 114 neural pairs; this is not a full-corpus result
- these precision recall and filter figures are not official or full-pool results
- full-pool calibration, test outputs, and locked audit: **pending**
