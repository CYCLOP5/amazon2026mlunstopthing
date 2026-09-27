# historical learned-component training

the complete submitted pipeline, including run-6, graph, hybrid and collective fusion, is documented in [the full-pipeline guide](../submission/docs/pipeline.md)
this page retains the earlier learned-component training workflow

inference reproduction uses the frozen supplied checkpoints and matching calibration
retraining is a separate operation and requires a new calibration
byte-identical retraining across different hardware is not claimed

## 1. fixed data

```sh
uv run python src/data.py --data student_resource/dataset --out cache/data
```

keep `cache/data/meta.json`
for this initial component, fold 2 fits models and fold 0 develops decisions
fold 1 was initially reserved for an audit; later experiments consulted it and later fusion stages use their separately recorded protocols
do not move aliases of the same reference business between folds

## 2. native lexical candidate runs

repeat these two commands for `us` and `india` using distinct output paths

```sh
uv run python src/block.py \
  --data cache/data --cache cache/block --country india --fold 2 \
  --n 5000 --neg 1000 --k 20 --threads 8 \
  --out cache/runs/lex_train_india

uv run python src/block.py \
  --data cache/data --cache cache/block --country india --fold 0 \
  --n 2000 --neg 1000 --k 20 --threads 8 \
  --out cache/runs/lex_val_india
```

candidate scores must use corrected block score version 2
reference pools retain the intended full country search space
sampled query diagnostics do not establish final macro f0.5

## 3. native boosted-tree fit

```sh
uv run python src/train.py \
  --data cache/data \
  --train cache/runs/lex_train_us cache/runs/lex_train_india \
  --val cache/runs/lex_val_us cache/runs/lex_val_india \
  --out models/native-gate --model lgb --threads 8 --trees 800
```

this is the native feature/trainer path
the running original gate uses its own frozen lightgbm and catboost artifacts
do not substitute a newly trained one-model artifact while reusing that gate's calibration

## 4. supervised pair data

repeat for the us and india runs
fit data can add a known positive missed by retrieval; tuning data cannot

```sh
uv run --group neural python src/neural.py prepare \
  --data cache/data --run cache/runs/lex_train_india \
  --out cache/neural_pairs/train_india --hard 8 --random 2

uv run --group neural python src/neural.py prepare \
  --data cache/data --run cache/runs/lex_val_india \
  --out cache/neural_pairs/val_india
```

the recorded neural training run used 502,635 pairs including 34,785 positives
hard negatives come from retrieval rather than random unrelated names alone

## 5. neural training

```sh
uv run --group neural python src/neural.py train \
  --train cache/neural_pairs/train_us cache/neural_pairs/train_india \
  --val cache/neural_pairs/val_us cache/neural_pairs/val_india \
  --out models/neural \
  --model intfloat/multilingual-e5-base \
  --revision d128750597153bb5987e10b1c3493a34e5a4502a \
  --sources reports/model_sources.json \
  --batch 128 --epochs 2 --maxlen 384 --device cuda
```

retain weights tokenizer files and `neural_metadata.json`
inference uses the saved local checkpoint rather than downloading a replacement classifier
the checkpoint and numeric precision must match the calibration

## 6. team gate provenance

the upgraded gate is a separately fitted lightgbm booster with 54 ordered features and 248 trees
the controlled experiment used 1,858,304 fit pairs and 716,835 validation pairs
it removed the four sampled reference aggregates described in [arch](arch.md#removed-sampling-shortcut)

production files

```text
artifacts/upgraded-gate/
  lgb.txt
  metadata.json
```

the metadata stores the backend feature order score version fit population and hashes of our feature/prep/rule files
the booster retains its fitted trees and model parameters
experimental scripts and feature matrices are local artifacts rather than a public dataset distribution

`src/tfeat.py`, `src/tm_prep.py`, and `src/tm_rules.py` are the verified runtime port
their arrays and checkpoint predictions matched the earlier experiment exactly on the recorded parity sample
see [parity evidence](../reports/team_runtime_parity.json)

the native `src/train.py` fit command above does not recreate this different checkpoint
reuse the packaged verified booster for inference reproduction
any new fit must record its feature contract fit inputs and validation population before being used in a complete calibration

## 7. experiment interpretation

- e5-base already supports multilingual and cross-lingual retrieval
- large-instruct requires its own query instruction format and raw ref text
- adding large-instruct recovered five extra selected india queries at width 100
- the new gate improved paired high-precision recall with the same neural scores
- those components were then integrated and verified with a real-weight pipeline smoke test
- complete scoring of that combination is needed before claiming a better challenge score

the [evidence index](../reports/README.md) maps measurements to their exact scopes
the [runbook](ops.md) covers parallel inference calibration and submission production
