# ml challenge 2026 business entity resolution

**team name:** Amazites
**team members:** Varun Jhaveri, Shivsharan Sanjawad, Raj Mathuria, Aastha Singh
**submission date:** 2026-09-25

## 1. executive summary

the selected system retrieves candidates with complementary lexical e5 and qwen3 views then applies a tree gate and a fine tuned e5 cross encoder

it is precision focused because macro per anchor f0.5 penalizes false positive links strongly. all work uses supplied records and eligible local pretrained models only

## 2. methodology

### 2.1 problem analysis

training contains 2206821 reference anchors and test contains 1732544. france is about 15 percent of test anchors but has no labeled training counterpart

names and addresses repeat often. aliases can be non-ascii and addresses can be blank. all audited training links remain within country but country labels are discovered from data rather than hardcoded

the score is macro per anchor f0.5 including empty sets and singletons. a pairwise threshold or sampled candidate score is therefore insufficient for final selection

### 2.2 solution strategy

**approach type:** hybrid retrieval tree gate and cross encoder
**core innovation:** independent unicode lexical and multilingual dense candidate views followed by bounded learned filtering and calibrated set decoding

the data stage preserves raw unicode name and address text. it creates a separate nfkc casefold punctuation-normalized offline transliterated view using `anyascii`. transliteration is comparison support not replacement text and no translation api is used

training folds are frozen with seed 42 at the reference entity level. fold 2 fits models fold 0 tunes decisions and fold 1 is held for locked audit. folds are stratified by country link degree unicode aliases and blank aliases

the selected retrievers are `intfloat/multilingual-e5-base` revision `d128750597153bb5987e10b1c3493a34e5a4502a` and `Qwen/Qwen3-Embedding-0.6B` revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. e5 uses its documented query and passage prefixes. qwen3 uses the identity retrieval instruction and left padding

sources

- `https://huggingface.co/intfloat/multilingual-e5-base`
- `https://huggingface.co/Qwen/Qwen3-Embedding-0.6B`
- `https://arxiv.org/abs/2004.00584`

## 3. candidate generation blocking

- **blocking keys used:** exact normalized name and address keys character name and address retrieval offline transliterated name retrieval and dense multilingual retrieval
- **candidate pairs generated:** pending full-pool run
- **how true matches were protected:** channels are unioned within country and source with lexical width 20 plus dense width 100 from each retriever. exact collisions are retained. no dense pair matrix is materialized

on the selected india query diagnostic lexical retrieval plus both dense encoders reached link recall `0.99538` at width 100 each. this is candidate coverage only not a final matching score

the final matcher input is the last post-gate candidate set. the selected configuration is `klex=20` `kdense=100` and upstream tree filter `k=3`. candidates dropped before neural scoring are counted by the run manifest rather than hidden

## 4. matching model

**features used:**

- name features: normalized and raw edit and token similarities transliteration agreement length and ambiguity features
- address features: character and token similarity digit agreement disagreement missingness and retrieval evidence
- other: country source channel ranks score gaps frequency and dense similarities

**model type:** baseline boosted tree gate plus e5 binary cross encoder
**threshold selection method:** choose the decoder and cutoff only from complete full-training-pool calibration using exact coverage and configuration fingerprints

the cross encoder starts from `intfloat/multilingual-e5-base` and uses field-labeled pair text. its binary classification head is trained with bce-with-logits loss at batch 128 for two epochs with maximum length 384

the trained neural model is initialized from e5. the e5 model-source parameter field is the conservative safetensors-element bound `278044162`. selected retrieval plus neural parameters total about 1.152b. all deployed source licenses are mit or apache 2.0 and the stack remains below 8b parameters. provided local checkpoints are used offline when present and can be regenerated only from the pinned artifacts

the selected inference blend is weighted-logit with neural weight `0.6`. the tree gate artifact is `models/gate` and the neural artifact is `models/neural`

cuda precision must be identical in calibration and test inference. `src/run.py` records this and rejects incompatible resumed output

## 5. results and error analysis

- **f0.5 score macro:** pending full-corpus calibration, test export, and locked audit
- **common false positives wrong merges:** repeated generic names shared addresses and close lexical competitors
- **common false negatives missed matches:** script or transliteration changes shortened aliases weak address text and blank addresses

the neural training stage used 502635 pairs including 34785 positives and completed in about 25 minutes including validation on the a100 stage. selected-query diagnostics reached `0.934` link recall at `0.995` precision with neural weight `0.6`. the tree top 3 filter reduced neural calls from about 1.5 million to 53481. a real local cli run completed 38 queries and scored 114 neural pairs. these are diagnostics not an official score and do not establish full-pool performance

the full training calibration test export and locked audit have not completed. no sampled threshold is reused for submission

## 6. conclusion

the workflow favors complementary retrieval and hard negatives over a larger model. final claims wait for complete training calibration full test coverage strict validation and the locked audit

## appendix

### a. code artefacts

`src/data.py` prepares provided tsv files and freezes folds. `src/block.py` creates lexical candidate runs. `src/train.py` fits the tree gate. `src/embed.py` writes dense validation candidates. `src/neural.py` prepares pairs trains the cross encoder and scores pairs

`src/run.py` is the final entry point. it launches one country run per worker validates exact coverage calibrates only complete training coverage and exports only complete test coverage

the required final outputs are `output/matching_results.tsv` and `output/candidate_pairs.tsv`. `src/package.py` requires both outputs `models/gate` `models/neural` and the final full-corpus `models/calibration.json`, as well as this README, this methodology document, the source lock, and model provenance. it rejects sampled or partial calibration

`src/package.py` accepts the committed uv files instead of a separate requirements file. it packages only calibration-selected retrievers with their sourced provenance; with `--hf-cache` it includes their safe pinned snapshots for offline inference

### b. compute and reproducibility

the verified remote worker was `Standard_NC24ads_A100_v4` with an 80 gb a100 and 24 cpu cores using `torch 2.8.0+cu128`. the local benchmark environment had 12 cpu threads 15.37 gib ram and a 6 gib gpu where e5 dense validation completed

workers and threads are bounded so country-parallel inference uses at most 24 cpu threads on the remote worker. batches must fit actual vram. no run should consume all 350 workspace cpu cores

the cloud runner enforces a 500 usd cap finite job duration zero minimum nodes one maximum node 120 second idle deletion and task-only ownership tags. it preserves pre-existing workspace resources

remote artifacts can be supplied as `azureml://datastores/<datastore>/paths/<project>/<run>/out/` inputs. `--no-download` records the output uri for chaining without local transfer. subscription resource group workspace and pricing are supplied at run time and are intentionally not recorded here

### c. additional results

| evidence | observed value | scope |
| --- | ---: | --- |
| india e5 qwen lexical retrieval recall | 0.99538 | selected query diagnostic at dense width 100 each |
| neural candidate validation pairs | 1508516 | candidate pair validation |
| neural validation throughput | about 6356 pairs per second | a100 final validation pass |
| full-corpus calibration, test outputs, and locked audit | pending | no optimistic completion claim |

the relevant primary sources are `https://huggingface.co/intfloat/multilingual-e5-base` `https://huggingface.co/Qwen/Qwen3-Embedding-0.6B` and `https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-optimize-cost?view=azureml-api-2`
