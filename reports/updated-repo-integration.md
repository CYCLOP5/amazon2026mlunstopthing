# updated repo integration

source: `/home/cyclops/Downloads/Amazon-ML-Challenge-hmm`, updated sep 26. the reported public score is 0.987; the supplied experiment log records 0.984806 → 0.985292 for the large cross-encoder change and describes subsequent generator-aware and france-specific changes. an exact 0.987 artifact is not available.

## implemented

- `src/gfeat.py`: versioned `hybrid-v3` name-change features, initials/domain/alias consistency, full-reference word frequencies, common-word replacements and house-number evidence.
- `src/ceprep.py`: five fixed, three-million-pair hard-data samples. each includes roughly 2.08m positives. true positives are restored before capping; validation separates reference and target-owner groups.
- `src/ce.py`: mean-pooled multilingual cross-encoders, frozen input embeddings, binary cross-entropy, mixed precision, distributed training, length-grouped batches and saved model checkpoints. an ensemble retains individual model probabilities.
- `src/frule.py`: a separately identified france pattern profile. exact address-preserving initials/domain/alias cases require an unambiguous reference. explicit drops take precedence over additions. the profile is restricted to the configured unlabeled country and binds its normalization and implementation hashes.
- `src/fullprep.py` and `src/full.py`: shared immutable inputs, encoder-unseen gate-fitting owners, gate training, complete scoring, stack fitting, calibration and export.

the main cross-encoder comparisons retain global batch size 80 and learning rate 2e-5 from the kaggle recipes. the sweep also tests 1e-5/3e-5, lengths 96/128/160 and global batch 160. model families are multilingual e5 small/base/large-instruct and bge-reranker-v2-m3. large members are evaluated in the gate-uncertain band; skipped values explicitly fall back to the gate probability.

## verification and review

checks cover name-change features, owner-isolated hard-pair splits, cross-encoder backward/save/load, individual ensemble outputs, package membership, rule precedence, stack feature contracts, matching resume, run manifests and submission packaging.

review fixes include:

- hard drops cannot be undone by a later france addition
- score workers validate visible gpus and share an explicit cpu-thread budget
- distributed training uses an isolated rendezvous port
- staged compact base weights were compared with the pinned hub snapshot
- empty blank-address reverse banks retain a valid empty lookup
- the rule implementation and normalizer must match the packaged calibration

## remaining-gain assessment

[primary-source review and ranked experiments](next-gains-research.md).

the next decisions should use a compact, fully scoped gate/reranker comparison before expanded test scoring. prioritize candidate survival, missing-address errors, ambiguity-aware reranking, entity-level precision–recall frontiers and isolated cross-fitting. france rules are a separate domain-shift hypothesis, not a substitute for labeled controls. full-population context must be constructed before reference-incidence projection.

the trained compact retriever improved top-50 recall from 98.43% to 99.73% in india and 99.39% to 99.67% in the us on complete reference pools. blank-address india recall rose from 79.30% to 94.53%. those are retrieval measurements; final matching requires the new gate and cross-encoders, complete scoring, calibration and a separate development comparison.

## current training layout

the full plan uses 17 independent jobs: one gate-building job and 16 complementary cross-encoder configurations. the allocation uses standard nc96ads a100 v4 and nc24ads a100 v4 machines. inputs are staged once and shared by immutable datastore uri. training, scoring and postprocessing retain per-stage hashes and replayable outputs.

the previously validated optuna submission is the frozen fallback, with development macro f0.5 0.984546. no new public score is inferred from the new training runs.

## reusable cpu tuning

gpu training produces checkpoints. the subsequent full train/test pass produces the fixed candidate set and per-member probabilities. these outputs support repeated cpu-only stack, calibration and decoder experiments.

`full.py finish` now defaults to a 64-trial cpu search on the new cached scores. trial 0 reuses the previous optuna winner from `reports/optuna-search.json`: learning rate 0.0680600552, 127 leaves, unrestricted depth, minimum leaf count 393 and l2 0.00102611617. the prior selection's input hash is recorded; it is a starting configuration for new data, not evidence that it remains optimal. `--optuna-trials 0` refits that configuration directly with grouped early stopping.

the search carries the exact generator-feature and neural-member column contract into every trial and the selected model. partition 1 selects the model; partition 2 remains the separate development comparison. the regression check executes a two-trial search using the expanded feature schema and verifies that the selected prior configuration is trial 0.

final postprocessing retains `tuning-cache/` with the fitting and evaluation matrices, and `scores/` with base/stack train/test parquet files and their sidecars. a later cpu sweep can reuse the downloaded tuning cache:

```sh
.venv/bin/python src/opt.py run --root artifacts/full-result/tuning-cache --out artifacts/cpu-retune --trials 64 --workers 6 --threads 2
```

new hyperparameters require rescoring and recalibration with the supplied prepared data before exporting a new matching file. fixed candidates and cached neural probabilities require no new encoder pass. changing retrieval or neural weights requires regenerated gpu-dependent scores. improved leaderboard accuracy is never assumed from a lower pair loss or another search trial alone.

## candidate-budget audit

the completed gate holdout contains 10,000 reference owners, all 34,563 of their true aliases and sampled orphan targets. fixed top-3 filtering retains 0.990365 of true links, versus 0.992420 at top 5 and 0.995920 at top 20. the gate's sampled pair threshold at 0.017682 retains 0.995313 with 44,119 selected pairs from 44,563 queries. these are candidate and sampled-pair diagnostics, not full-population matching scores.

`gateprobe.py` audits fixed and score-adaptive budgets against the same owner-complete holdout. the adaptive policy keeps at least the highest-ranked candidate, adds candidates above the recorded gate floor and applies an explicit maximum. production selection is frozen before full scoring and recorded in the source configuration. `--gate-floor` is distinct from the later `--neural-floor`.

multi-node inference also uses explicit within-country shards, so the configured gpu workers process independent query ranges after a shared cache warmup. regression checks cover the adaptive cap/minimum, deterministic ties, cli forwarding, shard ownership and resumed manifests.
