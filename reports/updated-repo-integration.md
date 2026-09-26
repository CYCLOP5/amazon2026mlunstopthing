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
