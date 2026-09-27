# learned scoring

- 15 complete-epoch cross-encoders selected: 6 e5-large-instruct, 4 e5-base, 2 e5-small, 3 bge-reranker-v2-m3
- one interrupted bge checkpoint retained locally, excluded from this ensemble
- model membership and metadata hashes: [selection](learned-model-selection.json)
- gate floor 0.001, minimum 1, maximum 50; neural floor 0.001; [candidate audit](learned-candidate-selection.json)
- 16 country/range scoring jobs, 62 a100 devices; 200k-target work units, two workers per gpu, warmed shared reference indexes
- preserve each member's train/test scores and exact target coverage; cpu search follows complete training scores
- first live india/us/france batches pass finite-score, unique-pair and declared mean-logit aggregation checks: [smoke results](learned-score-smoke.json)
- cpu search starts from the prior selected optuna parameters, then compares 64 trials on the new cached features
- strict full-truth/full-competition evaluation, calibrated base and bounded france-rule exports, id-enabled validation
- retain checkpoints, raw/prepared scores, fit/eval matrices, trial models, journal, normalizers, reverse indexes and file hashes locally
- ordinary rules export now uses the supplied gate directory to resolve its normalizer; `src/run.py --check` covers this path

scoring and search are in progress; no new matching or leaderboard score is claimed here.
