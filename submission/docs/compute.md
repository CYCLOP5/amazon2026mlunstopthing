# azure compute and execution design

the trained outputs and reconstruction checkpoints described here are distributed through [Kaggle version 1](https://www.kaggle.com/datasets/cycl0p5/amazites-ml-2026-reproduction-assets/versions/1)
the under-1024-mb Unstop zip includes the [automatic download and hash-check workflow](reproduce.md); the historical training architecture below is unchanged

## 1. why we split the workload

our pipeline mixes very different operations
parsing records and calculating grouped features are largely cpu/memory tasks; transformer training and dense pair scoring benefit from gpus; final calibration and output generation are cpu tasks again

we separated those stages so an expensive model pass produced reusable evidence
once train/test probabilities and candidate ids were frozen, a new tree fit, calibration curve, threshold or france blend could be evaluated without rerunning the encoders

the orchestration uses azure machine learning command/pipeline jobs
shared inputs are versioned by metadata and hashes, and outputs include model/configuration records as well as scores
the software route for full reconstruction is in [pipeline.md](pipeline.md)

## 2. machines and their roles

| machine | hardware class | actual role |
| --- | --- | --- |
| local workstation | 12 logical cpu threads, about 15 gib ram, rtx 2060 with 6 gib vram | eda, small retrieval experiments, compact-retriever training, checks, final replay and packaging |
| `Standard_NC24ads_A100_v4` | 24 vcpus and one a100 80 gb | single-gpu fitting/scoring work and the small cross-encoder variants |
| `Standard_NC96ads_A100_v4` | 96 vcpus and four a100 80 gb gpus | distributed cross-encoder fitting, gate/reverse-index work, multi-gpu pair scoring and bounded hybrid gpu work |
| `Standard_E64ds_v4` | 64 vcpus and large memory | full-population feature construction, graph/sibling heads, fusion and multi-process cpu search |
| `Standard_E16ds_v4` | 16 vcpus | final country-policy, blend and output passes |

these are recorded execution choices, not a requirement that exact hardware be used for byte-preserving score replay
the replay only needs its pinned cpu environment, supplied records and recorded score assets

## 3. shared assets before parallel work

the learned training/scoring jobs share:

- prepared source1/source2/source3 tables and data metadata
- the task-trained retrieval bundle
- fixed hard-pair samples and text tables
- the learned normalization mapping
- pinned base-model/tokenizer snapshots
- query populations reserved for gate fitting and holdout checks

the preparation code verifies these files before publishing the shared input location
workers receive the same immutable datastore uri instead of each submitting a separate copy of a large local directory

this avoided an observed parallel-upload race in which several sdk jobs attempted to create the same blob
the job's source snapshot and model/data fingerprint still identify the exact work requested

## 4. compact retriever fitting

the local retriever run was deliberately small enough for the available gpu
it used 96-token symmetric name/address serialization, batch 64, learning rate 0.00005 and gradient checkpointing
the saved recipe records bf16 execution and the actual warm-start and fitting populations

training began from a 64,000-pair checkpoint and continued on the requested 1.5-million-pair population
1,499,968 pairs were processed by the completed pass
the bundle binds the base revision, pair ids, fitting groups, tokenizer/model files and completion state

the complete-reference comparison then ran on a100 hardware
that distinction matters: the reported recall gains came from evaluating against the complete country reference pools rather than only in-batch training negatives

## 5. full cross-encoder training allocation

the main full-training envelope was **1,488 vcpus / 62 a100 gpus**, within the then-approved 1,500-core low-priority quota
this was a requested capacity envelope; scheduling, startup, retries and preemption were separate runtime concerns

the plan contained:

- one four-gpu gate-building job
- fourteen four-gpu cross-encoder configurations
- two single-gpu small cross-encoder configurations

the cross-encoder portion used 1,392 vcpus / 58 gpus
the separate four-gpu gate job brought the envelope to the recorded total
the 17 jobs were independent experiments/stages, not a single data-parallel fit spanning the entire allocation

### 5.1 training matrix

the following matrix records the requested cross-encoder comparisons
the selected bundle's own metadata remains authoritative for the final checkpoint and any retry

| run | family | seed | learning rate | token limit | batch per gpu | gpus | final use |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `large21` | e5-large-instruct | 21 | 0.00002 | 128 | 20 | 4 | selected |
| `large41` | e5-large-instruct | 41 | 0.00001 | 128 | 20 | 4 | selected |
| `large51` | e5-large-instruct | 51 | 0.00003 | 128 | 20 | 4 | selected |
| `large160` | e5-large-instruct | 71 | 0.00002 | 160 | 20 | 4 | selected |
| `large96` | e5-large-instruct | 131 | 0.00002 | 96 | 20 | 4 | selected |
| `large160b` | e5-large-instruct | 131 | 0.00002 | 128 | 40 | 4 | selected after retry |
| `bge51` | bge-reranker-v2-m3 | 51 | 0.00002 | 128 | 20 | 4 | selected |
| `bge21` | bge-reranker-v2-m3 | 21 | 0.00001 | 128 | 20 | 4 | interrupted checkpoint excluded |
| `bge131` | bge-reranker-v2-m3 | 131 | 0.00003 | 128 | 20 | 4 | selected |
| `bge160` | bge-reranker-v2-m3 | 41 | 0.00002 | 160 | 20 | 4 | selected |
| `base71` | e5-base | 71 | 0.00002 | 128 | 20 | 4 | selected |
| `base21` | e5-base | 21 | 0.00001 | 128 | 20 | 4 | selected |
| `base160` | e5-base | 41 | 0.00003 | 160 | 20 | 4 | selected |
| `base160b` | e5-base | 131 | 0.00002 | 128 | 40 | 4 | selected |
| `small21` | e5-small | 21 | 0.00002 | 128 | 80 | 1 | selected |
| `small71` | e5-small | 71 | 0.00003 | 128 | 80 | 1 | selected |

most comparisons hold global batch size at 80
the `160b` names denote the larger global-batch alternatives; they do not imply a 160-token limit
learning rate and sequence length were varied independently where recorded

### 5.2 process layout inside a gpu job

[`ce.py`](../src/neural_v2/src/ce.py) creates one distributed worker per visible gpu
workers use nccl, explicit device ownership and a shared distributed rendezvous
the training rows are length-bucketed, shuffled reproducibly and divided across ranks

each worker uses an explicit cpu thread count
tokenization, blas and other thread pools are constrained so that nested libraries do not each consume every cpu core
rank zero writes model snapshots; barriers keep snapshot and stop decisions coordinated

periodic model snapshots occur every 2,000 completed steps
the final metadata records completion, training configuration and hashes of the model/tokenizer files
the gate and full-scoring stages consume only the frozen selected members

## 6. gate fitting and reverse-index work

the gate job first builds reverse retrieval banks for the recorded split/country scopes
the first scope warms shared data, then independent scopes run across the configured gpus
blank-address reverse retrieval requests top 5 targets per reference

forward gate-training candidates use lexical top 10 and dense top 50
full-corpus name and word-frequency caches are constructed before gate fitting
the fitting owner groups are reserved from the retriever's fitting identities

the gate itself is a cpu lightgbm fit over the `hybrid-v3` feature contract
the gpu allocation is used for the retrieval/embedding work that supplies those features
the resulting gate bundle includes its normalizer and exact feature order

## 7. complete train/test scoring

full scoring was partitioned into 16 disjoint assignments
country and target-row intervals are explicit in the plan rather than inferred from worker launch order
this allowed small-country work and independent large-country shards to use the assigned gpus concurrently

inside a score job, [`full.py`](../src/neural_v2/src/full.py) coordinates the inference driver with:

- explicit gpu ids and a bounded worker count
- a 200,000-target shard size
- query batches of 2,048
- encoder batches of 512
- neural batches of 128
- cpu threads divided across score workers
- frozen retrieval, gate, ensemble and normalizer identities

these are the recorded orchestration defaults used by this route; effective neural batches also depend on the selected model path and available memory
the scoring configuration records the actual values used for each run

### 7.1 ownership and resume

every target belongs to one declared scoring assignment
manifest checks reject gaps, duplicate ownership and inconsistent source configuration hashes
resume logic reuses a finished part only when its configuration and id ownership agree with the current run

the completed learned score inventory covers 10,320,219 train targets and 9,969,589 test targets
the test receipt records 4,891 score parts across 53 run manifests and all fifteen member columns
the total target coverage is 20,289,808

the last india shards, not the already-completed small-country work, were the final bottleneck
we used completed-country outputs for schema/probability smoke checks while waiting for full coverage before fitting the final calibration

## 8. cpu fitting, calibration and search

the first larger optuna searches used an e64 node with eight processes and eight tree threads per process
the learned finish workflow starts with a fixed fitting/evaluation cache and queues the previous selected settings as trial zero

the retained learned fitting matrix is 322,009 by 101
the active evaluation matrix is 2,055,155 by 101
these matrices are subsets used for fitting/search, while the population-dependent statistics and candidate competition are established before that selection

the later 2,560-trial search distributed work across 208 cpu cores
its finalist slightly improved the reserved development comparison, but that global model was not the final submitted replacement
the final sprint instead reused the combined model scores and tested country decisions and france blends

parallel studies use optuna journal storage with file locking
sqlite storage had allowed duplicate claims of an enqueued trial during the earlier parallel run
per-trial models, metrics and the journal make the search reproducible and auditable

## 9. graph and bounded hybrid execution

the independent graph/sibling work used large-memory cpu nodes for score unions, grouped text features, rank/binary fits and sibling refinement
these operations materialize substantial full-population context and therefore do not belong in a small sampled-query feature loop

the bounded hybrid was a dependency-linked azure ml pipeline:

1. separate train and test lexical preparation jobs produced the bounded lexical lanes
2. a four-a100 step produced field embeddings, fused candidates, dense comparisons and missing pair scores
3. an e64 cpu step fitted weighted residual heads, selected the policy and validated exports

the recorded definition is [`final_hybrid_pipeline.yml`](../src/fusion/business_entity_resolution/azure/final_hybrid_pipeline.yml)
it pins the cpu image, the acpt pytorch/cuda environment for gpu work, shared-memory sizing, environment setup and stage inputs
the gpu step checks that four cuda devices are visible before starting

the final residual fusion and collective graph are cpu workflows over the prepared score union
their expensive text/context work is vectorized or chunked; the collective message representation has an explicit directed-row budget

## 10. environments

the project preserves stage-specific environments rather than claiming one environment produced every historical artifact

| environment | purpose |
| --- | --- |
| top-level `requirements.txt` | compact final replay |
| `requirements/tuning.txt` | country-cut and france-blend selection |
| `requirements/run6.txt` | run-6 lexical/stack runtime |
| `src/neural_v2/pyproject.toml` and `uv.lock` | learned retrieval, cross-encoder and rich-stack route |
| `src/fusion/pyproject.toml` and `uv.lock` | later cpu fusion and graph experiments |
| `src/fusion/upstream_neural/` environment records | earlier neural and hybrid gpu dependencies |

the learned route uses its recorded python 3.11 environment
the later fusion cpu job definitions use python 3.13
the bounded hybrid gpu definition uses the upstream-neural environment on python 3.12 with its pinned acpt image

code and artifact metadata record exact package/model identities
model-source names, revisions and licenses are listed in [models.md](models.md)

## 11. observed failures and engineering responses

| failure or bottleneck | response | why it mattered |
| --- | --- | --- |
| parallel jobs uploaded the same local asset directory | stage shared assets once and reuse their immutable uri | startup reliability and consistent inputs |
| the cross-encoder launcher referenced the wrong hard-pair filename | match the staged manifest and run a real-data cuda backward/save/load smoke check before restarting | a launcher can be syntactically valid while pointing to the wrong data |
| cuda device busy/unavailable during startup | isolate the failed assignment, inspect device ownership and retry against the frozen configuration | preserve completed work and complete target coverage |
| a low-priority bge run returned to the queue near completion | retain its periodic artifact, exclude it from the frozen scoring ensemble | every selected member needs a declared, reproducible state |
| late large-country shards dominated elapsed scoring time | explicit within-country ownership and reusable completed parts | capacity must be allocated to remaining work rather than only by country |
| row-id or source hashes could differ after moving caches | verify data and score hashes before rebasing the location | prevent silent joins between incompatible populations |
| full train/test replay exceeded local ram when run together | sequential local checks, projected columns and large-memory nodes for full fitting | local oom is a scheduling/data-shape issue, not a model-quality signal |
| rule metadata was not bound to the loaded normalizer | bind implementation and normalizer hashes in the recipe | the same threshold must mean the same transformation at export |

## 12. exact replay after training

the final replay reads only the columns and records needed for the fixed policy
it uses the recorded collective score table, france component projections and the run-3 swap pool
external ids are restored and reference groups are exported in bounded chunks

both release variants were checked against the exact uploaded matching and candidate hashes
the final-france check also began from the original raw challenge files
this route is independent of live azure access once its score assets and original records are available

see [reproduce.md](reproduce.md) for commands and [results.md](results.md) for what each measured result supports
