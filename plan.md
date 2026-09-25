# amazon ml 2026 research and execution plan

research date 2026-09-25

## decision

build a precision focused entity matcher with complementary lexical and multilingual retrieval

start with character tfidf and offline transliteration plus a boosted tree matcher
add a multilingual retriever where it recovers missed links
fine tune a small multilingual pair model and use it where it improves held out macro f0.5
reserve a 4b model for hard cases only if the measured improvement justifies its cost

the strongest current evidence points to retrieval quality and hard negative training before model size
the final architecture will be selected from measured experiments rather than assumed from generic model leaderboards

azure budget is capped at 500 usd for this project
use uv and push code docs and compact metrics to the requested github repo at meaningful milestones
keep raw competition data large model files and generated candidate files out of normal git history

## rules that determine the solution

the supplied [statement](Ml_Challenge.txt) and [readme](student_resource/README.md) are the authoritative task sources
this is business entity resolution rather than the older amazon price prediction or image extraction tasks

- input fields are entity id business name business address and country
- s1 is the reference source and can match zero or many records in either other source
- multiple matches within s2 and within s3 are valid
- country is an open string label and france must be processed
- models must have mit or apache 2 licenses and at most 8b parameters
- external business lookup geocoding identity services and internet data augmentation are prohibited
- research papers software and eligible pretrained model weights are the relevant online resources
- use provided record text for inference rather than hosted translation or identity apis
- azure hosted training and self hosted eligible model inference do not require external identity lookup
- final results need one row for every test s1 id including empty predictions
- the candidate file must contain the actual final matcher input set and every predicted link
- rankings ultimately use the private leaderboard

the supplied validator defaults to skipping target id existence checks
it also treats missing candidate files and matches outside the candidate set as warnings
our final checks must enforce these conditions and run the supplied validator with `--check-ids`

## dataset census

all six source files were scanned completely with explicit tab parsing
all training labels were audited
source file hashes are in [eda](reports/eda.json) and the label hash is in [labels](reports/labels.json)

| split | source | us | india | france | total |
| --- | --- | ---: | ---: | ---: | ---: |
| train | s1 | 1323633 | 883188 | 0 | 2206821 |
| train | s2 | 3016817 | 2017799 | 0 | 5034616 |
| train | s3 | 3170056 | 2115547 | 0 | 5285603 |
| test | s1 | 663106 | 809986 | 259452 | 1732544 |
| test | s2 | 1871330 | 2312565 | 703378 | 4887273 |
| test | s3 | 1945701 | 2405000 | 731615 | 5082316 |

- 12527040 training records and 11702133 test records
- 10320219 training targets and 9969589 test targets
- 7638365 positive training links
- 2681854 training targets have no matching s1 label and supply genuine distractors
- 25.99 percent of training targets are unlinked
- 123247 training anchors are singletons or 5.5848 percent
- singleton rates are 5.5828 percent in us and 5.5878 percent in india
- mean true links per anchor is 3.4613
- observed link counts range from 0 to 11
- observed source counts reach 5 in s2 and 6 in s3
- these observed maxima are descriptive and must not become test time caps

| true links | share of training anchors percent |
| ---: | ---: |
| 0 | 5.585 |
| 1 | 5.399 |
| 2 | 17.002 |
| 3 | 24.055 |
| 4 | 21.937 |
| 5 | 14.589 |
| 6 | 7.471 |
| 7 | 2.899 |
| 8 | 0.846 |
| 9 | 0.191 |
| 10 | 0.024 |
| 11 | 0.002 |

### integrity and ambiguity

- no duplicate source ids malformed source rows wrong source prefixes or whitespace padded ids
- no missing or duplicate training anchor labels
- no invalid training targets or repeated ids within a truth list
- no target is assigned to multiple training anchors
- no id overlap between train and test for any source
- every one of the 7638365 positive links stays within country
- no duplicate normalized full s1 records in train
- one normalized full s1 collision in france test caused by spacing and case differences
- no normalized full s1 signature overlap between train and test
- these signature checks are not proof that every underlying business is entity disjoint
- preserve both test ids in the ambiguous pair and do not merge reference rows

names and addresses alone are heavily repeated
training has 685397 excess normalized name occurrences and 76648 excess address occurrences
test has 503920 and 58590 respectively
a training name appears for 253 different anchors
a france test name appears for 205 anchors and a france address for 101 anchors
shared names and shared facilities are major hard negative sources

### distribution shift

| anchor share percent | us | india | france |
| --- | ---: | ---: | ---: |
| train | 59.979 | 40.021 | 0 |
| test | 38.274 | 46.751 | 14.975 |

india has more weight in test and france has no training labels
the target pool grows from 4.677 records per anchor in train to 5.754 in test
this does not establish test match cardinality but it does increase retrieval competition
validation on a small easy target pool would be misleading

### missingness scripts and text lengths

all business names are populated
all s1 addresses are populated
s2 and s3 address missingness ranges from about 2.28 to 3.68 percent across split and country
the full per field counts and length quantiles are in `reports/eda.json`

- us and india s1 names are entirely ascii
- india s2 train names are 27.87 percent non ascii including 13.35 percent devanagari and 10.16 percent other indic scripts
- india s3 train names are 18.49 percent non ascii including 7.47 percent devanagari and 5.70 percent other indic scripts
- roughly 23 percent of india alias addresses contain non ascii text
- france s1 names are 15.72 percent non ascii and addresses are 28.27 percent non ascii
- france alias names are about 24 percent non ascii
- url shaped strings occur in roughly 3 to 4 percent of many alias name slices
- literal null like tokens occur inside some addresses and are not always whole field missing values
- p99 s1 name lengths are 41 chars for us 43 for india and 33 for france
- p99 s1 address lengths are 55 chars for us 134 for india and 78 for france
- these are character lengths rather than tokenizer lengths
- measure tokenizer truncation before selecting sequence limits

missing addresses are much more frequent among linked targets than unlinked targets
for example us s2 has 4.90 percent blank addresses among linked rows versus 0.32 percent among unlinked rows
this is a potential generator shortcut rather than sufficient matching evidence
hard negatives must include noisy records belonging to other anchors

## measured probes

`src/probe.py` samples 5000 anchors per training country using stable id hashes
the 10000 anchors have 34366 positive links
exact matching probes searched every training s2 and s3 row
normalization was nfkc casefold ascii punctuation replacement and whitespace collapse
these are descriptive untrained probes rather than final model validation results

| exact rule | us link precision | india link precision | us macro f0.5 | india macro f0.5 |
| --- | ---: | ---: | ---: | ---: |
| name | 0.0644 | 0.1427 | 0.3536 | 0.2606 |
| address | 0.8052 | 0.8321 | 0.2175 | 0.1957 |
| name and nonempty address | 1.0000 | 1.0000 | 0.0959 | 0.0700 |

the last rule had only 424 predicted links so it does not establish perfect precision on unseen data
its link recall was only 1.68 percent for us and 0.78 percent for india
it is a useful seed candidate channel rather than a complete solution

### positive pair difficulty

| slice | pairs | zero name token overlap percent | blank address either side percent | disjoint digit sets percent of all pairs |
| --- | ---: | ---: | ---: | ---: |
| us s2 | 8302 | 7.77 | 4.99 | 10.70 |
| us s3 | 8940 | 8.34 | 4.45 | 9.25 |
| india s2 | 8285 | 28.90 | 4.02 | 3.46 |
| india s3 | 8839 | 19.16 | 3.77 | 3.48 |

digit disagreement counts require both addresses to contain digits
they include formatting changes leading zeros number edits and address changes
do not turn numeric disagreement into an unconditional rejection rule

almost all sampled positive pairs retain some token overlap in at least one field
the difficult tail includes script changes combined with heavily shortened addresses
other cases include website only names abbreviations transliterated state names and edited house numbers
address and name retrieval must be separate complementary channels

### full pool rare token blocker

500 anchors were tested against all 10320219 training targets
the blocker used up to three rare name tokens and three rare address tokens with s1 document frequency at most 500
it then retained independent name address and mixed jaccard rankings per target source

| retained top k per ranking and source | final pairs | us link recall | india link recall | us oracle macro f0.5 | india oracle macro f0.5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 8617 | 0.7944 | 0.8463 | 0.8483 | 0.9031 |
| 10 | 17723 | 0.8037 | 0.8567 | 0.8505 | 0.9053 |
| 20 | 34527 | 0.8118 | 0.8624 | 0.8558 | 0.9072 |
| 50 | 78980 | 0.8142 | 0.8647 | 0.8561 | 0.9078 |

before top k pruning recall was only 0.8165 for us and 0.8681 for india
the main loss is initial retrieval rather than shortlist size
the oracle assumes perfect classification of retrieved pairs and is not an achieved model score
the small 250 anchor country samples make this a diagnostic rather than a precise generalization estimate

## online research findings

### matching architecture

[ditto](https://arxiv.org/abs/2004.00584) casts entity matching as pretrained sequence pair classification
its evidence supports a fine tuned cross encoder and field aware inputs
its published f1 results do not predict this challenge score

[sudowoodo](https://arxiv.org/abs/2207.04122) supports contrastive representations for blocking and matching
use the idea with supplied positive clusters and hard negatives
the [reference code](https://github.com/megagonlabs/sudowoodo) uses old dependencies and a bsd license so it is not our final model package

[beyond scale and generation](https://arxiv.org/abs/2607.24688) is a july 2026 preprint covering 1215 qwen3 fine tuning runs across nine datasets
it reports benefits from embedding specific initialization and joint pair encoding
larger models do not consistently win and can learn stronger shortcuts
generative models have some advantages under distribution shift
this supports testing a 4b model on hard out of domain cases rather than making it the default
the paper is under review and none of its benchmark scores are challenge guarantees

[calibration research](https://arxiv.org/html/2509.19557v2) finds temperature scaling useful for confidence quality but no consistent f1 improvement
calibration is needed for reliable decisions and does not by itself prove a score gain

[transclean](https://arxiv.org/html/2506.04006v1) shows how false positive edges can corrupt multi source groups and uses transitive consistency to remove them
use this as motivation for consistency features and conflict analysis
do not apply blind connected component closure

[carl em](https://arxiv.org/html/2609.01195v1) studies adaptive model spending but explicitly assumes at most one true match and small candidate lists
its selection controller is not directly suitable for this one to many task

### multilingual handling and translation

retain original unicode text for every record
use nfkc and casefold for a comparison view while keeping the raw field
preserve indic combining marks and retain a separate accent folded latin view

[anyascii](https://github.com/anyascii/anyascii) is a small offline isc licensed character transliterator
it has no contextual understanding and some characters can disappear
test its output as an additional retrieval view rather than replacing the original name
the model license constraint is separate from the licenses of supporting libraries

[indicxlit](https://github.com/AI4Bharat/IndicXlit) provides an 11m parameter transliteration model with an explicit mit license for code and weights
test indic to roman inference if simple transliteration leaves significant missed links
use the released model only and do not download its external training dataset or call hosted apis
retain multiple plausible spellings where helpful and measure collisions as well as recall

[indictrans2](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) is a local translation alternative
translation changes meaning while transliteration preserves spelling or sound
proper business names often need the latter
full corpus translation has no demonstrated benefit yet and would add substantial inference work

### eligible model shortlist

licenses were checked on first party model cards and model metadata
parameter sizes below are approximate unless an exact count is shown
pin checkpoint revisions before training and retain licenses with the submitted model

| role | checkpoint or library | size | license | initial use |
| --- | --- | ---: | --- | --- |
| pair features and classifier | [catboost](https://github.com/catboost/catboost) | data dependent trees | apache 2 | first supervised baseline |
| fuzzy features | [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) | no neural weights | mit | edit distance and token features |
| multilingual retrieval | [e5 base](https://huggingface.co/intfloat/multilingual-e5-base) | 278m | mit | efficient first dense retriever |
| multilingual retrieval | [qwen3 embedding 0.6b](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 595776512 | apache 2 | alternative retrieval quality |
| hybrid retrieval alternative | [bge m3](https://huggingface.co/BAAI/bge-m3) | about 0.57b | mit | compare incremental recall |
| trainable pair encoder | [xlm roberta base](https://huggingface.co/FacebookAI/xlm-roberta-base) | about 279m | mit | small supervised cross encoder |
| multilingual reranker | [bge reranker v2 m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 567755777 | apache 2 | encoder based pair model |
| multilingual reranker | [qwen3 reranker 0.6b](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 595776512 | apache 2 | alternative pair model |
| hard case reranker | [qwen3 reranker 4b](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | 4021784576 | apache 2 | only after measured gain |

e5 requires documented query and passage prefixes even for non english input
bge m3 does not require a query instruction
qwen retrieval and reranking are instruction aware
the default reranker prompt concerns web relevance so replace it with a business identity task instruction
raw reranker scores or their sigmoid are not automatically calibrated identity probabilities

until aggregate parameter rules are clarified keep the complete deployed neural stack below 8b
two nominal 4b rerankers would already total more than 8b using the exact counts above

## implementation design

### data and normalization

- preserve original ids and use dense internal row indices only for storage
- read explicit tsv columns with empty strings preserved
- use country equality for the primary partition since all audited training links preserve country
- discover country labels from the data and never filter to a fixed country list
- support an unpartitioned fallback for missing or unknown country labels
- retain raw unicode normalized unicode and optional transliterated views
- expose legal suffix reduced and token reordered views without deleting the raw name
- parse embedded urls and dba fragments from provided text without visiting the urls
- retain house unit floor and postal digit tokens with separate leading zero normalized features
- derive abbreviation and alias rules from training data where possible
- avoid external address databases and geocoders
- deduplicate identical serialized model inputs for compute caching while preserving every record id

### candidate generation

union independent channels within each country and target source

1. exact normalized name and address keys
2. character tfidf name similarity with short character ngrams
3. character and token address similarity
4. transliterated name similarity
5. rare token and numeric compound keys
6. multilingual dense nearest neighbors over a field labeled record

start with modest per channel top k and measure marginal recovered truth
preserve source specific candidate coverage and add reverse retrieval candidates where useful
increase budget for weak fields and ambiguous common names rather than applying one tight cutoff

use [sparse dot topn](https://github.com/ing-bank/sparse_dot_topn) for bounded sparse products or an equivalent bounded inverted index
never materialize the full dense pair matrix
use [faiss](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) for dense retrieval
measure ann recall against exact search on sampled queries before tuning memory compression

initial target is at least 99.5 percent link recall with a preference for 99.9 percent if affordable
also track per anchor oracle macro f0.5 and the fraction of anchors with all true links retrieved
require slice reports for country source script changes common names and blank addresses
these are targets and have not yet been achieved

the exported candidate file is the post retrieval set scored by the final matcher
if a tree scores all candidates and a neural model scores only an uncertain subset export all tree scored candidates
record routing and scores so the composite matcher can be audited

### supervised matcher

first model is a boosted tree over name address retrieval and ambiguity features

- raw and normalized edit similarity jaro winkler token overlap and tfidf similarity
- separate field agreement rather than one concatenated fuzzy score
- directional token containment and unmatched token counts
- number agreement disagreement and missingness without absolute numeric vetoes
- script compatibility transliteration agreement and lengths
- name and address frequency to distinguish rare identity evidence from generic words
- per channel ranks score gaps reciprocal target ranks and source indicators
- dense similarity where available

rapidfuzz token set similarity can score a strict subset as a perfect match
retain length and unmatched token features rather than trusting that score alone

train on true positives plus retrieved hard negatives
include same name different address same address different name and close lexical or semantic competitors
include noisy aliases of other anchors and genuine unlinked targets
random negatives alone are too easy
mask all aliases of the same true entity when doing contrastive negative sampling

start with an anchor stratified subset and build a learning curve before training on all labels
test anchor balanced weighting because the evaluation averages anchors rather than links
calibrate and select thresholds on the actual candidate distribution rather than the sampled training class ratio

fine tune the best small pair encoder after the tree baseline
use field labeled inputs and train only on supplied training labels and identity preserving transformations
prefer number preserving augmentations and do not label every edited number as a negative
mine a second hard negative round if error analysis supports it

retain frozen multilingual features when fine tuning damages out of country performance
blend only complementary models using held out or out of fold predictions

### macro f0.5 decoding

for nonempty truth the per anchor score is

```text
f = 1.25 * tp / (pred_count + 0.25 * true_count)
  = 1.25 * tp / (1.25 * tp + fp + 0.25 * fn)
```

empty truth and empty prediction score 1
empty truth and any prediction score 0
nonempty truth and empty prediction score 0

the count formula gives fp a coefficient four times that of fn
the statement describes a two times precision emphasis but the written formula is what must be implemented
this does not imply that one universal probability threshold such as 0.8 is optimal

begin with a global threshold optimized directly for macro f0.5
then test a calibrated empty set gate and per anchor prefix selection
expected set utility approximations must account for their independence assumptions and missing candidates
compare them with the simple threshold before retaining them
do not use pair accuracy micro f0.5 or average binary classification f0.5 as the competition score

test reciprocal target conflict handling because no training target belongs to two anchors
never enforce at most one match per source for an anchor
do not automatically propagate uncertain edges through connected components

## validation

1. persist an entity grouped 80 10 10 train tune and locked audit split
2. keep all known aliases of each anchor together
3. group suspicious duplicate reference signatures together where present
4. stratify by country link count script and address quality
5. restrict supervised training pairs and mined negatives to the training partition
6. evaluate held out anchors against the full target pool to preserve distractor density
7. fit learned transforms on the training partition and freeze them for validation
8. retain all singleton anchors and compute the exact macro metric
9. tune on the tune partition and use the locked audit partition only for promotion decisions
10. use paired anchor level uncertainty estimates for small score differences

run us to india and india to us transfer experiments
hold out frequent name and address families in additional stress tests
report known country results separately and use the observed test mix only for a clearly labeled known country summary
there is no genuine labeled france validation score
do not manufacture one from pseudo labels

default to no test pseudo labeling or test fine tuning until competition policy on transductive training is clarified
reading provided test records for indexing and distribution analysis is required for inference

## experiment order and promotion gates

| stage | work | promotion evidence |
| --- | --- | --- |
| e0 | full audit and untrained probes | complete and reproducible |
| e1 | unicode lexical and transliteration blocking | full pool recall and memory timing |
| e2 | hard negative tree matcher | valid end to end macro score and error slices |
| e3 | e5 and qwen or bge retrieval comparison | extra recovered true links per added candidate and gpu hour |
| e4 | small supervised pair model | better tune and audit macro score with feasible full inference |
| e5 | singleton decoding and complementary blend | stable improvement across countries and stress tests |
| e6 | 4b hard case model or stronger transliteration | gain on actual difficult cases that survives locked audit |
| e7 | final refit inference and package | complete outputs strict validation reproducibility and cost audit |

log dataset hashes split seed model revision package lock retrieval settings macro score precision recall singleton false merge rate candidate count wall time peak memory and estimated usd
keep false positives and missed candidates as separate error categories
public leaderboard movement can support a hypothesis but must not replace local validation

## compute and budget

local hardware observed is 12 cpu threads 15.37 gib ram and an rtx 2060 with 6 gib gpu memory
the original python is 3.14 and has no ml packages installed
create an isolated uv python environment with tested package versions rather than modifying system python

full test all pairs would require 17272751604416 comparisons
country partitioning still leaves 6724569566212 comparisons

| mean final candidates per anchor | test pairs | 64 float32 features alone gib |
| ---: | ---: | ---: |
| 60 | 103952640 | 24.78 |
| 100 | 173254400 | 41.31 |
| 200 | 346508800 | 82.61 |

100 candidates per anchor would take 48.13 hours at an assumed 1000 pair scores per second
this is arithmetic rather than a measured model throughput claim
use cheap classification and selective neural work with an audited routing rule

test target embeddings alone require 14.26 gib at 768 dims fp16 or 19.02 gib at 1024 dims fp16
float32 storage doubles those figures and ann structures add overhead
use country shards memory mapped arrays bounded batches and compact internal indices

### azure choice

[t4 v3](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncast4v3-series) supplies 16 gb t4 gpus for smaller pilots
[nc a100 v4](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series) supplies an 80 gb a100 and 220 gib ram in `Standard_NC24ads_A100_v4`
the a100 option is well suited to large retrieval indexes and short sequence training if price quota and capacity permit
use a memory rich cpu vm when the workload is sparse retrieval or tree training rather than paying for an idle gpu

[flash attention support](https://github.com/Dao-AILab/flash-attention) differs across gpu generations
use compatible fp16 attention on the local 2060 and t4
do not assume the a100 bf16 and flash attention path works unchanged on turing

[azure quotas](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2) are regional and quota does not guarantee capacity
the existing account is enabled and its current resource group is in eastus
we have not provisioned or charged cloud compute during this research phase

### hard spending controls

- total authorized azure spend is 500 usd
- target experimental allocation is 400 usd with 75 usd reserved for final inference and 25 usd contingency
- price every selected resource using current regional pricing before creation
- isolate project resources in a new tagged resource group
- track conservative accrued compute disk storage and transfer estimates in a durable ledger
- use maximum job runtimes and resource cleanup in addition to billing alerts
- billing data can lag so do not use delayed cost reports as the only limit
- stop experiments early when projected final inference would consume the remaining reserve
- deallocate idle compute and delete unused project disks and networking resources
- checkpoint and export artifacts before deleting the project resource group
- never delete unrelated preexisting resources

[azure cost guidance](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-optimize-cost?view=azureml-api-2) recommends zero minimum cluster nodes and job termination limits
it states that legacy low priority aml allocations migrated to spot behavior after 2026-03-31
spot prices vary and preemption requires resumable jobs
prefer spot only when checkpointing is verified and the deadline can tolerate interruptions

## reproducibility and final package

the current analysis can be rerun from the project root

```sh
python3 src/eda.py --check
python3 src/probe.py --check
python3 src/eda.py
python3 src/eda.py --labels --out reports/labels.json
python3 src/probe.py
```

the observed full audit took about 5 minutes the country label audit 55 seconds and the probe 2 minutes 35 seconds on this machine
raw probe examples remain local in `reports/probe.json`
the plan records the aggregate findings

the implemented pipeline will use a committed uv lock exact model revisions fixed splits and resumable stage artifacts
record the exact final commands and measured scores after they exist
produce `output/matching_results.tsv` and `output/candidate_pairs.tsv`
assert all required anchors valid target ids no duplicates and match subset membership

run the supplied validator from `student_resource` with the correct output paths

```sh
python3 utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

package both outputs the self contained pipeline with pinned dependencies run instructions and the filled methodology template
large candidate files may need a memory rich machine for the supplied validator
also implement bounded memory format and subset checks
verify cloud cleanup and the spend ledger before declaring the goal complete

## remaining external facts

- challenge submission deadline and daily submission limit
- organizer interpretation of the aggregate 8b cap and test transductive training
- current azure family quotas regional availability and live hourly prices
- whether any final zip or artifact size limit applies

these facts must not be guessed
the implementation can progress with the conservative model and data policy above
