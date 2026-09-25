# amazon ml 2026 research and execution plan

research date 2026-09-25

## decision

build a precision focused entity matcher with complementary lexical and multilingual retrieval

start with character tfidf and offline transliteration plus a boosted tree matcher
add a multilingual retriever where it recovers missed links
fine tune a small multilingual pair model and use it where it improves held out macro f0.5
evaluate a 4b model for hard cases only if its measured quality gain justifies the added runtime

the strongest current evidence points to retrieval quality and hard negative training before model size
the final architecture will be selected from measured experiments rather than assumed from generic model leaderboards

azure ml provides the gpu training and parallel inference platform
the deadline is sunday 2026-09-27 at 08:00 ist or 02:30 utc
zero of three total submissions had been used at the last user confirmation
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
- final ranking also reviews candidate generation code and candidate count per s1 alongside leaderboard matching quality

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

[carl em](https://arxiv.org/html/2609.01195v1) studies adaptive model routing but explicitly assumes at most one true match and small candidate lists
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
increase candidate width for weak fields and ambiguous common names rather than applying one tight cutoff

the first implementation will also test reverse retrieval as the primary direction
each target has at most one labeled reference and the reference index is roughly five times smaller
retrieve several reference candidates for each target then regroup pairs by reference for the required outputs
retain all exact name and address collisions as candidates instead of dropping tied identities at a small top k
compare recall and runtime with forward retrieval before choosing the final union

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
| e7 | final refit inference and package | complete outputs strict validation and reproducibility |

log dataset hashes split seed model revision package lock retrieval settings macro score precision recall singleton false merge rate candidate count wall time and peak memory
keep false positives and missed candidates as separate error categories
public leaderboard movement can support a hypothesis but must not replace local validation

## compute and runtime

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
the a100 option is well suited to large retrieval indexes and short sequence training
use a memory rich cpu vm for sparse retrieval tree training calibration and output generation

[flash attention support](https://github.com/Dao-AILab/flash-attention) differs across gpu generations
use compatible fp16 attention on the local 2060 and t4
do not assume the a100 bf16 and flash attention path works unchanged on turing

[azure quotas](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2) are regional and quota does not guarantee capacity
the existing account is enabled and its current resource group is in eastus
resumable work units bind data model and numerical fingerprints so completed predictions remain reusable across compatible machines

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
verify the selected model provenance exact target coverage and output checksums before delivery

## remaining external facts

- challenge submission deadline and daily submission limit
- organizer interpretation of the aggregate 8b cap and test transductive training
- current azure family quotas and regional availability
- whether any final zip or artifact size limit applies

these facts must not be guessed
the implementation can progress with the conservative model and data policy above

## execution updates

- uv now uses python 3.11.16 with a committed lock and a verified fp16 cuda operation on the local gpu
- research commit `2366c31` is pushed to main
- strict validation utilities are included on the implementation branch with self checks passing
- implementation is tracked in [draft pr 1](https://github.com/CYCLOP5/amazon2026mlunstopthing/pull/1)
- the data preparation stage preserves original text creates ascii comparison views and stratifies whole entities by country degree script and address quality
- full preparation completed in about 2 minutes 35 seconds and verified every supplied positive link
- azure compute network storage and quota providers are registered
- eastus has 65 dedicated vcpus available but initially zero dedicated gpu family quota and only 3 spot vcpus
- the a100 quota request returned `ContactSupport` and the t4 request returned `QuotaNotAvailableForResource`
- the spot quota request was throttled with an explicit one hour retry delay
- gpu quota failure is a compute constraint while cpu modeling and local gpu experiments continue

### measured implementation results

the reverse character retrieval probe used 2000 held out anchors per country and searched every reference in that country

| country | true links | retrieved links | link recall | oracle macro f0.5 | mean candidates per target |
| --- | ---: | ---: | ---: | ---: | ---: |
| us | 6902 | 6816 | 0.98754 | 0.99623 | 41.00 |
| india | 6925 | 6524 | 0.94209 | 0.97822 | 39.43 |

results are saved in `reports/lex_val_us.json` and `reports/lex_val_india.json`
india still has a substantial script and spelling gap
this is why the next measured experiment uses a multilingual encoder

the first tree used 5000 training anchors per country with hard negative candidates
its sampled query macro estimate was 0.9860 but the threshold selected on that incomplete population had only 0.5798 pair precision
wrong links into unsampled anchors were not charged to that sampled macro score
that threshold is not suitable for submission
`src/infer.py` therefore requires complete target coverage before producing a full pool calibration
it tunes on fold 0 and opens fold 1 only for an explicitly requested audit

at a provisional threshold of 0.95 the original tree had 0.99577 pair precision and 0.79967 recall over all true links in the sampled queries
these are diagnostics rather than a competition score
the model still needs a better precision recall tradeoff

candidate score version 2 computes both field similarities for every candidate
previously a candidate introduced by one channel could have a zero score recorded for the other channel despite real overlap
on the india validation candidates this restored 20146 nonzero name scores and 32536 address scores without changing any candidate or label
model metadata and inference now require matching score versions

### cloud execution status

the user supplied the existing `mlworkloads` azure ml workspace in eastus
its cluster quotas are independent of ordinary virtual machine quotas
the verified dedicated cpu quotas are 350 edsv4 vcpus and 100 esv3 vcpus
azure ml approved 24 total low priority vcpus after a separate request
this permits requesting one 24 vcpu a100 spot node subject to capacity

the execution layer resolves model metadata relative to the uploaded source package
regression checks verify portable model identity and explicit artifact locations
training and scoring use persistent checkpoints and finite runtime settings

### multilingual retrieval evidence

the frozen e5 base encoder was tested on the same 2000 india anchors against all 883188 india references
it ran successfully on the local 6 gib gpu

| dense width | dense link recall | dense and lexical union recall |
| ---: | ---: | ---: |
| 1 | 0.90296 | 0.96491 |
| 5 | 0.94267 | 0.97545 |
| 10 | 0.95437 | 0.97978 |
| 20 | 0.96217 | 0.98383 |
| 50 | 0.97227 | 0.98816 |
| 100 | 0.97718 | 0.99004 |
| 200 | 0.98209 | 0.99264 |

the curve uses returned fp16 neighbor order and tied scores can change boundary membership
the top 50 and top 200 reports are saved under `reports/e5_india_k50.json` and `reports/e5_india_k200.json`
search over the cached reference vectors took about 3 seconds for the top 50 probe
encoder construction and corpus encoding are measured separately

there are 51 links still missing from the top 200 union
23 are s2 links and 28 are s3 links
21 have non ascii alias names and 8 have blank addresses
blank address residual risk is 3.125 percent versus 0.645 percent for populated addresses
eight residuals have both normalized name and address string likeness below 50 percent
two blank address residuals have ambiguous normalized reference names
no conflicting owners were found in the checked identical raw record groups

going from top 100 to top 200 adds 892500 pairs for 18 more links or roughly 49600 additional pairs per recovered link
the next experiment is a complementary frozen retrieval model followed by supervised hard negative fine tuning
large uniform candidate lists and speculative phonetic rules are lower priority

### execution reliability

full inference now filters parquet data before batching
a real 7974 target pilot retained identical 312428 scored pairs while runtime fell from about 112 to 77 seconds
index failures now raise errors rather than silently creating empty candidate sets
when a validation country has no fit partition rows its lexical transform uses other fit partition countries only

the azure ml sdk returned sas credentials where its artifact helpers expected account keys
this caused log signature and incorrect padding failures
the runner now waits on normalized job status enum values and downloads from an explicit task output uri using the actual credential type
this path passed a real 12 file download check without modifying the workspace credentials

### first supervised multilingual matcher

the a100 80 gib spot node completed training with persistent checkpoints
502635 pair examples contain 34785 positives and supplied data hard negatives
the encoder is initialized from the pinned mit licensed multilingual e5 base model
its new binary classification head is trained with bce rather than a regression objective
the trained checkpoint has 278044417 parameters

two epochs completed in about 1516 seconds including validation
complete candidate pair validation contains 1508516 pairs
the last validation pass processed about 6356 pairs per second on the a100
these are stage timings rather than end to end inference guarantees

with at most one predicted owner per target the selected query diagnostic reaches about 0.916 link recall at 0.995 precision
at a fixed 0.98 probability cutoff it has 12522 true links and 49 false links
false links into unsampled references are counted
this remains a selected query diagnostic and final thresholds still require complete target pool calibration

directory markers in mounted azure output use `hdi_isfolder` metadata
the downloader now ignores those markers instead of treating a directory as an empty file
the completed training checkpoint was recovered and verified

### complementary retrieval and candidate filtering

qwen3 embedding 0.6b is weaker than e5 alone on india but recovers different links
lexical retrieval plus both encoders reaches 0.99292 link recall at dense width 50 each
width 100 each reaches 0.99538 and width 200 each reaches 0.99639
the model licenses revisions and measured parameter counts are pinned in `reports/model_sources.json`
the width curve is in `reports/retrieval_union_india.json`

the final matcher can blend neural and existing tree log odds
on the selected query diagnostic a 0.6 neural weight raises recall at 0.995 precision from 0.91611 to 0.93397
a top 3 upstream tree filter retains essentially the same precision focused result while reducing neural calls from about 1.5 million to 53481
its lower candidate recall is measured explicitly rather than hidden
the full width and precision comparisons are in `reports/neural_gate_diagnostics.json`
this motivates a learned blocking filter followed by the final neural matcher
only the last candidate set actually scored by that final matcher is exported
full pool calibration and the locked audit remain required before final selection

## production execution and review

the complete a100 pilot processed 7974 records and scored 23922 final candidates with exact coverage
reference and model setup took 1301.85 seconds
steady processing took 95.03 seconds comprising 52.29 retrieval 33.68 feature and gate work and 9.06 neural matching
this is about 84 records per second on one node and motivates parallel country and row shards

the multi-gpu execution used `Standard_NC96ads_A100_v4` with 96 vcpus and four a100 80 gb gpus
single-gpu training and parallel scoring used `Standard_NC24ads_A100_v4` with 24 vcpus and one a100 80 gb
cpu aggregation and export used `Standard_E16ds_v4` with 16 vcpus

review fixes require resumed children to revalidate current model hashes before reusing predictions
final packages require all selected retriever snapshots and automatically use their local offline cache
missing or unmatched country labels use an explicitly marked unpartitioned reference fallback
normal country retrieval including france remains partitioned
production workers receive explicit gpu affinity and bounded tree inference threads
run indexes bind dataset and model content rather than azure mount paths so interrupted shards can resume in a new job
root inference signatures and exact coverage are rechecked after relocation

the dataset-wide validation score final cutoff test outputs and submission archive are still pending

## follow up on indic language and address suggestions

sources checked on 2026-09-25

- `https://huggingface.co/intfloat/multilingual-e5-base`
- `https://huggingface.co/intfloat/multilingual-e5-large-instruct`
- `https://arxiv.org/html/2402.05672v1`
- `Ml_Challenge.txt` lines 141 to 150

the task includes cross-script matching from indian-language aliases to predominantly latin-script references
the existing anyascii comparison view therefore remains important
raw unicode is retained for neural input and within-script comparisons but a raw-unicode-only lexical index would not bridge scripts

the claim that e5-base is suitable only for monolingual search is incorrect
its official card describes multilingual training including translation pairs and support inherited from xlm-roberta for 100 languages
the technical report evaluates cross-lingual bitext mining as well as multilingual retrieval
language coverage does not guarantee business identity accuracy in every language

large-instruct initializes from xlm-roberta-large and has 24 layers and 1024-dimensional embeddings
it also uses different instruction-tuning data rather than being only a wider base model
queries require `Instruct: <task>\nQuery: <record>` while reference documents have no query or passage prefix
the cited sources do not establish that token fragmentation alone explains an indic performance gap or that every larger variant is better for this dataset

existing features already include character tfidf raw unicode fuzzy comparisons token overlap transliterated ratios number-set agreement missing-address flags and mined hard negatives
a matched isolated experiment added compact and transliterated name comparisons token containment 2/3/4-gram evidence and soft address number/locality cues
both baselines used the same india training entities fixed lexical candidate universe 800 lightgbm trees and fold-0 tuning queries
recall at 0.995 pair precision improved from 0.855303 to 0.880135
top-3 candidate-owner recall changed from 0.939206 to 0.939061 so this is mainly a scoring gain rather than a retrieval gain
the possible-pin feature had zero tree importance in this experiment
details are in `reports/indic_feature_followup.json`

an exact-pair intersection check reused existing neural scores without new inference
it covered all 351878 lexical candidates but only 46.72 percent of the larger lexical-plus-e5 neural universe
conditional recall at 0.995 precision improved from 0.938535 to 0.950950 with the 0.6 neural logit blend
that denominator contains only the 6524 positive owners covered by the intersection rather than all 6925 selected positive owners
the corresponding all-selected-positive recall is 6123/6925 versus 6204/6925
performance at 0.999 precision was slightly worse
these are sampled diagnostics not official full-target macro f0.5 and do not by themselves justify replacing the production configuration
the locked fold-1 audit must not be repurposed to select a feature variant

a separate pinned large-instruct experiment uses the full india reference pool and the same 8925 held-out queries
revision `274baa43b0e13e37fafa6428dbc7938e62e5c439` has 559890432 parameters and mit metadata
its local benchmark uses the documented instruction format and records truncation recall and runtime
the existing production configuration continues while this experiment runs

geocoding apis are explicitly prohibited by the challenge
an offline external gazetteer would still add external reference data
address cues must instead come from the supplied strings and training labels
numeric evidence remains soft because genuine matching pairs can contain edited or missing numbers

### completed large-instruct comparison

the local large-instruct run completed against all 883188 india references and the same 8925 tuning queries
none of these records were truncated at 512 tokens; the longest query was 124 tokens and reference 107
the original export failed because its output directory was absent
that was fixed and both fully encoded caches were reused for the final retrieval result
the reported cache-reload times are not fresh encoding throughput measurements

at equal dense width 100 the current lexical plus base plus qwen union recalls 0.995379
replacing base with large-instruct recalls 0.995090
replacing qwen with large-instruct recalls 0.995235
adding large-instruct as a third dense encoder recalls 0.996101 and recovers 5 additional positive queries
at width 200 the third encoder recovers only 3 additional positive queries
large-instruct performs better at some narrow widths but is not a better two-encoder replacement at the selected production width
this initially left the two-retriever baseline unchanged pending a direct matcher comparison
the five-link gain did justify that follow-up; the later three-retriever deployment is described below

full same-width comparisons and scope are recorded in `reports/e5_large_same_width_india.json`
the individual model result is in `reports/e5_large_instruct_india.json`

the submission deadline is sunday 2026-09-27 at 08:00 ist
team amazites comprises varun jhaveri, shivsharan sanjawad, raj mathuria, and aastha singh

## current execution update

the baseline and upgraded implementations are now distinct frozen configurations

| item | baseline | upgrade |
| --- | --- | --- |
| dense retrievers | e5-base and qwen | e5-base qwen and e5-large |
| gate | native lightgbm/catboost mean | 54-feature teammate lightgbm without four sampled s1 aggregates |
| final matcher | trained e5 pair classifier | same trained e5 pair classifier |
| final candidates per target | up to 3 | up to 3 |
| neural logit weight | 0.6 | 0.6 |

the teammate feature port matched its original transform arrays and checkpoint probabilities exactly on 2613 pairs from 64 queries
the real upgraded runtime smoke covered 12 targets and 36 final candidates
the sharded launcher was also checked with the actual selected models
these establish implementation parity and execution, not final matching quality

the safer teammate gate improved paired recall at 99.5 percent precision from about 91.89 to 93.32 percent with unchanged neural scores
the compared population is a fixed selected candidate diagnostic
the uploaded diagnostics referenced a 0.9688 validation result and an earlier 0.958 leaderboard figure; they did not verify a new 0.97 leaderboard result for the uploaded ensemble version

validation target scoring and competition test scoring now run independently
the upgraded work is divided into four validation partitions and four test partitions on separate single-a100 workers
the original baseline test work also runs across four single-a100 workers
each worker uses two processes with twelve cpu threads per process
calibration and export are downstream cpu work

the original four-a100 validation worker lost allocation after 9161442 targets
its 41 manifests had no missing or unlisted checkpoint files
remaining work was moved to a smaller a100 using a byte-verified original runtime archive and copied outputs
copied checkpoint counts are distinguished from new scoring progress

an initial eight-job upgrade launch failed before inference when the azure sdk concurrently uploaded the same local gate folder
the fix was to stage and byte-verify that asset once and use its datastore uri for every job
all eight retry jobs subsequently reached running state and began writing prediction parts

## submission size criterion and remaining gates

candidate count per s1 is now an explicit organizer ranking criterion
the strict validator reports total pairs empty rows mean nearest-rank p50/p95/p99 maximum and the full size histogram
the file remains the actual final pre-neural candidate set, including pairs the matcher rejects
three candidates per target does not imply three candidates per reference

the current dense search uses exact chunked similarity scans
it bounds memory but does not establish billion-record approximate-search scalability
an ann replacement remains a measured future scaling change rather than an implemented result

quota is distinct from regional physical gpu availability

full labeled-pool calibration locked audit complete test export strict validation and archive checks remain required
the current priority is an upgraded first upload, then two refinements informed by validation and leaderboard feedback
complete full-pool validation is not a prerequisite for the explicitly provisional first export
complete test coverage correct file format and exact candidate membership remain mandatory
upgraded v1 was submitted with provisional cutoff 0.8 and received a reported public leaderboard f0.5 of 0.969
the original baseline offline locked-audit macro f0.5 is 0.9755015568; these results use different evaluation populations

current operator documentation is in [arch](docs/arch.md), [ops](docs/ops.md), [training](docs/training.md), and [status](docs/status.md)
the [evidence index](reports/README.md) maps every main result to its scope
