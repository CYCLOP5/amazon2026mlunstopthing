# models and licenses

our final pipeline uses public pretrained sources and task-specific fine-tuned checkpoints
model names retain their exact upstream identifiers
the inventory below records identity and purpose; complete machine-readable provenance remains with the model and scoring configurations

| upstream model | license | role |
| --- | --- | --- |
| `intfloat/multilingual-e5-small` | mit | task-trained retriever and pair-model members |
| `intfloat/multilingual-e5-base` | mit | earlier retrieval, pair classifiers and hybrid field embeddings |
| `intfloat/multilingual-e5-large-instruct` | mit | earlier retrieval and pair-model members |
| `BAAI/bge-reranker-v2-m3` | apache-2.0 | pair-model members |
| `Qwen/Qwen3-Embedding-0.6B` | apache-2.0 | earlier candidate-retrieval lane |
| `intfloat/multilingual-e5-large` | mit | hybrid neural expert |

lightgbm models are trained from the challenge-derived features and labels
frozen score assets preserve the predictions used by the final decision layer
using recorded scores does not change which upstream models generated them

## provenance

- neural source identities and revisions: `src/neural_v2/reports/model_sources.json`
- pair-ensemble member fingerprints: `src/neural_v2/reports/final_ensemble_members.json`
- earlier neural runtime: `src/fusion/upstream_neural/`
- hybrid and collective training configurations: `src/fusion/business_entity_resolution/configs/`
- score-producer identifiers and selected columns: `assets/manifest.json`

## notices and primary sources

- [e5 license notice](../src/neural_v2/licenses/e5-mit.txt)
- [apache-2.0 license text](../src/neural_v2/licenses/qwen3-apache-2.0.txt)
- [multilingual e5](https://huggingface.co/intfloat/multilingual-e5-base)
- [multilingual e5 instruction model](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- [bge reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [qwen embedding model](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [lightgbm](https://github.com/microsoft/LightGBM)

upstream license texts retain their original wording
the software environments separately record the licenses and versions of supporting libraries
