# Amazon Books Qwen Dual-Encoder

Recommendation system demo using a Qwen-based dual-encoder architecture
for the Amazon Books dataset. Demonstrates feature engineering with
Chronon, distributed data processing with Spark, and model training with
Ray.

Migrated from core `michelangelo`'s
[`python/examples/amazon_books_qwen/`](https://github.com/michelangelo-ai/michelangelo/tree/main/python/examples/amazon_books_qwen).

## Prerequisites

- **Python 3.11+** (this project requires Python 3.11 or later; older
  interpreters will fail immediately with a clear error message)
- **Kaggle API credentials** (`~/.kaggle/kaggle.json`) -- the dataset is
  downloaded from Kaggle at runtime (not shipped with this package). See
  [Kaggle API setup](https://github.com/Kaggle/kaggle-api#api-credentials)
  for instructions.

## Install

```bash
pip install "michelangelo-examples[amazon-books-qwen]"
```

## Pipelines

| Pipeline | Description | Shape |
|----------|-------------|-------|
| [`train`](pipelines/train/) | Chronon feature engineering + Qwen dual-encoder training | Uniflow only (no `__main__.py` local runner) |

## Why no `__main__.py`

Every task in this pipeline requires either a live Spark session
(`download_kaggle_dataset`, `compute_chronon_features_with_spark`) or
Ray (`train_dual_encoder`). There is no cheap, laptop-runnable subset
that would meaningfully exercise the pipeline without reimplementing it
entirely. `bert_cola` in this repo is an existing precedent for a
uniflow-only project with no local runner.

## Network requirements

This pipeline requires network egress at **run time** (not just install
time):
- Kaggle API for dataset download
- Maven Central for the Chronon JAR (the Hadoop S3A/AWS SDK JARs are
  preinstalled in the Spark image)
- HuggingFace Hub for model/tokenizer weights

If running in a network-restricted sandbox, ensure these endpoints are
reachable.
