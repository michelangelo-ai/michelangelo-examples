"""Amazon Books Qwen Dual-Encoder Pipeline.

Main workflow entry point for training a Qwen-based recommendation model
on the Amazon Books dataset using Chronon feature engineering and Spark.

This is a uniflow-only pipeline (no ``__main__.py`` local runner) because
every task in the workflow -- ``download_kaggle_dataset``,
``compute_chronon_features_with_spark``, and ``train_dual_encoder`` --
requires either a live Spark session or Ray, making a lightweight
laptop-runnable subset infeasible without reimplementing the workflow
entirely. See ``bert_cola`` for an existing precedent of a uniflow-only
project in this repo.

Migrated from core ``michelangelo``'s
``python/examples/amazon_books_qwen/amazon_books_qwen.py``.
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "amazon-books-qwen requires Python 3.11+; this interpreter is "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import UF_PLUGIN_RAY_USE_FSSPEC

from michelangelo_examples.amazon_books_qwen.pipelines.train.chronon_tasks import (
    compute_chronon_features_with_spark,
)
from michelangelo_examples.amazon_books_qwen.pipelines.train.download import (
    download_kaggle_dataset,
)
from michelangelo_examples.amazon_books_qwen.pipelines.train.train import (
    train_dual_encoder,
)

__all__ = [
    "amazon_books_qwen_workflow",
    "compute_chronon_features_with_spark",
    "download_kaggle_dataset",
    "train_dual_encoder",
]


@uniflow.workflow()
def amazon_books_qwen_workflow(sample_size=100):
    """Complete workflow for training Qwen dual-encoder on Amazon Books data.

    Steps:
        1. Download dataset from Kaggle (or use cached local copy)
        2. Compute Chronon temporal features via Spark
        3. Train Qwen-based dual-encoder model via Ray
    """
    dataset_config = {
        "max_query_tokens": 128,
        "max_doc_tokens": 512,
        "sample_size": sample_size,
        "negative_ratio": 1.0,
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
    }

    books_dv, reviews_dv = download_kaggle_dataset(dataset_config=dataset_config)

    train_dv, val_dv, test_dv = compute_chronon_features_with_spark(
        dataset_config=dataset_config, books_dv=books_dv, reviews_dv=reviews_dv
    )

    model_result = train_dual_encoder(
        train_dv=train_dv,
        val_dv=val_dv,
        test_dv=test_dv,
        embedding_dim=512,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=2,
        num_workers=1,
        use_gpu=False,
        distributed=False,
    )

    return model_result


if __name__ == "__main__":
    print("=" * 80)
    print("Amazon Books Qwen Dual-Encoder Pipeline")
    print("=" * 80)

    ctx = uniflow.create_context()

    ctx.environ["DATA_SIZE"] = "100"
    ctx.environ[UF_PLUGIN_RAY_USE_FSSPEC] = "0"
    ctx.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0"
    ctx.environ["MA_NAMESPACE"] = "default"
    ctx.environ["IMAGE_PULL_POLICY"] = "IfNotPresent"
    ctx.environ["S3_ALLOW_BUCKET_CREATION"] = "True"

    ctx.environ["QWEN_MODEL_SIZE"] = "local"
    ctx.environ["ENABLE_BF16"] = "False"
    ctx.environ["MAX_QUERY_LENGTH"] = "128"
    ctx.environ["MAX_DOC_LENGTH"] = "512"
    ctx.environ["SPARK_LOG_URL_PREFIX"] = "http://localhost:9091/logs"

    sample_size = 1000
    if ctx.is_local_run():
        print("=" * 80)
        print("Using smaller dataset")
        print("=" * 80)
        sample_size = 100

    result = ctx.run(amazon_books_qwen_workflow, sample_size)
