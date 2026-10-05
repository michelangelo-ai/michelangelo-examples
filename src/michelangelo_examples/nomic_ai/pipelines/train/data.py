"""Data loading utilities for the Nomic BERT / WikiText training example.

Ported from michelangelo core's ``python/examples/nomic_ai/data.py``, with
the ``@uniflow.task`` decorator and its ``RayTask`` resource config removed
-- this project ships ``__main__.py``-only (no ``pipeline.py``/
``pipeline.yaml`` to port; see this project's own README), so ``load_data``
is called as a plain function from ``__main__.py`` instead of being
dispatched as a uniflow task.

Loads and tokenizes WikiText dataset for Nomic BERT model training.
"""

from __future__ import annotations

import logging

import ray
from datasets import load_dataset
from ray.data import Dataset
from transformers import AutoTokenizer

log = logging.getLogger(__name__)


def load_data(
    model_name: str = "nomic-ai/nomic-bert-2048",
    dataset_name: str = "wikitext",
    max_length: int = 512,
    dataset_size: int = 200,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load and tokenize WikiText dataset for model training.

    Args:
        model_name: HuggingFace model name for tokenizer. Defaults to
            "nomic-ai/nomic-bert-2048".
        dataset_name: Dataset name from HuggingFace. Defaults to "wikitext".
        max_length: Maximum sequence length for tokenization. Defaults to 512.
        dataset_size: Number of samples per split, before the subsequent
            1% random sample applied below. Defaults to 200.

    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset) as Ray Datasets.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize_function, batched=True)

    for split in ["train", "validation", "test"]:
        if split in dataset:
            dataset[split] = dataset[split].select(
                range(min(dataset_size, len(dataset[split])))
            )

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Required more memory in cluster run, so sample 1% of the data
    def _load_slice(data_slice) -> Dataset:
        ds = ray.data.from_huggingface(data_slice)
        ds = ds.random_sample(0.01, seed=1)

        return ds

    return (
        _load_slice(dataset["train"]),
        _load_slice(dataset["validation"]),
        _load_slice(dataset["test"]),
    )
