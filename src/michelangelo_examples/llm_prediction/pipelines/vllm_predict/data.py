"""Data loading utilities for LLM prediction workflows.

This module provides functions to load datasets from HuggingFace and convert them
to Ray Datasets for distributed processing.

Ported unchanged from michelangelo core's
``python/examples/llm_prediction/data.py``. An identical copy lives in the
sibling ``hf_predict`` pipeline: each ``pipelines/<name>/`` folder in
this repo is self-contained rather than sharing code across pipelines.
"""

import logging

import datasets
import michelangelo.uniflow.core as uniflow
import ray
from michelangelo.uniflow.plugins.ray import RayTask
from ray.data import Dataset

log = logging.getLogger(__name__)


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_memory="2Gi",
        worker_cpu=1,
        worker_memory="2Gi",
        worker_instances=1,
        # breakpoint=True,
    )
)
def load_data(
    path: str,
    name: str,
    data_slice: str,
    predict_column: str,
    limit: int | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load dataset from HuggingFace and convert to Ray Dataset.

    Args:
        path: HuggingFace dataset path.
        name: Dataset configuration name.
        data_slice: Dataset split to load (e.g., 'train', 'test').
        predict_column: Column name to rename to 'text' for prediction.
        limit: Optional row limit for testing. Defaults to None.

    Returns:
        Ray Dataset with renamed prediction column.
    """
    hf_dataset = datasets.load_dataset(path=path, name=name)
    hf_dataset = hf_dataset.rename_column(predict_column, "text")
    if limit:
        hf_dataset = hf_dataset[data_slice].select(range(limit))
    return ray.data.from_huggingface(hf_dataset)


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_memory="2Gi",
        worker_cpu=1,
        worker_memory="2Gi",
        worker_instances=1,
        # breakpoint=True,
    )
)
def write_data(
    dataset: Dataset,
    out_path: str,
    partitions: int | None = None,
):
    """Write Ray Dataset to CSV files.

    Args:
        dataset: Ray Dataset to write.
        out_path: Output directory path for CSV files.
        partitions: Optional number of partitions to repartition before writing.
            Defaults to None.
    """
    if partitions:
        dataset = dataset.repartition(partitions)
    dataset.write_csv(out_path)
    log.info(f"Wrote {dataset.count()} items to {out_path}")
