"""Local entrypoint for LLM batch prediction, HuggingFace Transformers backend.

Ported from michelangelo core's
``python/examples/llm_prediction/hf_prediction.py``. Like core, this runs
``llm_prediction_workflow`` (a real ``@uniflow.workflow()``) locally via
``ctx.run(...)``; there is no ``pipeline.yaml``, because core never dispatched
this example through ``ma pipeline apply``/``ma pipeline run`` either. The
workflow is identical to the one in the sibling ``vllm_predict`` pipeline
except for the backend-specific ``predict`` task it calls; it is duplicated
rather than shared, matching this repo's self-contained ``pipelines/<name>/``
convention.

Defaults are a CPU-feasible smoke test: 2 prompts from the ``openai/gsm8k``
test split, batch size 1, ``Qwen/Qwen2.5-0.5B``. Predictions are written as
CSV under ``./llm_prediction/``. (Core defaulted to ``THUDM/LongBench``,
which is a loading-script-only dataset that ``datasets`` no longer loads
without a pre-populated local cache -- see this project's README.)

Usage:
    python -m michelangelo_examples.llm_prediction.pipelines.hf_predict
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "llm-prediction (hf backend) requires Python 3.11+; this interpreter is "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import UF_PLUGIN_RAY_USE_FSSPEC

from michelangelo_examples.llm_prediction.pipelines.hf_predict.data import (
    load_data,
    write_data,
)
from michelangelo_examples.llm_prediction.pipelines.hf_predict.predict import predict


@uniflow.workflow()
def llm_prediction_workflow(
    data_path: str,
    data_name: str,
    data_slice: str,
    data_predict_column: str,
    data_limit: int,
    batch_size: int,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    worker_instances: int = 1,
    worker_gpu: int = 0,
):
    """LLM prediction workflow using HuggingFace models.

    Loads data from HuggingFace datasets, runs batch prediction with HF transformers,
    and writes results to storage.

    Args:
        data_path: HuggingFace dataset path.
        data_name: Dataset configuration name.
        data_slice: Dataset split to use.
        data_predict_column: Column containing text to predict on.
        data_limit: Maximum number of samples to process.
        batch_size: Batch size for prediction.
        model_name: HuggingFace model identifier.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum tokens to generate.
        worker_instances: Number of Ray workers. Defaults to 1.
        worker_gpu: GPUs per worker. Defaults to 0.
    """
    predict_data = load_data(
        path=data_path,
        name=data_name,
        data_slice=data_slice,
        predict_column=data_predict_column,
        limit=data_limit,
    )

    result = predict(
        predict_data,
        worker_gpu=worker_gpu,
        worker_instances=worker_instances,
        batch_size=batch_size,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    if result:
        write_data(
            dataset=result,
            out_path="llm_prediction",
            partitions=1,
        )
    print("ok.")


if __name__ == "__main__":
    ctx = uniflow.create_context()

    # Disable use of fsspec in Ray Plugin. See UF_PLUGIN_RAY_USE_FSSPEC
    # docstring for more information.
    ctx.environ[UF_PLUGIN_RAY_USE_FSSPEC] = "0"
    ctx.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0"
    ctx.environ["MA_NAMESPACE"] = "default"
    ctx.environ["IMAGE_PULL_POLICY"] = "IfNotPresent"

    # Run with HF for CPU or GPU
    ctx.run(
        llm_prediction_workflow,
        worker_gpu=0,
        worker_instances=1,
        data_path="openai/gsm8k",
        data_name="main",
        data_slice="test",
        data_predict_column="question",
        data_limit=2,
        batch_size=1,
        model_name="Qwen/Qwen2.5-0.5B",
        temperature=0.95,
        top_p=0.95,
        max_tokens=128,
    )
