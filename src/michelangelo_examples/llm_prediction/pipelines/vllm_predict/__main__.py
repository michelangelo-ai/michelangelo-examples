"""Local entrypoint for LLM batch prediction, vLLM backend (GPU-only).

Ported from michelangelo core's
``python/examples/llm_prediction/vllm_prediction.py``. Like core, this runs
``llm_prediction_workflow`` (a real ``@uniflow.workflow()``) locally via
``ctx.run(...)``; there is no ``pipeline.yaml``, because core never dispatched
this example through ``ma pipeline apply``/``ma pipeline run`` either. The
workflow is identical to the one in the sibling ``hf_predict`` pipeline
except for the backend-specific ``predict`` task (and its extra
``tensor_parallel_size`` parameter); it is duplicated rather than shared,
matching this repo's self-contained ``pipelines/<name>/`` convention.

Requires the ``llm-prediction-vllm`` extra, which only installs on Linux
AMD64 with a CUDA-capable GPU. Defaults: 15 prompts from the ``openai/gsm8k``
test split, batch size 8, one GPU, ``Qwen/Qwen2.5-0.5B``. Predictions are
written as CSV under ``./llm_prediction/``.

Usage:
    python -m michelangelo_examples.llm_prediction.pipelines.vllm_predict
"""

from __future__ import annotations

import sys

# vllm==0.9.2 (this backend's pinned version) declares requires-python <3.13,
# so the supported window is 3.11 (this project's floor) through 3.12.
if not (3, 11) <= sys.version_info < (3, 13):
    raise RuntimeError(
        "llm-prediction (vllm backend) requires Python 3.11 or 3.12 "
        "(vllm==0.9.2 does not support 3.13+); this interpreter is "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import UF_PLUGIN_RAY_USE_FSSPEC

from michelangelo_examples.llm_prediction.pipelines.vllm_predict.data import (
    load_data,
    write_data,
)
from michelangelo_examples.llm_prediction.pipelines.vllm_predict.predict import (
    predict,
)


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
    tensor_parallel_size: int = 1,
):
    """LLM prediction workflow using vLLM for high-performance inference.

    Loads data from HuggingFace datasets, runs batch prediction with vLLM,
    and writes results to storage. Supports tensor parallelism for large models.

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
        tensor_parallel_size: Number of GPUs for tensor parallelism. Defaults to 1.
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
        tensor_parallel_size=tensor_parallel_size,
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

    # Run with vLLM; requires a CUDA GPU.
    ctx.run(
        llm_prediction_workflow,
        worker_gpu=1,
        worker_instances=1,
        tensor_parallel_size=1,
        data_path="openai/gsm8k",
        data_name="main",
        data_slice="test",
        data_predict_column="question",
        data_limit=15,
        batch_size=8,
        model_name="Qwen/Qwen2.5-0.5B",
        temperature=0.95,
        top_p=0.95,
        max_tokens=128,
    )
