# llm_prediction: vllm_predict

Offline batch LLM inference with [vLLM](https://github.com/vllm-project/vllm)
over Ray Data, with optional tensor parallelism across GPUs. Part of the
[`llm_prediction`](../../README.md) project; the sibling
[`hf_predict`](../hf_predict/) pipeline runs the same workflow on
HuggingFace Transformers (CPU-feasible).

## Run

Requires Python 3.11 or 3.12, and Linux AMD64 with a CUDA-capable GPU: the
`llm-prediction-vllm` extra pins `vllm==0.9.2`, which publishes a Linux
x86_64 wheel only and doesn't support Python 3.13+.

```bash
pip install "michelangelo-examples[llm-prediction-vllm]"
python -m michelangelo_examples.llm_prediction.pipelines.vllm_predict
```

Defaults: `Qwen/Qwen2.5-0.5B`, 15 prompts from the `openai/gsm8k` test
split, batch size 8, one GPU, `tensor_parallel_size=1`. Output: CSV
(`prompt`, `generated_text`) under `./llm_prediction/`.

When `tensor_parallel_size > 1`, `predict` reserves a `STRICT_PACK` Ray
placement group with one GPU bundle per tensor-parallel worker for each
model replica. If `worker_gpu` isn't divisible by `tensor_parallel_size`,
it logs a warning and uses all of `worker_gpu` for tensor parallelism.

## Workflow

```
load_data  →  predict  →  write_data
  (Ray)        (Ray)        (Ray)
```

## Files

- `__main__.py`: the Python 3.11/3.12 guard, `llm_prediction_workflow`, and
  the `ctx.run(...)` local invocation with default parameters.
- `predict.py`: `VLLMPredictor` (a Ray Data `map_batches` callable class)
  and the `predict` Ray task.
- `data.py`: `load_data` (HuggingFace dataset to Ray Dataset) and
  `write_data` (Ray Dataset to CSV) Ray tasks. Identical to
  `hf_predict/data.py`.
