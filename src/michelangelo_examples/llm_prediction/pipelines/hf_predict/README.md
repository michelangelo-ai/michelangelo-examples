# llm_prediction: hf_predict

Offline batch LLM inference with HuggingFace Transformers
(`transformers.pipeline("text-generation")`) over Ray Data, on CPU or GPU.
Part of the [`llm_prediction`](../../README.md) project; the sibling
[`vllm_predict`](../vllm_predict/) pipeline runs the same workflow on vLLM.

## Run

Requires Python 3.11+.

```bash
pip install "michelangelo-examples[llm-prediction]"
python -m michelangelo_examples.llm_prediction.pipelines.hf_predict
```

Defaults: `Qwen/Qwen2.5-0.5B`, 2 prompts from the `openai/gsm8k` test split,
batch size 1, `temperature=0.95`, `top_p=0.95`, one CPU worker. Output: CSV
(`prompt`, `generated_text`) under `./llm_prediction/`. To run on a GPU, set
`worker_gpu=1` in `__main__.py`'s `ctx.run(...)` call.

## Workflow

```
load_data  →  predict  →  write_data
  (Ray)        (Ray)        (Ray)
```

## Files

- `__main__.py`: the Python 3.11+ guard, `llm_prediction_workflow`, and the
  `ctx.run(...)` local invocation with default parameters.
- `predict.py`: `HFPredictor` (a Ray Data `map_batches` callable class) and
  the `predict` Ray task.
- `data.py`: `load_data` (HuggingFace dataset to Ray Dataset) and
  `write_data` (Ray Dataset to CSV) Ray tasks. Identical to
  `vllm_predict/data.py`.
