# llm_prediction

A project (use case): offline batch inference with a large language model
over a HuggingFace dataset, run as a local uniflow workflow on Ray. Ships two
sibling pipelines, one per inference backend, with the same workflow shape:

| Pipeline | Backend | Hardware | Extra |
|---|---|---|---|
| [`hf_predict`](pipelines/hf_predict/) | HuggingFace Transformers | CPU or GPU | `llm-prediction` |
| [`vllm_predict`](pipelines/vllm_predict/) | vLLM | Linux AMD64 + CUDA GPU only | `llm-prediction-vllm` |

Each backend has its own extras group, so the HF path never installs vLLM.
The project's built image
(`ghcr.io/michelangelo-ai/michelangelo-examples:llm-prediction`) bakes in
the `llm-prediction` (HF) extra. The project is applied via its own
[`config/project.yaml`](config/project.yaml)
(`ma project apply -f src/michelangelo_examples/llm_prediction/config/project.yaml`).

## Requires Python 3.11+

Unlike this repo's other projects (`bert-cola`, `california-housing`,
still on the package-wide `>=3.10` floor), both entry points raise a
`RuntimeError` at import time on Python <3.11. The vLLM backend also
requires Python <3.13, because `vllm==0.9.2` doesn't support 3.13+. This is
a project-level requirement, not a `pip install`-time restriction: the
top-level `michelangelo-examples` package still declares
`requires-python = ">=3.10"`.

## Run

HuggingFace Transformers backend (CPU-feasible: 2 prompts, batch size 1):

```bash
pip install "michelangelo-examples[llm-prediction]"
python -m michelangelo_examples.llm_prediction.pipelines.hf_predict
```

vLLM backend (needs a CUDA GPU: 15 prompts, batch size 8, 1 GPU):

```bash
pip install "michelangelo-examples[llm-prediction-vllm]"
python -m michelangelo_examples.llm_prediction.pipelines.vllm_predict
```

Install each backend's extra on its own. If you use `uv sync` against this
repo's lockfile, `llm-prediction-vllm` can't be combined with any other
extra: `vllm==0.9.2` pins `torch==2.7.0`, so it gets its own resolution.

Both pipelines run `load_data` -> `predict` -> `write_data` as uniflow Ray
tasks on a local Ray instance. They load `Qwen/Qwen2.5-0.5B` and the
`openai/gsm8k` test split from HuggingFace, so the first run needs network
egress (both are cached under `~/.cache/huggingface/` afterward). Both log
`Wrote <n> items to llm_prediction` and print `ok.` on success. Predictions
(`prompt`, `generated_text`) are written as CSV under `./llm_prediction/`.

To change the model, dataset, sampling parameters, or worker/GPU counts,
edit the `ctx.run(...)` arguments at the bottom of each pipeline's
`__main__.py`.

## Layout

- `config/project.yaml`: this project's Michelangelo Project CRD config.
- `pipelines/hf_predict/` and `pipelines/vllm_predict/`: each holds a
  `__main__.py` (the Python version guard, `llm_prediction_workflow`, and
  its `ctx.run(...)` call), a `predict.py` (the backend-specific `predict`
  Ray task), a `data.py` (`load_data`/`write_data` Ray tasks), and its own
  `README.md`.
  `data.py` and the workflow function are intentionally duplicated across
  the two folders, because each `pipelines/<name>/` folder in this repo is
  self-contained.

## Deviations from core

Migrated from core `michelangelo`'s `python/examples/llm_prediction/`
(`hf_prediction.py`/`hf_predict.py`, `vllm_prediction.py`/`vllm_predict.py`,
`data.py`).

- **Two pipeline folders instead of four flat files.** Core's
  `hf_prediction.py`/`vllm_prediction.py` became each folder's `__main__.py`,
  and `hf_predict.py`/`vllm_predict.py` became each folder's `predict.py`.
  The uniflow decorators, Ray task configs, and `ctx.run(...)` invocation
  are unchanged. There is still no `pipeline.yaml`: core never dispatched
  this example through `ma pipeline run` either.
- **Default dataset changed from `THUDM/LongBench` (`2wikimqa`) to
  `openai/gsm8k` (`main`, column `question`).** LongBench ships only as a
  `LongBench.py` loading script. `datasets>=4` refuses to run loading
  scripts (`Dataset scripts are no longer supported`). `datasets` 3.x
  refuses it without `trust_remote_code=True`, which would execute code
  pulled from the Hub. Either way, core's example fails unless LongBench is
  already in the local HuggingFace cache. GSM8K is parquet-native, about 7 MB
  (vs. LongBench's 114 MB archive), and has short prompts that suit a CPU
  smoke test. `load_data` itself is unchanged, so any dataset with a text
  column can be passed in instead.
- **Python version guards.** Both entry points require Python 3.11+. The
  vLLM entry point also rejects Python 3.13+.
