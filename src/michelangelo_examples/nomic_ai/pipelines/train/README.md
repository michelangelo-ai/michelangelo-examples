# Nomic BERT / WikiText example

Fine-tunes/encodes `nomic-ai/nomic-bert-2048` -- a long-context (2048-token)
BERT architecture -- against a small sample of the WikiText-2 language
modeling dataset, via PyTorch Lightning. CPU-feasible, no credentials
required.

Migrated from core `michelangelo`'s `python/examples/nomic_ai/`. Unlike
`movielens`, core's `nomic_ai.py` *does* define a real
`@uniflow.workflow()` (`train_workflow`) wiring `load_data` -> `train` --
but per this migration's design, this project still ships the
`__main__.py`-only shape, not a ported `pipeline.py`/`pipeline.yaml`. See
[Deviations from core](#deviations-from-core) below.

## Run

Requires Python 3.11+ (see this project's own [README](../../README.md)).

```bash
pip install "michelangelo-examples[nomic-ai]"
python -m michelangelo_examples.nomic_ai.pipelines.train
```

The first invocation downloads the WikiText-2 dataset and the
`nomic-ai/nomic-bert-2048` model/tokenizer weights from HuggingFace, so it
needs network egress on first run (HuggingFace caches both locally
afterward, under `~/.cache/huggingface/` by default). The fine-tuned model
and tokenizer are saved to `./nomic_ai`.

## What it exercises

- Loading a `trust_remote_code=True` HuggingFace model
  (`nomic-ai/nomic-bert-2048`) and wrapping it in a plain
  `pytorch_lightning.LightningModule`.
- Tokenizing and sampling a HuggingFace `datasets` dataset, then converting
  it to a Ray Dataset via `ray.data.from_huggingface`.
- A plain (non-Ray-dispatched) `pytorch_lightning.Trainer` fit loop, with a
  `DeepSpeedStrategy` path when CUDA is available and a CPU fallback
  otherwise.

## Files

- `data.py` -- loads and tokenizes a WikiText-2 sample, returns Ray
  datasets (`load_data`).
- `model.py` -- `HuggingFaceLightningModel`, a thin
  `pytorch_lightning.LightningModule` wrapper around the HuggingFace model.
- `train.py` -- wires the Ray datasets into plain `torch` `DataLoader`s and
  drives the `pl.Trainer` fit (`train`).
- `__main__.py` -- the local entrypoint (`python -m ...pipelines.train`);
  calls `load_data` then `train` directly in plain Python; also carries the
  Python 3.11+ runtime guard.

## Deviations from core

- **No ported `pipeline.py`/`pipeline.yaml`.** Core's `nomic_ai.py` is a
  real `@uniflow.workflow()`, unlike `movielens`'s flat script -- but this
  migration ships the same `__main__.py`-only shape as `movielens` for both
  projects, per this repo's migration design. `__main__.py` reproduces
  `train_workflow`'s `load_data` -> `train` call directly, without the
  uniflow dispatch wrapper.
- **`load_data`/`train` are plain functions, not uniflow tasks.** The
  `@uniflow.task`/`RayTask` decorators and their Ray-cluster resource
  configs are removed; `__main__.py` calls both functions as ordinary
  Python, matching the "no Cadence, no sandbox" local-tier framing this
  repo's README describes.
- **`__main__.py` uses a smaller sample than core's default** (`max_length`
  128 tokens instead of 512, `dataset_size` 500 instead of 200 rows before
  the existing 1% random sub-sample) so the local smoke test completes
  quickly on CPU while still leaving a non-empty post-sample split.
- **Requires `transformers<5`** (see this project's `pyproject.toml` extras
  comment for the full writeup). `nomic-ai/nomic-bert-2048`'s own
  `trust_remote_code=True` modeling file calls
  `self.get_extended_attention_mask()`, a method transformers 5.x removed
  from `PreTrainedModel`/`ModuleUtilsMixin` -- confirmed by actually running
  this pipeline against `transformers==5.18.0` (fails at `validation_step`
  with `AttributeError: 'NomicBertModel' object has no attribute
  'get_extended_attention_mask'`) and against `transformers==4.57.6` (runs
  to completion).
