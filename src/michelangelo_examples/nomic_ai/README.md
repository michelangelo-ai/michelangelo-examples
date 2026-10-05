# nomic_ai

A project (use case): fine-tuning/encoding with a long-context
(2048-token) BERT architecture, `nomic-ai/nomic-bert-2048`, over the
WikiText-2 language-modeling dataset, via PyTorch Lightning. Pipelines
under this project share one dependency set (`pip install
"michelangelo-examples[nomic-ai]"`) and one built image
(`ghcr.io/michelangelo-ai/michelangelo-examples:nomic-ai`), applied
together via this project's own
[`config/project.yaml`](config/project.yaml)
(`ma project apply -f src/michelangelo_examples/nomic_ai/config/project.yaml`).

## Requires Python 3.11+

Unlike this repo's other projects (`bert-cola`, `california-housing`,
still on the package-wide `>=3.10` floor), this project's entry point
(`pipelines/train/__main__.py`) raises a `RuntimeError` at import time on
Python <3.11. This is a project-level requirement, not a `pip install`-time
restriction -- the top-level `michelangelo-examples` package still declares
`requires-python = ">=3.10"` so installing `bert-cola`/`california-housing`
on 3.10 is unaffected.

## Pipelines

- [`train`](pipelines/train/) -- downloads a small WikiText-2 sample, loads
  `nomic-ai/nomic-bert-2048`, and fine-tunes it via PyTorch Lightning. No
  `pipeline.py`/`pipeline.yaml` -- migrated from core `michelangelo`'s
  `python/examples/nomic_ai/`, following the same `__main__.py`-only shape
  as this repo's `movielens` project. See its own README for how to run it
  locally.

## Layout

- `config/project.yaml` -- this project's Michelangelo Project CRD config.
- `pipelines/train/` -- data, model, and training code, plus a
  `__main__.py` local runner and its own `README.md`. These ship as real
  package contents -- `pip install` gets the code and `README.md` together,
  not just the `.py` files.
