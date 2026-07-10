# michelangelo-examples

Pip-installable, ready-to-run example models and pipelines for
[Michelangelo](https://github.com/michelangelo-ai/michelangelo) — a
lightweight on-ramp for trying Michelangelo without a full monorepo
checkout, the heavyweight `example` extra, or a running sandbox.

```bash
pip install michelangelo-examples[<example-name>]
```

## Why this repo

The full example pipelines in
[`michelangelo/python/examples/`](https://github.com/michelangelo-ai/michelangelo/tree/main/python/examples)
are the canonical, full-fidelity reference — they exercise real
Cadence/Temporal orchestration, Spark, and a live sandbox, and they aren't
going anywhere. But trying one today means checking out the whole monorepo
and installing one heavyweight extra that bundles every example's
dependencies (`torch`, `transformers`, `xgboost`, `pyspark`, `ray`,
`mlflow`, `peft`, and more) at once, regardless of which example you
actually want.

`michelangelo-examples` is a second, lighter tier: each example is its own
pip extra with only the dependencies it needs, plus a plain-Python
`run_local.py` entrypoint that trains/evaluates locally — no Cadence, no
Spark, no sandbox. It's the "try it in five minutes" path underneath the
full pipeline reference, not a replacement for it.

## Examples

*(Coming soon — v1 will port a small initial set, starting with
`california-housing`, `movielens`, and `bert-cola`. See the
[project spec](https://github.com/michelangelo-ai/michelangelo/tree/main/python/examples)
for the full list this repo is drawing from.)*

Each example will live under `examples/<example-name>/` with its own
`README.md`, `model.py` (or an import from
[`michelangelo-models`](https://github.com/michelangelo-ai/michelangelo-models)
where applicable), and `run_local.py`.

## Installing an example

```bash
pip install michelangelo-examples[california-housing]
python -m michelangelo_examples.california_housing.run_local
```

Only that example's dependencies are installed.

## Relationship to other Michelangelo repos

- [`michelangelo`](https://github.com/michelangelo-ai/michelangelo) — the
  core platform; `python/examples/` there remains the full pipeline
  reference (Cadence/Temporal/Spark/sandbox).
- [`michelangelo-models`](https://github.com/michelangelo-ai/michelangelo-models) —
  pluggable model architecture definitions. Some examples here may import
  a seed architecture from there instead of defining their own model.

## Development

This repo uses [`uv`](https://docs.astral.sh/uv/) for dependency
management.

```bash
uv build
uv run pytest
```

## License

[Apache License 2.0](LICENSE)
