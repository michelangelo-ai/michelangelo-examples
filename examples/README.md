# Examples

Each example lives in its own subdirectory here, e.g. `examples/california-housing/`,
with:

- `model.py` — the example's model definition (or an import from
  [`michelangelo-models`](https://github.com/michelangelo-ai/michelangelo-models)
  where an equivalent architecture already exists there).
- `run_local.py` — a plain-Python entrypoint that trains/evaluates locally,
  with no Cadence/Temporal/Spark/sandbox dependency.
- `README.md` — what the example demonstrates and how to run it.

v1 candidates (not yet ported): `california-housing`, `movielens`,
`bert-cola`. See the harness spec for the full design rationale and open
questions on per-example dependency isolation.
