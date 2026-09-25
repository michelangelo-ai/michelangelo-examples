# California Housing: Retrain (triggers pytorch-train)

Demonstrates triggering an already-registered pipeline (`pytorch-train`) as
a child run from another pipeline's Uniflow workflow, via
`michelangelo.uniflow.plugins.pipeline.run_pipeline`.

Part of the `california_housing` project (use case); for the pipeline this
one triggers, see [`pytorch_train/`](../pytorch_train/README.md).

No local runner: `run_pipeline` requires a live `PipelineRunService` (a real
sandbox/cluster), unlike `pytorch_train`/`xgb_train`'s `python -m` local
smoke path.

## Prerequisites

- A Michelangelo sandbox running (`ma sandbox create`)
- The `california-housing` project applied from this project's own config:
  `ma project apply -f src/michelangelo_examples/california_housing/config/project.yaml`
- `pytorch-train` already registered (see
  [`pytorch_train/README.md`](../pytorch_train/README.md)'s "End-to-end:
  sandbox to running pipeline" section)
- This `retrain` pipeline itself registered:
  `ma pipeline apply -f src/michelangelo_examples/california_housing/pipelines/retrain/pipeline.yaml`

## Run

```bash
ma pipeline run -n california-housing --name retrain
```

Or start the `weekly-retrain` trigger declared in `pipeline.yaml` so it runs
on its own cron schedule (Monday 06:00 UTC), the same one-time way
`pytorch_train`'s own `daily-noon-run` trigger is started:

```bash
ma trigger_run create --namespace=california-housing --pipeline=retrain --trigger-name=weekly-retrain
```

## Future follow-up (out of scope here)

A fuller retrain-and-deploy example (model lookup via
`get_models_by_pipeline_run` + `create_or_update_deployment`/
`wait_for_deployment`) is possible since `pytorch-train`'s `push_step` does
register a discoverable model in non-local runs — left as a separable
future addition.
