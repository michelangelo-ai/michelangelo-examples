# bert_cola

A project (use case): fine-tuning BERT for linguistic acceptability
classification on the Corpus of Linguistic Acceptability (CoLA) task from
the GLUE benchmark. Pipelines under this project share one dependency set
(`pip install "michelangelo-examples[bert-cola]"`) and one built image
(`ghcr.io/michelangelo-ai/michelangelo-examples:bert-cola`), applied
together via this project's own
[`config/project.yaml`](config/project.yaml)
(`ma project apply -f src/michelangelo_examples/bert_cola/config/project.yaml`).

## Pipelines

- [`train`](pipelines/train/) -- fine-tunes `bert-base-cased` on CoLA via a
  plain HuggingFace `Trainer`, packages it for Triton via the custom
  (Python-backend) assembler path, and pushes it to storage/registry. See
  its own README for how to run it locally or against a Michelangelo
  sandbox.

## Deploying to Triton

[`inferenceserver/`](inferenceserver/) holds this project's deploy
manifests. Rather than a custom serving image, `pythonDependencies` in
`inferenceserver.yaml` declares the packages this project's serving code
needs. See [`inferenceserver/README.md`](inferenceserver/README.md) (or
the `train` pipeline's README) for how to deploy a trained model.

## Layout

- `config/project.yaml` -- this project's Michelangelo Project CRD config.
- `pipelines/<pipeline-name>/` -- one directory per pipeline: model,
  training, and pipeline-task code, this pipeline's `pipeline.yaml`, and
  its own `README.md`. These ship as real package contents -- `pip
  install` gets the code, `pipeline.yaml`, and `README.md` together, not
  just the `.py` files.
- `inferenceserver/` -- this project's `InferenceServer`/`Deployment`
  manifests for deploying a trained model. Declares the Python packages its
  serving code needs via `servingSpec.pythonDependencies` rather than a
  custom Dockerfile.
