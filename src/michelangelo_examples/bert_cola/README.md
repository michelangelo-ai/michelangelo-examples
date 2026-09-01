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

[`inferenceserver/`](inferenceserver/) holds this project's own Triton
serving image and deploy manifests -- this project owns its own serving
image, built from whatever deps its models actually need, rather than
relying on a single shared default. See
[`inferenceserver/README.md`](inferenceserver/README.md) (or the
`train` pipeline's README) for how to deploy a trained model.

## Layout

- `config/project.yaml` -- this project's Michelangelo Project CRD config.
- `pipelines/<pipeline-name>/` -- one directory per pipeline: model,
  training, and pipeline-task code, this pipeline's `pipeline.yaml`, and
  its own `README.md`. These ship as real package contents -- `pip
  install` gets the code, `pipeline.yaml`, and `README.md` together, not
  just the `.py` files.
- `inferenceserver/` -- this project's Triton serving Dockerfile and
  `InferenceServer`/`Deployment` manifests for deploying a trained model.
