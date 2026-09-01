# bert_cola: Triton Serving

This project's own Triton serving image and deploy manifests -- each
project owns its own serving image, built from whatever deps its models
actually need, rather than relying on a single shared default image every
model gets forced onto.

- `Dockerfile` -- adds `torch`/`transformers` on top of the stock
  `nvcr.io/nvidia/tritonserver` image (needed by the custom python-backend
  package `../pipelines/train/assembler.py` produces). Pinned to
  `torch==2.4.1`/`transformers==4.44.2`, the last versions with Python 3.8
  wheels (the base image's own Python version).
- `inferenceserver.yaml` -- the `InferenceServer` CR, wired to the image
  above via `servingSpec.containerBuildTemplate`.
- `deployment.yaml` -- the `Deployment` CR. Set `spec.desiredRevision.name`
  to the model name printed by a real `train_workflow` run's `push_step`
  before applying.

Built and pushed by `.github/workflows/build-triton-image.yaml`
whenever `Dockerfile` changes.

## Deploy

```bash
ma inference_server apply -f inferenceserver.yaml
# then, with desiredRevision.name set:
ma deployment apply -f deployment.yaml
```
