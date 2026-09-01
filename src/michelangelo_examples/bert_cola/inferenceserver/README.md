# bert_cola: Triton Serving

This project's own Triton deploy manifests. Rather than a custom serving
image, `inferenceserver.yaml` declares the Python packages the custom
python-backend `../pipelines/train/assembler.py` produces needs
(`servingSpec.pythonDependencies`) -- the Triton pod installs them into a
shared volume via an init container at startup, so no Dockerfile, registry,
or CI build is required.

- `inferenceserver.yaml` -- the `InferenceServer` CR. `servingSpec.pythonDependencies`
  lists `torch==2.4.1`/`transformers==4.44.2`, the last versions with
  Python 3.8 wheels (the stock `nvcr.io/nvidia/tritonserver` image's own
  Python version).
- `deployment.yaml` -- the `Deployment` CR. Set `spec.desiredRevision.name`
  to the model name printed by a real `train_workflow` run's `push_step`
  before applying.

## Deploy

```bash
ma inference_server apply -f inferenceserver.yaml
# then, with desiredRevision.name set:
ma deployment apply -f deployment.yaml
```
