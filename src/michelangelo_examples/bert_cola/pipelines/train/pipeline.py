"""BERT fine-tuning workflow for CoLA linguistic acceptability task.

Example workflow demonstrating BERT fine-tuning on the Corpus of Linguistic
Acceptability (CoLA) task from the GLUE benchmark.
Support workflow parameters via dict or Starlark-compatible parameters.
"""

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import UF_PLUGIN_RAY_USE_FSSPEC

from michelangelo_examples.bert_cola.pipelines.train.assembler import assembler
from michelangelo_examples.bert_cola.pipelines.train.data import load_data
from michelangelo_examples.bert_cola.pipelines.train.push import push_step
from michelangelo_examples.bert_cola.pipelines.train.train import train

__all__ = [
    "assembler",
    "load_data",
    "push_step",
    "train",
    "train_workflow",
]


@uniflow.workflow()
def train_workflow(path="nyu-mll/glue", name="cola", tokenizer_max_length=128):
    """Training workflow for BERT model on GLUE datasets."""
    print("[train_workflow] Starting with config:")
    print("  - Dataset: " + path + "/" + name)
    print("  - Tokenizer max length: " + str(tokenizer_max_length))

    # Load data using configuration
    train_data, validation_data, test_data = load_data(
        path=path,
        name=name,
        tokenizer_max_length=tokenizer_max_length,
    )
    train_result, model_variable = train(
        train_data,
        validation_data,
        test_data,
    )
    print("train_result:", train_result)

    assembled = assembler(
        model_variable,
        # Must match train.py's hardcoded lr/eps.
        lr=2e-5,
        eps=1e-8,
        tokenizer_max_length=tokenizer_max_length,
    )
    print("assembled model:", assembled)

    push_results = push_step(assembled)
    print("push results:", push_results)
    print("ok.")


if __name__ == "__main__":
    ctx = uniflow.create_context()

    # Set the environment variable DATA_SIZE to let the load_data task
    # know how much data to generate.
    ctx.environ["DATA_SIZE"] = "10"

    # Disable use of fsspec in Ray Plugin. See UF_PLUGIN_RAY_USE_FSSPEC
    # docstring for more information.
    ctx.environ[UF_PLUGIN_RAY_USE_FSSPEC] = "0"
    ctx.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0"
    ctx.environ["MA_NAMESPACE"] = "default"
    ctx.environ["IMAGE_PULL_POLICY"] = "IfNotPresent"
    ctx.environ["S3_ALLOW_BUCKET_CREATION"] = "True"

    ctx.run(train_workflow)
