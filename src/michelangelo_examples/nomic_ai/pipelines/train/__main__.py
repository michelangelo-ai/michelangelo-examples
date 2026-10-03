"""Local entrypoint for the Nomic BERT / WikiText example.

This project has no full uniflow ``pipeline.py``/``pipeline.yaml`` -- per
this migration's design, ``nomic_ai`` ships the same ``__main__.py``-only
local-runner shape as ``movielens``. Core's own
``python/examples/nomic_ai/nomic_ai.py`` does wire ``load_data`` ->
``train`` via a ``@uniflow.workflow()`` (``train_workflow``), but that
dispatch wiring is uniflow/Cadence-sandbox orchestration glue, not training
logic -- this ``__main__.py`` reproduces the same two-step call directly in
plain Python instead of porting the uniflow workflow wrapper itself.

Usage:
    python -m michelangelo_examples.nomic_ai.pipelines.train
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "nomic-ai requires Python 3.11+; this interpreter is "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

import logging
import os

from michelangelo_examples.nomic_ai.pipelines.train.data import load_data
from michelangelo_examples.nomic_ai.pipelines.train.train import train

log = logging.getLogger(__name__)

_MODEL_NAME = "nomic-ai/nomic-bert-2048"


def main() -> dict:
    """Load a small WikiText sample, train Nomic BERT, and return the result."""
    train_data, validation_data, test_data = load_data(
        model_name=_MODEL_NAME,
        # Smaller than core's default (200/512) so the local smoke test
        # finishes quickly on CPU -- the subsequent 1% random sample in
        # load_data() still needs a large-enough pre-sample pool to leave a
        # non-empty split afterward.
        max_length=128,
        dataset_size=500,
    )

    result = train(train_data, validation_data, test_data, _MODEL_NAME)
    log.info("Training workflow result: %r", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Mirrors core's nomic_ai.py __main__ env-var workaround for Apple
    # Silicon MPS memory limits during local CPU/MPS training.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0")
    main()
