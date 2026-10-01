"""Training task for Nomic BERT models.

Ported from michelangelo core's ``python/examples/nomic_ai/train.py``, with
the ``@uniflow.task`` decorator and its ``RayTask`` resource config removed
(this project ships ``__main__.py``-only, see this project's own README) and
the model import repointed from core's ``examples.nomic_ai.model`` to this
project's own ``model.py``.

Implements distributed training using PyTorch Lightning with DeepSpeed
strategy for efficient multi-GPU training.
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DeepSpeedStrategy
from ray.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from michelangelo_examples.nomic_ai.pipelines.train.model import (
    HuggingFaceLightningModel,
)

log = logging.getLogger(__name__)


def train(
    train_data: Dataset,
    validation_data: Dataset,
    test_data: Dataset,
    model_name: str = "nomic-ai/nomic-bert-2048",
) -> dict:
    """Train Nomic BERT model using PyTorch Lightning.

    Trains the model with DeepSpeed strategy if GPUs are available,
    otherwise uses CPU training.

    Args:
        train_data: Ray Dataset containing training data.
        validation_data: Ray Dataset containing validation data.
        test_data: Ray Dataset containing test data. Unused -- ported as-is
            from core, which accepts but never consumes this argument.
        model_name: HuggingFace model identifier. Defaults to
            "nomic-ai/nomic-bert-2048".

    Returns:
        Dictionary with training status.
    """
    del test_data
    log.info("Starting training...")

    # Training configuration
    output_dir = "./nomic_ai"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class RayDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, ray_dataset):
            self.data = ray_dataset.take_all()

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            return {
                "input_ids": torch.tensor(item["input_ids"]),
                "attention_mask": torch.tensor(item["attention_mask"]),
            }

    train_dataloader = DataLoader(
        RayDatasetWrapper(train_data), batch_size=8, shuffle=True
    )
    val_dataloader = DataLoader(RayDatasetWrapper(validation_data), batch_size=8)

    model = HuggingFaceLightningModel(model_name)

    use_deepspeed = torch.cuda.is_available()
    strategy = DeepSpeedStrategy(stage=2) if use_deepspeed else "auto"

    trainer = pl.Trainer(
        max_epochs=1,
        precision=16 if torch.cuda.is_available() else 32,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy=strategy,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {"status": "Training completed successfully"}
