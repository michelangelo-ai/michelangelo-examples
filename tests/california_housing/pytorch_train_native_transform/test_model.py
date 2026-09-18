"""Unit tests for ``NativeTxRegressionModel``.

Exercises the model directly with synthetic tensors -- no Ray, no real
``tabular_native_transform`` run required, since the model only needs
dict/tensor batches shaped like ``native_transform.FEATURE_COLUMNS``.
"""

from __future__ import annotations

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from michelangelo_examples.california_housing.pipelines.pytorch_train_native_transform.model import (
    NativeTxRegressionModel,
)
from michelangelo_examples.california_housing.pipelines.pytorch_train_native_transform.native_transform import (
    FEATURE_COLUMNS,
)


def _make_model() -> NativeTxRegressionModel:
    return NativeTxRegressionModel(
        feature_columns=FEATURE_COLUMNS, label_column="target", learning_rate=1e-2
    )


def _make_batch(n: int) -> dict[str, torch.Tensor]:
    batch = {c: torch.randn(n) for c in FEATURE_COLUMNS}
    batch["target"] = torch.randn(n)
    return batch


def test_forward_with_tensor_input():
    model = _make_model()
    x = torch.randn(4, len(FEATURE_COLUMNS))

    out = model(x)

    assert out.shape == (4, 1)


def test_forward_with_dict_input_matches_tensor_input():
    model = _make_model()
    batch = _make_batch(4)
    tensor_x = torch.stack([batch[c].float() for c in FEATURE_COLUMNS], dim=1)

    out_from_dict = model(batch)
    out_from_tensor = model(tensor_x)

    torch.testing.assert_close(out_from_dict, out_from_tensor)


def test_training_step_returns_scalar_loss():
    model = _make_model()
    batch = _make_batch(8)

    loss = model.training_step(batch, 0)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_validation_step_returns_scalar_loss():
    model = _make_model()
    batch = _make_batch(8)

    loss = model.validation_step(batch, 0)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_configure_optimizers_uses_configured_learning_rate():
    model = _make_model()

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 1e-2


def test_trains_end_to_end_and_scripts_after_fit():
    model = _make_model()
    n = 16
    tensors = [torch.randn(n) for _ in FEATURE_COLUMNS] + [torch.randn(n)]
    dataset = TensorDataset(*tensors)

    def collate(batch):
        columns = list(zip(*batch))
        out = {c: torch.stack(columns[i]) for i, c in enumerate(FEATURE_COLUMNS)}
        out["target"] = torch.stack(columns[-1])
        return out

    loader = DataLoader(dataset, batch_size=8, collate_fn=collate)
    trainer = Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
    )

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

    # Only scriptable once a Trainer is attached -- LightningModule.trainer
    # raises otherwise, and forward() is written to jit-script cleanly for
    # the packaging path once it is.
    scripted = torch.jit.script(model)
    out = scripted(torch.randn(4, len(FEATURE_COLUMNS)))
    assert out.shape == (4, 1)
