"""PyTorch Lightning regression model for the native-transform pipeline.

Unlike ``pytorch_train/model.py``'s ``TorchRegressionModel`` (which consumes
raw dataset columns), this model consumes ``native_transform.py``'s output
columns directly -- a mix of one raw passthrough column and three
transform-produced columns. A wrong or stale transform module changes these
values (and therefore the model's predictions/loss), so training on them
successfully is itself a check that the native-transform stage ran and
loaded correctly.
"""

from __future__ import annotations

from typing import Dict, Union  # noqa: UP035 -- see forward()'s docstring

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn

__all__ = ["NativeTxRegressionModel"]


class NativeTxRegressionModel(LightningModule):
    """Linear regressor trained directly on ``tabular_native_transform``'s outputs.

    Structured like ``pytorch_train/model.py``'s ``TorchRegressionModel``
    (dict-batch training/validation steps, a jit-scriptable ``forward()``),
    but with a single ``nn.Linear`` layer rather than an MLP -- this example
    is about exercising the native-transform integration, not model
    capacity.

    Attributes:
        feature_columns: Ordered list of input feature column names --
            expected to be ``native_transform.FEATURE_COLUMNS`` or a subset.
        label_column: Name of the regression target column.
        learning_rate: Adam learning rate.

    Example:
        >>> model = NativeTxRegressionModel(
        ...     feature_columns=["derived_house_age_log", "AveRooms"],
        ...     label_column="target",
        ... )
    """

    def __init__(
        self,
        feature_columns: list[str],
        label_column: str,
        learning_rate: float = 1e-3,
    ):
        """Build the linear layer and store constructor args as hyperparameters."""
        super().__init__()
        self.save_hyperparameters()
        # Plain, typed instance attribute -- not read via self.hparams.
        # torch.jit.script (used by the Triton packager's deployable-package
        # conversion) can't see Lightning's dynamically-attached `hparams`
        # object, so any attribute forward() depends on must be a regular
        # module attribute instead.
        self.feature_columns: list[str] = feature_columns
        self.linear = nn.Linear(len(feature_columns), 1)

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],  # noqa: UP006, UP007
    ) -> torch.Tensor:
        """Return the model's scalar prediction for a batch of feature vectors.

        Accepts either a pre-stacked feature tensor (what ``training_step``/
        ``validation_step`` pass, via ``_assemble_batch``) or a raw dict of
        per-feature tensors keyed by feature name -- the shape the Triton
        packager's validation/serving path invokes the model with.

        The type annotation uses typing.Dict/Union (not PEP 585/604 syntax)
        and the branch below checks for ``torch.Tensor`` rather than
        ``dict`` -- both are required for ``torch.jit.script`` to
        type-refine a ``Union`` parameter correctly.
        """
        if isinstance(x, torch.Tensor):
            return self.linear(x)
        x = torch.stack([x[c].float() for c in self.feature_columns], dim=1)
        return self.linear(x)

    def _assemble_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack the configured feature columns and extract the label column."""
        x = torch.stack([batch[c].float() for c in self.feature_columns], dim=1)
        y = batch[self.hparams.label_column].float().view(-1, 1)
        return x, y

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute and log the MSE training loss for one batch."""
        x, y = self._assemble_batch(batch)
        loss = F.mse_loss(self(x), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute and log the MSE validation loss for one batch."""
        x, y = self._assemble_batch(batch)
        loss = F.mse_loss(self(x), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the Adam optimizer configured with ``learning_rate``."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
