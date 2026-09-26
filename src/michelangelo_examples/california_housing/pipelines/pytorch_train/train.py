"""Lightning training task for the California Housing Lightning workflow.

Trains ``NativeTxRegressionModel`` on ``native_transform.py``'s output
columns via ``tabular_trainer``'s ``train_tabular()`` (Ray Train + Lightning
backend). Training successfully on these columns is itself a check that the
native-transform stage ran and loaded correctly -- a wrong or stale
transform module would change their values, and therefore the model's
predictions/loss.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.workflow.schema.tabular_trainer import (
    BatchIterConfig,
    ColumnConfig,
    DataloadingConfig,
    LightningTrainerConfig,
    LightningTrainerKwargs,
    ScalingConfig,
    TabularTrainerConfig,
)
from michelangelo.workflow.tasks.tabular_trainer.task import train_tabular

if TYPE_CHECKING:
    from michelangelo.workflow.variables import DatasetVariable, ModelVariable

log = logging.getLogger(__name__)

__all__ = ["train"]

LABEL_COLUMN = "target"


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_gpu=0,
        head_memory="4Gi",
        worker_cpu=1,
        worker_gpu=0,
        worker_memory="4Gi",
        worker_instances=2,
    ),
)
def train(
    train_data: DatasetVariable,
    validation_data: DatasetVariable,
    feature_columns: list[str],
) -> ModelVariable:
    """Train a Lightning regression model using Ray Train.

    Args:
        train_data: Training dataset -- ``native_transform``'s transformed
            "train" dataset, carrying both the original preprocessed columns
            (label included) and the transform's new output columns.
        validation_data: Validation dataset, same shape as ``train_data``.
        feature_columns: Ordered list of feature column names (excludes the
            label column) -- expected to be
            ``native_transform.FEATURE_COLUMNS``.

    Returns:
        ModelVariable wrapping the trained model as an intra-pipeline
        intermediate. ``train_tabular()`` persists it under
        ``UF_STORAGE_URL`` and, for its own distributed checkpointing on a
        multi-node cluster, defaults ``run_config`` to the same location via
        ``michelangelo.uniflow.plugins.ray.run_config.create_run_config()`` --
        no manual storage-backend or RunConfig plumbing needed here. The
        ``assembler`` task turns this into a registry-ready
        ``AssembledModel`` before ``push_step`` runs.
    """
    config = TabularTrainerConfig(
        lightning=LightningTrainerConfig(
            model_class=(
                "michelangelo_examples.california_housing.pipelines.pytorch_train.model."
                "NativeTxRegressionModel"
            ),
            model_kwargs={
                "feature_columns": feature_columns,
                "label_column": LABEL_COLUMN,
            },
            input_columns={c: ColumnConfig("torch.float32") for c in feature_columns},
            # shape=[1], not the scalar default: NativeTxRegressionModel's
            # final nn.Linear(len(feature_columns), 1) layer produces a
            # genuine (batch, 1) tensor -- a real one-element non-batch
            # dimension -- not a true 0-D-per-sample scalar, so the schema
            # must declare that dimension rather than default to an empty
            # (true-scalar) shape.
            output_columns={"prediction": ColumnConfig("torch.float32", [1])},
            labels={LABEL_COLUMN: ColumnConfig("torch.float32")},
            metadata_columns=[],
            scaling_config=ScalingConfig(cpu_per_worker=1),
            dataloading_config=DataloadingConfig(
                batch_iter_config=BatchIterConfig(batch_size=64, num_shuffle_batches=1)
            ),
            # precision explicitly forced to "32" rather than relying on the
            # dispatcher's "bf16-mixed" default: verified locally that
            # bf16-mixed does not error on CPU (real torch.autocast('cpu', ...)
            # AMP, not a silent fallback), but on x86 CPUs without
            # AVX512_BF16 it runs via slower software emulation. This example
            # prioritizes fast, deterministic CI/sandbox runs over AMP.
            lightning_trainer_kwargs=LightningTrainerKwargs(
                max_epochs=5, precision="32"
            ),
        )
    )

    return train_tabular(config, train_data, validation_data)
