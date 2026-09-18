"""Native-transform task for the California Housing native-transform pipeline.

Wraps ``tabular_native_transform()`` with a small two-level transform chain
over California Housing columns, mirroring the transform-DAG shape used by
Michelangelo's own native_transform integration tests: a few independent
Level 0 transforms, then a Level 1 transform chained onto one of Level 0's
outputs.
"""

from __future__ import annotations

import logging

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.workflow.schema.common import (
    IncrementalTrainingConfig,
    TrainingTypeConfig,
)
from michelangelo.workflow.schema.native_transform import (
    BatchOptions,
    TabularNativeTransformConfig,
)
from michelangelo.workflow.tasks.tabular_native_transform.task import (
    tabular_native_transform,
)
from michelangelo.workflow.variables import DatasetVariable
from michelangelo.workflow.variables.types import NativeTransformResult

log = logging.getLogger(__name__)

__all__ = ["FEATURE_COLUMNS", "native_transform"]

# Native-transform's output columns feed model.py's forward() directly --
# two raw/derived inputs plus their transformed counterparts, so the trained
# model is directly dependent on the transform pipeline having run
# correctly (a wrong or stale transform module would change these values,
# and therefore the model's predictions).
FEATURE_COLUMNS = [
    "derived_house_age_log",
    "derived_population_clipped",
    "AveRooms",
    "AveRooms_scaled",
]

_TRANSFORM_SPEC = {
    "transform_specs": [
        # --- Level 0 (independent, no inter-layer dependencies) ---
        {
            "transform_name": "Scale",
            "input_cols": ["AveRooms"],
            "output_cols": ["AveRooms_scaled"],
            "factor": 0.1,
        },
        {
            "transform_name": "LogTransform",
            "input_cols": ["derived_house_age"],
            "output_cols": ["derived_house_age_log"],
        },
        {
            "transform_name": "Clip",
            "input_cols": ["derived_population"],
            "output_cols": ["derived_population_clipped"],
            "min_value": 0.0,
            "max_value": 5000.0,
        },
        # --- Level 1 (chained: consumes Level 0's LogTransform output) ---
        {
            "transform_name": "Scale",
            "input_cols": ["derived_house_age_log"],
            "output_cols": ["derived_house_age_log_scaled"],
            "factor": 0.5,
        },
    ]
}


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_gpu=0,
        head_memory="4Gi",
        worker_cpu=1,
        worker_gpu=0,
        worker_memory="4Gi",
        worker_instances=0,
    ),
)
def native_transform(
    train_dv: DatasetVariable,
    validation_dv: DatasetVariable,
) -> NativeTransformResult:
    """Apply the native-transform chain to the train/validation datasets.

    Args:
        train_dv: Training dataset with ``derive_features``'s derived
            columns already present.
        validation_dv: Validation dataset with ``derive_features``'s derived
            columns already present.

    Returns:
        ``NativeTransformResult`` with the transformed datasets (keyed
        ``"train"``/``"validation"``) and the fitted transform module,
        wrapped as a ``ModelVariable``.
    """
    config = TabularNativeTransformConfig(
        transform_spec=_TRANSFORM_SPEC,
        batch_options=BatchOptions(batch_size=1000, concurrency=2),
        incremental_training=IncrementalTrainingConfig(
            training_type=TrainingTypeConfig.BASE
        ),
    )
    result = tabular_native_transform(
        config, {"train": train_dv, "validation": validation_dv}
    )
    log.info(
        "Native transform output schema: %s",
        result.transformed_datasets["train"].value.schema(),
    )
    return result
