"""PyTorch Lightning regression workflow for California Housing price prediction.

Workflow entry point that orchestrates the full California Housing pipeline
via ``tabular_trainer``'s Lightning backend: feature preparation, Spark
preprocessing, derived-feature computation, a native-transform stage
(Scale/LogTransform/Clip), distributed Lightning training with Ray Train, an
assembler step that packages the trained model into deployable/raw Triton
packages, and a pusher step that exports the packaged model and preprocessed
datasets to storage and registry.
"""

from __future__ import annotations

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.uniflow.plugins.spark import SparkTask

from michelangelo_examples.california_housing.pipelines.libs.tasks.feature_prep import (
    feature_prep,
)
from michelangelo_examples.california_housing.pipelines.libs.tasks.preprocess import (
    PreprocessResult,
    preprocess,
)
from michelangelo_examples.california_housing.pipelines.pytorch_train.assembler import (
    assembler,
)
from michelangelo_examples.california_housing.pipelines.pytorch_train.derive_features import (
    derive_features,
)
from michelangelo_examples.california_housing.pipelines.pytorch_train.native_transform import (
    FEATURE_COLUMNS,
    native_transform,
)
from michelangelo_examples.california_housing.pipelines.pytorch_train.push import (
    push_step,
)
from michelangelo_examples.california_housing.pipelines.pytorch_train.train import train

__all__ = [
    "PreprocessResult",
    "assembler",
    "derive_features",
    "feature_prep",
    "native_transform",
    "preprocess",
    "push_step",
    "train",
    "train_workflow",
]

# California Housing features + target column order.
# MedHouseVal (the sklearn target) is renamed to "target" in feature_prep.


@uniflow.workflow()
def train_workflow(
    dataset_cols: str = (
        "MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,target"
    ),
):
    """End-to-end ML workflow: feature prep, native-transform, training, push.

    Orchestrates the full ML lifecycle for California Housing using
    ``tabular_trainer``'s Lightning backend: feature preparation,
    preprocessing with Spark, derived-feature computation, a native-transform
    stage (Scale/LogTransform/Clip, run unconditionally -- it is a core part
    of this example, not an opt-in extra), distributed Lightning training
    with Ray Train on the transform's output columns, and a pusher step that
    pushes the trained model and preprocessed datasets to storage and
    registry.

    Args:
        dataset_cols: Comma-separated string of column names including
            features and target.

    Returns:
        List of PusherResult from push_step, one per artifact pushed.
    """
    _dataset_cols = dataset_cols.split(",")
    feature_prep_overrides = feature_prep.with_overrides(
        alias="feature_prep_overrides",
        config=RayTask(
            head_cpu=2,
            worker_instances=1,
        ),
    )
    train_dv, validation_dv = feature_prep_overrides(
        columns=_dataset_cols,
    )
    pr = preprocess.with_overrides(
        alias="preprocess_overrides",
        config=SparkTask(executor_cpu=1, driver_cpu=1),
    )(
        cast_float_columns=_dataset_cols,
        train_dv=train_dv,
        validation_dv=validation_dv,
    )
    derived_train_dv, derived_validation_dv = derive_features(
        pr.train_data, pr.validation_data
    )
    native_tx_result = native_transform(derived_train_dv, derived_validation_dv)
    model_artifact = train(
        native_tx_result.transformed_datasets["train"],
        native_tx_result.transformed_datasets["validation"],
        feature_columns=FEATURE_COLUMNS,
    )
    assembled = assembler(model_artifact, native_tx_result.model)
    return push_step(pr, assembled)


if __name__ == "__main__":
    ctx = uniflow.create_context()

    ctx.environ["IMAGE_PULL_POLICY"] = "IfNotPresent"

    # Pass MINIO_* and REGISTRY_* via --environ flags on the command line
    # so values reach remote Ray workers (see README Remote Run section).

    ctx.run(train_workflow)
