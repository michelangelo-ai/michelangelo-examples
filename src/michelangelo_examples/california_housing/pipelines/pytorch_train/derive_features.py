"""Derived-feature task for the California Housing Lightning workflow.

Computes the handful of simple derived columns that feed
``tabular_native_transform``'s scale/log/clip chain (see ``native_transform.py``).
Internal Michelangelo pipelines typically compute derived columns like these
via a Spark-DSL "tabular_transform" step; no OSS equivalent of that DSL
engine exists yet (it's a small string-expression interpreter, not something
``michelangelo`` itself needs to expose for a two-column example), so this
task hand-computes them with plain pandas instead.
"""

from __future__ import annotations

import logging

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.workflow.variables import DatasetVariable

log = logging.getLogger(__name__)

__all__ = ["derive_features"]


def _add_derived_columns(batch):
    """Add ``derived_house_age`` and ``derived_population`` to a pandas batch.

    Both derivations guard against nulls the way a real feature-engineering
    step would, even though the California Housing dataset itself has none —
    ``fillna(0)`` keeps the behavior well-defined if this example is ever
    pointed at messier data.

    Args:
        batch: A pandas ``DataFrame`` batch with ``HouseAge`` and
            ``Population`` columns.

    Returns:
        The same batch with two new columns appended.
    """
    batch["derived_house_age"] = batch["HouseAge"].fillna(0) + 1
    batch["derived_population"] = batch["Population"].fillna(0)
    return batch


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_gpu=0,
        head_memory="2Gi",
        worker_instances=0,
    ),
    cache_enabled=False,  # off for tutorial simplicity; enable in production
)
def derive_features(
    train_dv: DatasetVariable,
    validation_dv: DatasetVariable,
) -> tuple[DatasetVariable, DatasetVariable]:
    """Add the derived columns ``tabular_native_transform`` consumes.

    ``derived_house_age`` (``HouseAge + 1``, guarding against a zero input to
    the downstream ``LogTransform``) and ``derived_population`` (a
    null-guarded pass-through of ``Population``, the downstream ``Clip``
    layer's input) are computed here rather than inline in
    ``native_transform.py``, mirroring how a real pipeline separates feature
    derivation from the native-transform stage itself.

    Args:
        train_dv: Training dataset from ``feature_prep``.
        validation_dv: Validation dataset from ``feature_prep``.

    Returns:
        Tuple of (train_dataset, validation_dataset) with the two derived
        columns added, as new ``DatasetVariable``s.
    """
    train_dv.load_ray_dataset()
    validation_dv.load_ray_dataset()

    train_data = train_dv.value.map_batches(_add_derived_columns, batch_format="pandas")
    validation_data = validation_dv.value.map_batches(
        _add_derived_columns, batch_format="pandas"
    )

    out_train = DatasetVariable.create(train_data)
    out_train.save_ray_dataset()

    out_validation = DatasetVariable.create(validation_data)
    out_validation.save_ray_dataset()

    log.info("Derived-feature dataset schema: %s", train_data.schema())

    return out_train, out_validation
